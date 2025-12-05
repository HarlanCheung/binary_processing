import time
import argparse
import gc
from pathlib import Path
import numpy as np
import IO.IO as io
import IO.MMP as mmp
import ImageProcessing.Smoothing as sm


def calculate_blockshape_by_processes(full_shape, num_processes):
    """根据进程数计算合适的块大小 (size_max)。"""
    if num_processes <= 1:
        return full_shape

    total_voxels = np.prod(full_shape)
    target_voxels_per_block = total_voxels / num_processes
    current_shape = list(full_shape)

    while np.prod(current_shape) > target_voxels_per_block:
        longest_axis = np.argmax(current_shape)
        current_shape[longest_axis] = max(16, current_shape[longest_axis] // 2)

    return tuple(current_shape)


def parse_args():
    parser = argparse.ArgumentParser(description="二值体数据的拓扑平滑处理")
    parser.add_argument("input_tif", help="输入 TIFF 文件路径")
    parser.add_argument("output_tif", help="输出 TIFF 文件路径")
    parser.add_argument("--iterations", type=int, default=2, help="平滑迭代次数 (默认: 2)")
    parser.add_argument("--processes", type=int, default=1, help="并行进程数 (默认: 1, 即串行)")
    return parser.parse_args()


def main():
    args = parse_args()
    input_tif = args.input_tif
    output_tif = args.output_tif
    output_path = Path(output_tif)

    # 进程配置
    processes_param = None if args.processes <= 1 else args.processes

    t0 = time.perf_counter()
    print("[1/5] 读取 TIFF 到内存:", input_tif, flush=True)
    vol = io.read(input_tif)
    print("    读取完成，形状:", vol.shape, "dtype:", vol.dtype, "耗时: %.2fs" % (time.perf_counter() - t0), flush=True)

    # 转换为 bool (内存操作)
    print("[2/5] 转换为二值数据...", flush=True)
    source_array = (vol > 0)
    del vol
    gc.collect()

    # 使用 memmap 保存二值数据，避免后续进程间拷贝
    output_path.parent.mkdir(parents=True, exist_ok=True)
    memmap_source_path = output_path.with_name(output_path.name + ".source.mmp.npy")
    print(f"    创建/覆盖 memmap 源文件: {memmap_source_path}", flush=True)
    source_mmp = mmp.create(
        location=str(memmap_source_path),
        shape=source_array.shape,
        dtype=bool,
        order='F'
    )
    source_mmp[:] = source_array.astype(bool, copy=False)
    del source_array
    gc.collect()

    # -----------------------------------------------------------
    # 生成查找表
    # -----------------------------------------------------------
    t_lut = time.perf_counter()
    print(f"[3/5] 生成/加载查找表 (processes={processes_param})...", flush=True)
    sm.initialize_lookup_table(verbose=True, processes=processes_param)
    print("    查找表准备好，耗时: %.2fs" % (time.perf_counter() - t_lut), flush=True)

    # -----------------------------------------------------------
    # 分块参数
    # -----------------------------------------------------------
    processing_parameter = {}
    if args.processes > 1:
        max_block_size = calculate_blockshape_by_processes(source_mmp.shape, args.processes)
        processing_parameter = {
            "size_max": max_block_size,
            "axes": [0, 1, 2],
            "optimization": False,
            # memmap 支持按块加载，不需要额外内存开销
            "as_memory": True
        }
        print(f"    [自动分块] 进程数: {args.processes}, size_max: {max_block_size}", flush=True)

    # -----------------------------------------------------------
    # 进行拓扑平滑
    # -----------------------------------------------------------
    t2 = time.perf_counter()
    print(f"[4/5] 拓扑平滑开始 (iterations={args.iterations}, processes={processes_param})", flush=True)

    # 输出 memmap，平滑结果直接写入，不占用额外内存
    memmap_result_path = output_path.with_name(output_path.name + ".smooth.mmp.npy")
    print(f"    创建/覆盖 memmap 结果文件: {memmap_result_path}", flush=True)
    result_sink = mmp.create(
        location=str(memmap_result_path),
        shape=source_mmp.shape,
        dtype=bool,
        order=source_mmp.order
    )

    # 直接传入 memmap，子进程从磁盘按块读取，避免 pickling 大数组
    result = sm.smooth_by_configuration(
        source_mmp,
        sink=result_sink,
        iterations=args.iterations,
        processes=processes_param,
        processing_parameter=processing_parameter,
        verbose=True
    )
    
    # sm.smooth_by_configuration 返回的可能是 Source 对象或 array
    if hasattr(result, 'array'):
        result_array = result.array
    else:
        result_array = result

    print("    平滑完成，耗时: %.2fs" % (time.perf_counter() - t2), flush=True)

    # -----------------------------------------------------------
    # 转换为 TIFF 输出
    # -----------------------------------------------------------
    t3 = time.perf_counter()
    print("[5/5] 写出最终 TIFF:", output_tif, flush=True)

    # 确保输出目录存在
    Path(output_tif).parent.mkdir(parents=True, exist_ok=True)
    
    # 转换为普通 ndarray 再写出，避免把 memmap 当成 MMP 源导致写入路径为空
    result_uint8 = np.asarray(result_array, dtype=np.uint8)
    io.write(output_tif, result_uint8)
    print("    写出完成，耗时: %.2fs" % (time.perf_counter() - t3), flush=True)
    
    print("全流程耗时: %.2fs" % (time.perf_counter() - t0), flush=True)

if __name__ == "__main__":
    main()
