import time
import argparse
import os
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

    # -----------------------------------------------------------
    # 🔥 一定要提前创建输出目录，否则 MMP 创建会失败！
    # -----------------------------------------------------------
    output_dir = Path(output_tif).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_input_path = str(output_dir / "temp_input_binary.npy")
    temp_output_path = str(output_dir / "temp_output_smooth.npy")

    # 进程配置
    processes_param = None if args.processes <= 1 else args.processes

    t0 = time.perf_counter()
    print("[1/6] 读取 TIFF:", input_tif, flush=True)
    vol = io.read(input_tif)
    print("    读取完成，形状:", vol.shape, "dtype:", vol.dtype, "耗时: %.2fs" % (time.perf_counter() - t0), flush=True)

    try:
        t1 = time.perf_counter()
        print("[2/6] 转换为内存映射 (MMP)...", flush=True)

        # -----------------------------------------------------------
        # 🔥 必须转换为 uint8 避免 numpy bool memmap 的 header bug
        # -----------------------------------------------------------
        binary_vol = (vol > 0).astype(np.uint8)

        del vol
        gc.collect()

        print(f"    创建输入 MMP: {temp_input_path}", flush=True)
        source_mmp = mmp.create(
            location=temp_input_path,
            array=binary_vol,
            dtype=np.uint8,
            shape=binary_vol.shape,
            order="C",     # 🔥 强制 C-order，避免 Fortran-order mismatch
        )

        print(f"    创建输出 MMP: {temp_output_path}", flush=True)
        sink_mmp = mmp.create(
            location=temp_output_path,
            shape=binary_vol.shape,
            dtype=np.uint8,
            order="C",
        )

        print("    MMP 创建完成，耗时: %.2fs" % (time.perf_counter() - t1), flush=True)

        del binary_vol
        gc.collect()

        # -----------------------------------------------------------
        # 生成查找表（可能并行）
        # -----------------------------------------------------------
        t_lut = time.perf_counter()
        print(f"[3/6] 生成/加载查找表 (processes={processes_param})...", flush=True)
        sm.initialize_lookup_table(verbose=True, processes=processes_param)
        print("    查找表准备好，耗时: %.2fs" % (time.perf_counter() - t_lut), flush=True)

        # -----------------------------------------------------------
        # 分块参数（多进程）
        # -----------------------------------------------------------
        processing_parameter = {}
        if args.processes > 1:
            max_block_size = calculate_blockshape_by_processes(source_mmp.shape, args.processes)
            processing_parameter = {
                "size_max": max_block_size,
                "axes": [0, 1, 2],
                "optimization": False,
                "as_memory": False
            }
            print(f"    [自动分块] 进程数: {args.processes}, size_max: {max_block_size}", flush=True)

        # -----------------------------------------------------------
        # 进行拓扑平滑
        # -----------------------------------------------------------
        t2 = time.perf_counter()
        print(f"[4/6] 拓扑平滑开始 (iterations={args.iterations}, processes={processes_param})", flush=True)

        sm.smooth_by_configuration(
            source_mmp,
            sink=sink_mmp,
            iterations=args.iterations,
            processes=processes_param,
            processing_parameter=processing_parameter,
            verbose=True
        )

        print("    平滑完成，耗时: %.2fs" % (time.perf_counter() - t2), flush=True)

        # -----------------------------------------------------------
        # 转换为 TIFF 输出
        # -----------------------------------------------------------
        t3 = time.perf_counter()
        print("[5/6] 写出最终 TIFF:", output_tif, flush=True)

        io.write(output_tif, sink_mmp.array.astype(np.uint8))
        print("    写出完成，耗时: %.2fs" % (time.perf_counter() - t3), flush=True)

    finally:
        print("[6/6] 清理临时文件...", flush=True)

        # -----------------------------------------------------------
        # 🔥 不删除 source_mmp/sink_mmp 对象（避免文件提前关闭）
        # -----------------------------------------------------------

        for p in [temp_input_path, temp_output_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass

        gc.collect()

    print("全流程耗时: %.2fs" % (time.perf_counter() - t0), flush=True)


if __name__ == "__main__":
    main()