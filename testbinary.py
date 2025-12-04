import time
import argparse
import numpy as np
import IO.IO as io
import ImageProcessing.Smoothing as sm

def parse_args():
    parser = argparse.ArgumentParser(description="二值体数据的拓扑平滑处理")
    parser.add_argument("input_tif", help="输入 TIFF 文件路径")
    parser.add_argument("output_tif", help="输出 TIFF 文件路径")
    parser.add_argument("--iterations", type=int, default=2, help="平滑迭代次数 (默认: 2)")
    parser.add_argument("--processes", choices=["serial", "auto"], default="serial", help="并行模式 (默认: serial)")
    return parser.parse_args()

def main():
    args = parse_args()
    input_tif = args.input_tif
    output_tif = args.output_tif

    t0 = time.perf_counter()
    print("[1/5] 读取 TIFF:", input_tif, flush=True)
    vol = io.read(input_tif)
    print("    读取完成，形状:", vol.shape, "dtype:", vol.dtype, "耗时: %.2fs" % (time.perf_counter() - t0), flush=True)

    t1 = time.perf_counter()
    print("[2/5] 阈值二值化...", flush=True)
    binary = vol > 0
    print("    二值化完成，耗时: %.2fs" % (time.perf_counter() - t1), flush=True)

    t_lut = time.perf_counter()
    print("[3/5] 生成/加载查找表 (%s)..." % args.processes, flush=True)
    sm.initialize_lookup_table(verbose=True, processes=args.processes)
    print("    查找表准备好，耗时: %.2fs" % (time.perf_counter() - t_lut), flush=True)

    t2 = time.perf_counter()
    print("[4/5] 拓扑平滑开始 (iterations=%d, processes='%s')" % (args.iterations, args.processes), flush=True)
    smoothed = sm.smooth_by_configuration(binary, iterations=args.iterations, processes=args.processes, verbose=True)
    print("    平滑完成，耗时: %.2fs" % (time.perf_counter() - t2), flush=True)

    t3 = time.perf_counter()
    print("[5/5] 写出 TIFF:", output_tif, flush=True)
    io.write(output_tif, smoothed.astype(np.uint8))
    print("    写出完成，耗时: %.2fs" % (time.perf_counter() - t3), flush=True)
    print("全流程耗时: %.2fs" % (time.perf_counter() - t0), flush=True)

if __name__ == "__main__":
    main()