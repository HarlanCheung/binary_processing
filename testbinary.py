import time
import numpy as np
import IO.IO as io
import ImageProcessing.Smoothing as sm

input_tif = "/Users/harlan/Documents/binarysmoothing/ssp_test_lcc.tif"
output_tif = "/Users/harlan/Documents/binarysmoothing/ssp_test_lcc_smoothed.tif"

def main():
    t0 = time.perf_counter()
    print("[1/5] 读取 TIFF:", input_tif, flush=True)
    vol = io.read(input_tif)
    print("    读取完成，形状:", vol.shape, "dtype:", vol.dtype, "耗时: %.2fs" % (time.perf_counter() - t0), flush=True)

    t1 = time.perf_counter()
    print("[2/5] 阈值二值化...", flush=True)
    binary = vol > 0
    print("    二值化完成，耗时: %.2fs" % (time.perf_counter() - t1), flush=True)

    # 先单线程生成查找表，避免 mp.spawn 报错
    t_lut = time.perf_counter()
    print("[3/5] 生成/加载查找表 (serial)...", flush=True)
    sm.initialize_lookup_table(verbose=True, processes='serial')
    print("    查找表准备好，耗时: %.2fs" % (time.perf_counter() - t_lut), flush=True)

    t2 = time.perf_counter()
    print("[4/5] 拓扑平滑开始 (iterations=2, processes='serial')", flush=True)
    smoothed = sm.smooth_by_configuration(binary, iterations=2, processes='serial', verbose=True)
    print("    平滑完成，耗时: %.2fs" % (time.perf_counter() - t2), flush=True)

    t3 = time.perf_counter()
    print("[5/5] 写出 TIFF:", output_tif, flush=True)
    io.write(output_tif, smoothed.astype(np.uint8))
    print("    写出完成，耗时: %.2fs" % (time.perf_counter() - t3), flush=True)
    print("全流程耗时: %.2fs" % (time.perf_counter() - t0), flush=True)

if __name__ == "__main__":
    main()
