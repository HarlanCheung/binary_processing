import argparse
import gc
import time
from pathlib import Path

import numpy as np

import IO.IO as io
import ParallelProcessing.DataProcessing.ConvolvePointList as cpl
import ImageProcessing.skeletonization.PK12 as pk12


def parse_args():
    p = argparse.ArgumentParser(description="PK12 3D skeletonization for binary TIFF volumes")
    p.add_argument("input_tif", help="Path to the input binary TIFF stack")
    p.add_argument("output_tif", help="Path to write the skeletonized TIFF stack")
    p.add_argument("--steps", type=int, default=None, help="Max iteration steps (default: full thinning)")
    p.add_argument(
        "--method",
        choices=["PK12", "PK12i"],
        default="PK12i",
        help="PK12 variant; PK12i uses index-based version (faster)",
    )
    p.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Thread count for the Cython convolutions (default: use all available cores)",
    )
    p.add_argument(
        "--delete-border",
        action="store_true",
        help="Drop any foreground voxels touching the volume border before thinning",
    )
    p.add_argument(
        "--skip-border-check",
        action="store_true",
        help="Skip the empty-border check (PK12 requires an empty border unless delete-border is used)",
    )
    return p.parse_args()


def configure_convolution_threads(processes: int | None):
    """Override ConvolvePointList thread defaults so PK12 uses the requested value."""
    if processes is None:
        return
    if processes < 1:
        raise ValueError("processes must be >= 1")
    for fn in (
        cpl.convolve_3d,
        cpl.convolve_3d_points,
        cpl.convolve_3d_xyz,
        cpl.convolve_3d_indices,
        cpl.convolve_3d_indices_if_smaller_than,
    ):
        defaults = list(fn.__defaults__)
        defaults[-1] = processes
        fn.__defaults__ = tuple(defaults)


def main():
    args = parse_args()
    t0 = time.perf_counter()

    configure_convolution_threads(args.processes)

    print("[1/3] 读取 TIFF:", args.input_tif, flush=True)
    vol = io.read(args.input_tif)
    print("    形状:", vol.shape, "dtype:", vol.dtype, flush=True)

    print("[2/3] 转换为 Fortran-order 二值数组", flush=True)
    binary = np.asarray(vol > 0, dtype=bool, order="F")
    del vol
    gc.collect()

    print(f"[3/3] PK12 细化开始 (method={args.method}, steps={args.steps}, processes={args.processes or 'auto'})", flush=True)
    skeleton_fn = pk12.skeletonize if args.method == "PK12" else pk12.skeletonize_index
    skeleton = skeleton_fn(
        binary,
        steps=args.steps,
        check_border=not args.skip_border_check,
        delete_border=args.delete_border,
        verbose=True,
    )

    output_path = Path(args.output_tif)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io.write(str(output_path), np.asarray(skeleton, dtype=np.uint8))
    print("完成，输出:", output_path, "总耗时: %.2fs" % (time.perf_counter() - t0), flush=True)


if __name__ == "__main__":
    main()
