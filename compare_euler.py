import argparse
import numpy as np
import IO.IO as io
from skimage.measure import euler_number  # 需安装 scikit-image

def parse_args():
    p = argparse.ArgumentParser(description="比较两幅二值体数据的欧拉示性数")
    p.add_argument("image_a", help="第一幅 TIFF 路径")
    p.add_argument("image_b", help="第二幅 TIFF 路径")
    p.add_argument(
        "--connectivity",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="邻域连通性(1=6邻域/最小连通，2=18邻域，3=26邻域，默认为1)",
    )
    return p.parse_args()

def load_bool(path):
    arr = io.read(path)
    return np.asarray(arr > 0, dtype=bool, order="F")

def summarize(path, connectivity):
    arr = load_bool(path)
    chi = euler_number(arr, connectivity=connectivity)
    voxels = int(arr.sum())
    return voxels, chi

def main():
    args = parse_args()
    print(f"[1/2] 读取并计算: {args.image_a}", flush=True)
    vox_a, chi_a = summarize(args.image_a, args.connectivity)
    print(f"[2/2] 读取并计算: {args.image_b}", flush=True)
    vox_b, chi_b = summarize(args.image_b, args.connectivity)

    print("\n=== 结果 ===")
    print(f"A: voxels={vox_a}, Euler={chi_a}")
    print(f"B: voxels={vox_b}, Euler={chi_b}")
    print(f"差值: voxels(B-A)={vox_b - vox_a}, Euler(B-A)={chi_b - chi_a}")

if __name__ == "__main__":
    main()
