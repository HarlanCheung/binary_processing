import argparse
import numpy as np
import scipy.ndimage as ndi
import IO.IO as io

def parse_args():
    p = argparse.ArgumentParser(description="比较两幅二值体的内部空洞数量与体素体积")
    p.add_argument("image_a", help="第一幅二值体 TIFF 路径")
    p.add_argument("image_b", help="第二幅二值体 TIFF 路径")
    p.add_argument(
        "--connectivity",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="背景连通性：1=6邻域, 2=18邻域, 3=26邻域 (默认 1，保守判定空洞)",
    )
    return p.parse_args()

def count_cavities(binary: np.ndarray, connectivity: int):
    """返回空洞个数与空洞体素总数"""
    binary = np.asarray(binary, dtype=bool)
    background = ~binary
    structure = ndi.generate_binary_structure(3, connectivity)
    labels, n = ndi.label(background, structure=structure)
    if n == 0:
        return 0, 0

    # 找到接触边界的背景标签
    touching = set()
    slices = [
        labels[0, :, :],
        labels[-1, :, :],
        labels[:, 0, :],
        labels[:, -1, :],
        labels[:, :, 0],
        labels[:, :, -1],
    ]
    for s in slices:
        touching.update(np.unique(s))

    # 空洞 = 未接触边界的背景分量
    all_ids = np.arange(1, n + 1)
    cavity_ids = np.setdiff1d(all_ids, list(touching), assume_unique=False)
    if cavity_ids.size == 0:
        return 0, 0

    counts = np.bincount(labels.ravel())
    voxel_total = int(counts[cavity_ids].sum())
    return int(cavity_ids.size), voxel_total

def summarize(path, connectivity):
    arr = io.read(path)
    binary = np.asarray(arr > 0, dtype=bool, order="F")
    voxels_fg = int(binary.sum())
    cavities, cav_voxels = count_cavities(binary, connectivity)
    return {
        "foreground_voxels": voxels_fg,
        "cavity_count": cavities,
        "cavity_voxels": cav_voxels,
    }

def main():
    args = parse_args()
    print(f"[1/2] 读取并分析: {args.image_a}", flush=True)
    a = summarize(args.image_a, args.connectivity)
    print(f"[2/2] 读取并分析: {args.image_b}", flush=True)
    b = summarize(args.image_b, args.connectivity)

    print("\n=== 结果 ===")
    print(f"Raw: fg_voxels={a['foreground_voxels']}, cavities={a['cavity_count']}, cavity_voxels={a['cavity_voxels']}")
    print(f"Processed: fg_voxels={b['foreground_voxels']}, cavities={b['cavity_count']}, cavity_voxels={b['cavity_voxels']}")
    print("差值 (P - R):")
    print(f"  fg_voxels={b['foreground_voxels'] - a['foreground_voxels']}")
    print(f"  cavities={b['cavity_count'] - a['cavity_count']}")
    print(f"  cavity_voxels={b['cavity_voxels'] - a['cavity_voxels']}")

if __name__ == "__main__":
    main()
