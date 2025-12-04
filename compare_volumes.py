import argparse
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import IO.IO as io

def parse_args():
    parser = argparse.ArgumentParser(description="比较平滑前后体数据的体素统计")
    parser.add_argument("before_path", help="平滑前 TIFF 路径")
    parser.add_argument("after_path", help="平滑后 TIFF 路径")
    parser.add_argument("--output-png", default="compare_volumes.png", help="统计图输出路径 (默认: compare_volumes.png)")
    return parser.parse_args()

def summarize(arr):
    arr = np.asarray(arr, dtype=bool)
    voxels = int(arr.sum())
    struct = ndi.generate_binary_structure(3, 3)
    _, n_cc = ndi.label(arr, structure=struct)
    euler = None
    try:
        from skimage.measure import euler_number
        euler = euler_number(arr, connectivity=1)
    except Exception as exc:
        print("    Euler 计算跳过：", exc, flush=True)
    return voxels, n_cc, euler

def plot_metrics(metrics, output_png):
    labels = ["Voxel count", "Connected components"]
    values_before = [metrics["before"][0], metrics["before"][1]]
    values_after = [metrics["after"][0], metrics["after"][1]]

    if metrics["before"][2] is not None and metrics["after"][2] is not None:
        labels.append("Euler characteristic")
        values_before.append(metrics["before"][2])
        values_after.append(metrics["after"][2])

    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    axes = axes[0]
    width = 0.35

    for i, ax in enumerate(axes):
        x = np.arange(1)
        ax.bar(x - width / 2, [values_before[i]], width, label="before")
        ax.bar(x + width / 2, [values_after[i]], width, label="after")
        ax.set_title(labels[i])
        ax.set_xticks([])
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for offset, val, color in [(-width / 2, values_before[i], "blue"), (width / 2, values_after[i], "orange")]:
            ax.text(x + offset, val, f"{val}", ha="center", va="bottom", fontsize=9, color=color)
        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    print("统计图已保存:", output_png, flush=True)

def main():
    args = parse_args()

    print("[1/3] 读取原始:", args.before_path, flush=True)
    before = io.read(args.before_path)

    print("[2/3] 读取平滑:", args.after_path, flush=True)
    after = io.read(args.after_path)

    print("[3/3] 计算指标", flush=True)
    metrics = {"before": summarize(before), "after": summarize(after)}

    plot_metrics(metrics, args.output_png)

if __name__ == "__main__":
    main()