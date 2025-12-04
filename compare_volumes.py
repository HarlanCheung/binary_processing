import numpy as np
import scipy.ndimage as ndi
import IO.IO as io
import matplotlib.pyplot as plt

before_path = "/Users/harlan/Documents/binarysmoothing/ssp_test.tif"
after_path  = "/Users/harlan/Documents/binarysmoothing/ssp_test_smoothed.tif"
output_png  = "compare_volumes.png"

def summarize(arr):
    arr = np.asarray(arr, dtype=bool)
    voxels = int(arr.sum())
    struct = ndi.generate_binary_structure(3, 3)
    _, n_cc = ndi.label(arr, structure=struct)
    euler = None
    try:
        from skimage.measure import euler_number
        euler = euler_number(arr, connectivity=1)
    except Exception as exc:  # 兼容 ImportError / ABI 错误
        print("    Euler 计算跳过：", exc, flush=True)
    return voxels, n_cc, euler


print("[1/3] 读取原始:", before_path, flush=True)
before = io.read(before_path)
print("[2/3] 读取平滑:", after_path, flush=True)
after = io.read(after_path)

print("[3/3] 计算指标", flush=True)
metrics = {}
# 计算指标
metrics = {"before": summarize(before), "after": summarize(after)}

labels = ["Voxel count", "Connected components"]
values_before = [metrics["before"][0], metrics["before"][1]]
values_after  = [metrics["after"][0],  metrics["after"][1]]

# 如 Euler 可用就加一项
if metrics["before"][2] is not None and metrics["after"][2] is not None:
    labels.append("Euler characteristic")
    values_before.append(metrics["before"][2])
    values_after.append(metrics["after"][2])

import matplotlib.pyplot as plt
n = len(labels)
fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
axes = axes[0]
width = 0.35

for i, ax in enumerate(axes):
    x = np.arange(1)  # 单组位置
    ax.bar(x - width/2, [values_before[i]], width, label="before")
    ax.bar(x + width/2, [values_after[i]],  width, label="after")
    ax.set_title(labels[i])
    ax.set_xticks([])
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    # 标注数值
    for offset, val, color in [(-width/2, values_before[i], "blue"), (width/2, values_after[i], "orange")]:
        ax.text(x + offset, val, f"{val}", ha="center", va="bottom", fontsize=9, color=color)
    if i == 0:
        ax.legend()

plt.tight_layout()
plt.savefig("compare_volumes.png", dpi=200)
print("统计图已保存: compare_volumes.png", flush=True)
