"""
Run full inference pipeline on example images.
Downloads weights and example images automatically via torch.hub.

Usage:
    python run_inference.py

Outputs are saved to the 'output/initial/' directory.
"""

import os
import sys
import time
import torch
import cv2
import numpy as np

# ── Check environment ──────────────────────────────────────────────
print("=" * 60)
print("Environment Check")
print("=" * 60)
print(f"Python:  {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(f"VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: No GPU detected — inference will be slow.")
print("=" * 60)

repo = "KieDani/UpliftingTableTennis"
output_dir = "output/initial"
os.makedirs(output_dir, exist_ok=True)

# ── 1. Download example images ────────────────────────────────────
print("\n[1/3] Downloading example images...")
image_folder = torch.hub.load(repo, "download_example_images",
                              local_folder="example_images", trust_repo=True)
images = [cv2.imread(os.path.join(image_folder, f"{i:02d}.png")) for i in range(35)]
fps = 60.0
print(f"  Loaded {len(images)} images from '{image_folder}'")

# ── 2. Load the full pipeline (downloads weights on first run) ────
print("\n[2/3] Loading full pipeline (ball detection + table detection + uplifting)...")
t0 = time.time()
pipeline = torch.hub.load(repo, "full_pipeline", trust_repo=True)
print(f"  Pipeline loaded in {time.time() - t0:.1f}s")

# ── 3. Run inference ──────────────────────────────────────────────
print("\n[3/3] Running inference...")
t0 = time.time()
pred_spin, pred_pos_3d = pipeline.predict(images, fps)
elapsed = time.time() - t0

print(f"\n{'=' * 60}")
print("Results")
print(f"{'=' * 60}")
print(f"Inference time:          {elapsed:.2f}s")
print(f"3D positions shape:      {pred_pos_3d.shape}")
print(f"Predicted spin vector:   {pred_spin}")

spin_magnitude = pred_spin[1].item() / (2 * np.pi)
if spin_magnitude > 2:
    spin_type = "Topspin"
elif spin_magnitude < -2:
    spin_type = "Backspin"
else:
    spin_type = "No significant spin"
print(f"Spin classification:     {spin_type} ({spin_magnitude:.1f} Hz)")

# ── 4. Save visualizations ────────────────────────────────────────
print(f"\nSaving visualizations to '{output_dir}/'...")

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt

# Get camera matrices for reprojection (use pipeline's internal filtered keypoints)
table_kps, _ = pipeline.table_detector.predict(images)
table_kps_aux, _ = pipeline.table_detector_aux.predict(images)
filtered_table_kps = pipeline.table_detector_aux.filter_trajectory(
    table_kps, table_kps_aux)
Mint, Mext = pipeline.calibrate_camera(filtered_table_kps[0])
reprojected_2d = pipeline.reproject(pred_pos_3d, Mint, Mext)

# Plot reprojected 3D trajectory on a sample frame
frame_idx = 15
img_vis = images[frame_idx].copy()
for x, y in reprojected_2d:
    cv2.circle(img_vis, (int(x), int(y)), 5, (255, 255, 0), -1)

plt.figure(figsize=(14, 8))
plt.title(f"3D Uplifted Trajectory Reprojected (Spin: {spin_type})")
plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.tight_layout()
out_path = os.path.join(output_dir, "trajectory_reprojected.png")
plt.savefig(out_path, dpi=150)
plt.close()
print(f"  Saved: {out_path}")

# Plot 3D trajectory in 3D space
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot(pred_pos_3d[:, 0], pred_pos_3d[:, 1], pred_pos_3d[:, 2],
        "o-", markersize=4, label="Predicted 3D trajectory")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title(f"3D Ball Trajectory — {spin_type} ({spin_magnitude:.1f} Hz)")
ax.legend()
plt.tight_layout()
out_path = os.path.join(output_dir, "trajectory_3d.png")
plt.savefig(out_path, dpi=150)
plt.close()
print(f"  Saved: {out_path}")

print(f"\nDone.")
