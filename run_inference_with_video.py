"""
Run full inference pipeline with video annotation.
Downloads weights and example images automatically via torch.hub.

Usage:
    python run_inference_with_video.py

Outputs:
    - output/trajectory_reprojected.png - Single frame with 3D trajectory
    - output/trajectory_3d.png - 3D plot of trajectory
    - output/annotated_video.mp4 - Full video with ball detection overlay
    - output/results.json - Numerical results
"""

import os
import sys
import time
import json
import torch
import cv2
import numpy as np

# ── Check environment ──────────────────────────────────────────────
print("=" * 70)
print("Environment Check")
print("=" * 70)
print(f"Python:  {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(f"VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: No GPU detected — inference will be SLOW on CPU.")
print("=" * 70)

repo = "KieDani/UpliftingTableTennis"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# ── 1. Download example images ────────────────────────────────────
print("\n[1/4] Downloading example images...")
image_folder = torch.hub.load(repo, "download_example_images",
                              local_folder="example_images", trust_repo=True)
images = [cv2.imread(os.path.join(image_folder, f"{i:02d}.png")) for i in range(35)]
fps = 60.0
print(f"  ✓ Loaded {len(images)} images at {fps} fps")
print(f"  ✓ Duration: {len(images)/fps:.2f} seconds")
height, width = images[0].shape[:2]
print(f"  ✓ Resolution: {width}x{height}")

# The detection models output coordinates in a fixed 1920x1080 space.
# Scale to actual image resolution for drawing.
MODEL_W, MODEL_H = 1920, 1080
sx = width / MODEL_W
sy = height / MODEL_H


def scale_xy(x, y):
    """Convert model-space coordinates to video-space coordinates."""
    return int(x * sx), int(y * sy)


# ── 2. Load the full pipeline ─────────────────────────────────────
print("\n[2/4] Loading full pipeline...")
print("  (This downloads ~4GB of weights on first run)")
t0 = time.time()
pipeline = torch.hub.load(repo, "full_pipeline", trust_repo=True)
print(f"  ✓ Pipeline loaded in {time.time() - t0:.1f}s")

# ── 3. Run inference ──────────────────────────────────────────────
print("\n[3/4] Running inference...")
t0 = time.time()
pred_spin, pred_pos_3d = pipeline.predict(images, fps)
elapsed = time.time() - t0

# Interpret spin
spin_magnitude = pred_spin[1].item() / (2 * np.pi)
if spin_magnitude > 2:
    spin_type = "Topspin"
elif spin_magnitude < -2:
    spin_type = "Backspin"
else:
    spin_type = "No significant spin"

print(f"\n{'=' * 70}")
print("RESULTS")
print(f"{'=' * 70}")
print(f"Inference time:          {elapsed:.2f}s ({len(images)/elapsed:.1f} fps)")
print(f"3D trajectory points:    {pred_pos_3d.shape[0]}")
print(f"Spin vector (rad/s):     [{pred_spin[0]:.2f}, {pred_spin[1]:.2f}, {pred_spin[2]:.2f}]")
print(f"Spin classification:     {spin_type}")
print(f"Spin magnitude:          {spin_magnitude:.1f} Hz ({abs(spin_magnitude)*60:.0f} RPM)")
print(f"{'=' * 70}")

# Save results to JSON
results = {
    "inference_time_sec": elapsed,
    "fps": float(fps),
    "num_frames": len(images),
    "num_trajectory_points": int(pred_pos_3d.shape[0]),
    "spin_vector_rad_per_sec": [float(pred_spin[0]), float(pred_spin[1]), float(pred_spin[2])],
    "spin_type": spin_type,
    "spin_magnitude_hz": float(spin_magnitude),
    "spin_magnitude_rpm": float(abs(spin_magnitude) * 60)
}
with open(os.path.join(output_dir, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  ✓ Saved results to {output_dir}/results.json")

# ── 4. Generate visualizations ────────────────────────────────────
print("\n[4/4] Generating visualizations...")

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# Get camera calibration for reprojection (use pipeline's internal filtered keypoints)
print("  - Calibrating camera...")
table_kps, _ = pipeline.table_detector.predict(images)
table_kps_aux, _ = pipeline.table_detector_aux.predict(images)
filtered_table_kps = pipeline.table_detector_aux.filter_trajectory(
    table_kps, table_kps_aux)
Mint, Mext = pipeline.calibrate_camera(filtered_table_kps)
reprojected_2d = pipeline.reproject(pred_pos_3d, Mint, Mext)

# Also get ball detections for visualization
print("  - Running ball detection for visualization...")
ball_det_model = torch.hub.load(repo, "ball_detection",
                               model_name="segformerpp_b2", trust_repo=True)
image_triples = [(images[i-1].copy(), images[i].copy(), images[i+1].copy())
                 for i in range(1, len(images)-1)]
ball_positions, _ = ball_det_model.predict(image_triples)

# ── Visualization 1: Single frame with trajectory ────────────────
print("  - Creating trajectory visualization...")
frame_idx = 15
img_vis = images[frame_idx].copy()

# Draw all reprojected 3D points
for i, (x, y) in enumerate(reprojected_2d):
    color = (0, 255, 255)  # Cyan for 3D trajectory
    dx, dy = scale_xy(x, y)
    cv2.circle(img_vis, (dx, dy), 6, color, -1)

# Draw ball detections in different color for comparison
for i, (x, y, conf) in enumerate(ball_positions):
    if i == frame_idx - 1:  # Account for offset (triples start at frame 1)
        dx, dy = scale_xy(x, y)
        cv2.circle(img_vis, (dx, dy), 8, (0, 255, 0), 2)  # Green circle for detection

# Add text overlay
cv2.putText(img_vis, f"Spin: {spin_type} ({spin_magnitude:.1f} Hz)",
            (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
cv2.putText(img_vis, "Cyan: 3D Trajectory | Green: Ball Detection",
            (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

plt.figure(figsize=(16, 9))
plt.title(f"3D Trajectory Reprojected onto Frame {frame_idx}")
plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.tight_layout()
out_path = os.path.join(output_dir, "trajectory_reprojected.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {out_path}")

# ── Visualization 2: 3D trajectory plot ──────────────────────────
print("  - Creating 3D trajectory plot...")
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")
ax.plot(pred_pos_3d[:, 0], pred_pos_3d[:, 1], pred_pos_3d[:, 2],
        "o-", color='red', markersize=5, linewidth=2, label="Ball trajectory")

# Draw table outline for reference
table_length, table_width, table_height = 2.74, 1.525, 0.76
corners = np.array([
    [-table_length/2, -table_width/2, table_height],
    [-table_length/2, table_width/2, table_height],
    [table_length/2, table_width/2, table_height],
    [table_length/2, -table_width/2, table_height],
])
# Draw table edges
for i in range(4):
    next_i = (i + 1) % 4
    ax.plot([corners[i, 0], corners[next_i, 0]],
            [corners[i, 1], corners[next_i, 1]],
            [corners[i, 2], corners[next_i, 2]], 'k-', linewidth=2)

ax.set_xlabel("X (m)", fontsize=12)
ax.set_ylabel("Y (m)", fontsize=12)
ax.set_zlabel("Z (m)", fontsize=12)
ax.set_title(f"3D Ball Trajectory\nSpin: {spin_type} ({spin_magnitude:.1f} Hz)",
             fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True)
plt.tight_layout()
out_path = os.path.join(output_dir, "trajectory_3d.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {out_path}")

# ── Visualization 3: Annotated video ─────────────────────────────
print("  - Creating annotated video...")
video_path = os.path.join(output_dir, "annotated_video.mp4")
height, width = images[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

for frame_idx in range(len(images)):
    img = images[frame_idx].copy()

    # Draw all 2D ball detections up to this frame
    for i in range(min(frame_idx, len(ball_positions))):
        x, y, conf = ball_positions[i]
        alpha = 0.3 + 0.7 * (i / max(1, frame_idx))  # Fade older detections
        color = (0, int(255 * alpha), 0)
        dx, dy = scale_xy(x, y)
        cv2.circle(img, (dx, dy), 4, color, -1)

    # Draw current ball detection with highlight
    if 0 <= frame_idx - 1 < len(ball_positions):
        x, y, conf = ball_positions[frame_idx - 1]
        dx, dy = scale_xy(x, y)
        cv2.circle(img, (dx, dy), 12, (0, 255, 0), 3)
        cv2.putText(img, f"{conf:.2f}", (dx + 15, dy - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw reprojected 3D trajectory
    for i, (x, y) in enumerate(reprojected_2d):
        dx, dy = scale_xy(x, y)
        cv2.circle(img, (dx, dy), 5, (0, 255, 255), -1)

    # Draw table keypoints
    if frame_idx < len(table_kps):
        for kp_idx, (x, y, vis) in enumerate(table_kps[frame_idx]):
            if vis > 0.5:  # visible
                dx, dy = scale_xy(x, y)
                cv2.circle(img, (dx, dy), 4, (255, 0, 255), -1)

    # Add info overlay
    cv2.rectangle(img, (10, 10), (width - 10, 120), (0, 0, 0), -1)
    cv2.rectangle(img, (10, 10), (width - 10, 120), (255, 255, 255), 2)

    cv2.putText(img, f"Frame: {frame_idx + 1}/{len(images)}", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"Spin: {spin_type} ({spin_magnitude:.1f} Hz)",
               (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"Green: Ball Detection | Cyan: 3D Trajectory | Magenta: Table",
               (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    video_writer.write(img)

video_writer.release()
print(f"  ✓ Saved: {video_path}")

print(f"\n{'=' * 70}")
print("All visualizations saved to 'output/' directory")
print(f"{'=' * 70}")
print("  1. trajectory_reprojected.png  - Single frame with trajectory overlay")
print("  2. trajectory_3d.png           - 3D plot of ball trajectory")
print("  3. annotated_video.mp4         - Full video with detections")
print("  4. results.json                - Numerical results")
print(f"{'=' * 70}\n")
