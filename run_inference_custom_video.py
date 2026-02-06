"""
Run inference on your own custom video file.

Usage:
    python run_inference_custom_video.py --video path/to/video.mp4 [--fps 60]

Requirements:
    - Video should contain ONE table tennis rally shot
    - Table should be fully visible
    - Recommended: 30-120 frames (0.5-2 seconds at 60fps)

Outputs:
    - output/trajectory_reprojected.png
    - output/trajectory_3d.png
    - output/annotated_video.mp4
    - output/results.json
"""

import os
import sys
import time
import json
import argparse
import torch
import cv2
import numpy as np

# ── Parse arguments ────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run inference on custom video")
parser.add_argument("--video", type=str, required=True,
                    help="Path to input video file")
parser.add_argument("--fps", type=float, default=None,
                    help="Video FPS (auto-detected if not specified)")
parser.add_argument("--max_frames", type=int, default=120,
                    help="Maximum frames to process (default: 120)")
parser.add_argument("--start_frame", type=int, default=0,
                    help="Start frame (default: 0)")
args = parser.parse_args()

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

# ── Load video ─────────────────────────────────────────────────────
print(f"\n[1/4] Loading video: {args.video}")
if not os.path.exists(args.video):
    print(f"ERROR: Video file not found: {args.video}")
    sys.exit(1)

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(f"ERROR: Cannot open video file: {args.video}")
    sys.exit(1)

# Get video properties
video_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Use user-specified FPS or video FPS
fps = args.fps if args.fps is not None else video_fps

print(f"  ✓ Video FPS: {video_fps:.1f}")
print(f"  ✓ Using FPS: {fps:.1f} (for physics calculations)")
print(f"  ✓ Total frames: {total_frames}")
print(f"  ✓ Resolution: {width}x{height}")

# Read frames
images = []
cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
frame_count = 0
while frame_count < args.max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    images.append(frame)
    frame_count += 1

cap.release()

if len(images) < 3:
    print(f"ERROR: Need at least 3 frames, got {len(images)}")
    sys.exit(1)

print(f"  ✓ Loaded {len(images)} frames (from frame {args.start_frame})")
print(f"  ✓ Duration: {len(images)/fps:.2f} seconds")

# The detection models output coordinates in a fixed 2560x1440 space.
# Scale to actual video resolution for drawing.
MODEL_W, MODEL_H = 2560, 1440
sx = width / MODEL_W
sy = height / MODEL_H
print(f"  ✓ Coord scale: {MODEL_W}x{MODEL_H} -> {width}x{height} (sx={sx:.3f}, sy={sy:.3f})")


def scale_xy(x, y):
    """Convert model-space coordinates to video-space coordinates."""
    return int(x * sx), int(y * sy)


# ── Create output directory ────────────────────────────────────────
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# ── Load pipeline ──────────────────────────────────────────────────
print("\n[2/4] Loading pipeline...")
print("  (Downloads ~4GB of weights on first run)")
repo = "KieDani/UpliftingTableTennis"
t0 = time.time()
pipeline = torch.hub.load(repo, "full_pipeline", trust_repo=True)
print(f"  ✓ Pipeline loaded in {time.time() - t0:.1f}s")

# ── Run inference ──────────────────────────────────────────────────
print("\n[3/4] Running inference...")
t0 = time.time()
try:
    pred_spin, pred_pos_3d = pipeline.predict(images, fps)
    elapsed = time.time() - t0
except Exception as e:
    print(f"\nERROR during inference: {e}")
    print("\nPossible issues:")
    print("  - Video quality too low")
    print("  - Ball not visible in most frames")
    print("  - Table not fully visible")
    print("  - Too many frames (try --max_frames 60)")
    sys.exit(1)

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

# Save results
results = {
    "input_video": args.video,
    "inference_time_sec": elapsed,
    "fps": float(fps),
    "num_frames": len(images),
    "start_frame": args.start_frame,
    "num_trajectory_points": int(pred_pos_3d.shape[0]),
    "spin_vector_rad_per_sec": [float(pred_spin[0]), float(pred_spin[1]), float(pred_spin[2])],
    "spin_type": spin_type,
    "spin_magnitude_hz": float(spin_magnitude),
    "spin_magnitude_rpm": float(abs(spin_magnitude) * 60)
}
with open(os.path.join(output_dir, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  ✓ Saved results to {output_dir}/results.json")

# ── Generate visualizations ────────────────────────────────────────
print("\n[4/4] Generating visualizations...")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Get camera calibration
print("  - Calibrating camera...")
table_det_model = torch.hub.load(repo, "table_detection",
                                 model_name="segformerpp_b2", trust_repo=True)
table_kps, _ = table_det_model.predict(images)
Mint, Mext = pipeline.calibrate_camera(table_kps[0])
reprojected_2d = pipeline.reproject(pred_pos_3d, Mint, Mext)

# Get ball detections
print("  - Running ball detection...")
ball_det_model = torch.hub.load(repo, "ball_detection",
                               model_name="segformerpp_b2", trust_repo=True)
image_triples = [(images[i-1].copy(), images[i].copy(), images[i+1].copy())
                 for i in range(1, len(images)-1)]
ball_positions, _ = ball_det_model.predict(image_triples)

# Visualization 1: Single frame
print("  - Creating trajectory visualization...")
frame_idx = min(15, len(images) - 1)
img_vis = images[frame_idx].copy()

for i, (x, y) in enumerate(reprojected_2d):
    dx, dy = scale_xy(x, y)
    cv2.circle(img_vis, (dx, dy), 6, (0, 255, 255), -1)

for i, (x, y, conf) in enumerate(ball_positions):
    if i == frame_idx - 1:
        dx, dy = scale_xy(x, y)
        cv2.circle(img_vis, (dx, dy), 8, (0, 255, 0), 2)

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

# Visualization 2: 3D plot
print("  - Creating 3D trajectory plot...")
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")
ax.plot(pred_pos_3d[:, 0], pred_pos_3d[:, 1], pred_pos_3d[:, 2],
        "o-", color='red', markersize=5, linewidth=2, label="Ball trajectory")

# Table outline
table_length, table_width, table_height = 2.74, 1.525, 0.76
corners = np.array([
    [-table_length/2, -table_width/2, table_height],
    [-table_length/2, table_width/2, table_height],
    [table_length/2, table_width/2, table_height],
    [table_length/2, -table_width/2, table_height],
])
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

# Visualization 3: Annotated video
print("  - Creating annotated video...")
video_path = os.path.join(output_dir, "annotated_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

for frame_idx in range(len(images)):
    img = images[frame_idx].copy()

    # Ball detections trail
    for i in range(min(frame_idx, len(ball_positions))):
        x, y, conf = ball_positions[i]
        alpha = 0.3 + 0.7 * (i / max(1, frame_idx))
        color = (0, int(255 * alpha), 0)
        dx, dy = scale_xy(x, y)
        cv2.circle(img, (dx, dy), 4, color, -1)

    # Current ball detection
    if 0 <= frame_idx - 1 < len(ball_positions):
        x, y, conf = ball_positions[frame_idx - 1]
        dx, dy = scale_xy(x, y)
        cv2.circle(img, (dx, dy), 12, (0, 255, 0), 3)
        cv2.putText(img, f"{conf:.2f}", (dx + 15, dy - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 3D trajectory
    for i, (x, y) in enumerate(reprojected_2d):
        dx, dy = scale_xy(x, y)
        cv2.circle(img, (dx, dy), 5, (0, 255, 255), -1)

    # Table keypoints
    if frame_idx < len(table_kps):
        for kp_idx, (x, y, vis) in enumerate(table_kps[frame_idx]):
            if vis > 0.5:
                dx, dy = scale_xy(x, y)
                cv2.circle(img, (dx, dy), 4, (255, 0, 255), -1)

    # Info overlay
    cv2.rectangle(img, (10, 10), (width - 10, 120), (0, 0, 0), -1)
    cv2.rectangle(img, (10, 10), (width - 10, 120), (255, 255, 255), 2)

    cv2.putText(img, f"Frame: {frame_idx + 1}/{len(images)}", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"Spin: {spin_type} ({spin_magnitude:.1f} Hz)",
               (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"Green: Ball | Cyan: 3D Trajectory | Magenta: Table",
               (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    video_writer.write(img)

video_writer.release()
print(f"  ✓ Saved: {video_path}")

print(f"\n{'=' * 70}")
print("All visualizations saved to 'output/' directory")
print(f"{'=' * 70}")
print("  1. trajectory_reprojected.png  - Single frame with trajectory")
print("  2. trajectory_3d.png           - 3D plot of ball trajectory")
print("  3. annotated_video.mp4         - Annotated video")
print("  4. results.json                - Numerical results")
print(f"{'=' * 70}\n")
