"""
Process a full table tennis match video end-to-end.

Automatically:
  1. Detects the ball in every frame
  2. Segments the video into individual rallies (by detection gaps)
  3. Splits each rally into individual shots (by direction reversals)
  4. Runs the full 3D uplifting + spin estimation on each shot
  5. Outputs annotated videos + per-shot results

Usage:
    python run_inference_full_video.py --video match.mp4
    python run_inference_full_video.py --video match.mp4 --fps 60 --max_frames 3000

Outputs (in output/):
    - annotated_full_video.mp4          - Full video with all overlays
    - summary.json                      - Overview of all detected shots
    - shots/shot_NNN.json               - Per-shot results (spin, trajectory)
    - shots/shot_NNN.png                - Per-shot trajectory snapshot image
    - shots/shot_NNN_video.mp4          - Per-shot annotated video clip
"""

import os
import sys
import time
import json
import argparse
import torch
import cv2
import numpy as np
from collections import defaultdict

# ── Args ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Process full table tennis video")
parser.add_argument("--video", type=str, required=True)
parser.add_argument("--fps", type=float, default=None,
                    help="Override video FPS (auto-detected if omitted)")
parser.add_argument("--max_frames", type=int, default=900,
                    help="Max frames to process (default: 900)")
parser.add_argument("--confidence_threshold", type=float, default=0.3,
                    help="Min ball detection confidence (default: 0.3)")
parser.add_argument("--min_shot_frames", type=int, default=10,
                    help="Minimum frames per shot (default: 10)")
parser.add_argument("--gap_frames", type=int, default=20,
                    help="Consecutive low-confidence frames to mark rally break (default: 20)")
parser.add_argument("--smoothing_window", type=int, default=5,
                    help="Smoothing window for direction detection (default: 5)")
parser.add_argument("--detector_chunk_size", type=int, default=256,
                    help="Triples per detection batch (default: 256)")
parser.add_argument("--max_buffer_gb", type=float, default=6.0,
                    help="Abort if estimated frame buffer exceeds this many GB (default: 6.0)")
args = parser.parse_args()
if args.detector_chunk_size <= 0:
    parser.error("--detector_chunk_size must be > 0")
if args.max_buffer_gb <= 0:
    parser.error("--max_buffer_gb must be > 0")

# ── Environment ────────────────────────────────────────────────────
print("=" * 70)
print("Full Video Inference Pipeline")
print("=" * 70)
print(f"Python:  {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(f"VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 70)


# ── 1. Load video ─────────────────────────────────────────────────
print(f"\n[1/6] Loading video: {args.video}")
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(f"ERROR: Cannot open video: {args.video}")
    sys.exit(1)

video_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = args.fps if args.fps is not None else video_fps
if not np.isfinite(fps) or fps <= 0:
    print(f"WARNING: Invalid FPS '{fps}'. Falling back to 60.0.")
    fps = 60.0

frames_to_process = min(total_frames, args.max_frames)
estimated_gb = (frames_to_process * width * height * 3) / (1024 ** 3)

print(f"  Video FPS: {video_fps:.1f} | Using: {fps:.1f}")
print(f"  Total frames: {total_frames} | Processing: {frames_to_process}")
print(f"  Resolution: {width}x{height}")
print(f"  Estimated frame buffer: {estimated_gb:.2f} GB")

if frames_to_process < 3:
    print(f"ERROR: Need at least 3 frames to run temporal ball detection, got {frames_to_process}.")
    cap.release()
    sys.exit(1)

if estimated_gb > args.max_buffer_gb:
    safe_frames = int((args.max_buffer_gb * (1024 ** 3)) / max(1, width * height * 3))
    print(f"ERROR: Estimated frame buffer ({estimated_gb:.2f} GB) exceeds --max_buffer_gb={args.max_buffer_gb:.2f}.")
    print(f"       Reduce --max_frames to <= {safe_frames} for this resolution.")
    cap.release()
    sys.exit(1)

images = []
count = 0
while count < frames_to_process:
    ret, frame = cap.read()
    if not ret:
        break
    images.append(frame)
    count += 1
cap.release()
print(f"  Loaded {len(images)} frames ({len(images)/fps:.1f}s)")
if len(images) < 3:
    print(f"ERROR: Need at least 3 decodable frames to run temporal ball detection, got {len(images)}.")
    sys.exit(1)

# The detection models output coordinates in a fixed 1920x1080 space regardless
# of actual input resolution. We need scale factors to draw on the real frames.
MODEL_W, MODEL_H = 1920, 1080
sx = width / MODEL_W    # scale x from model space -> video space
sy = height / MODEL_H   # scale y from model space -> video space
print(f"  Model coord space: {MODEL_W}x{MODEL_H} -> scale factors: sx={sx:.3f}, sy={sy:.3f}")


def scale_xy(x, y):
    """Convert model-space coordinates to video-space coordinates."""
    return int(x * sx), int(y * sy)


def predict_ball_positions_chunked(detector, frames, chunk_size):
    """Run temporal ball detection without building one huge triple list."""
    if len(frames) < 3:
        return np.empty((0, 3), dtype=np.float32)

    chunked_positions = []
    for start in range(1, len(frames) - 1, chunk_size):
        end = min(start + chunk_size, len(frames) - 1)
        triples = [
            (frames[i - 1], frames[i], frames[i + 1])
            for i in range(start, end)
        ]
        positions, _ = detector.predict(triples)
        chunked_positions.append(positions)

    if not chunked_positions:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(chunked_positions, axis=0)


# ── 2. Run ball detection on full video ────────────────────────────
print(f"\n[2/6] Running ball detection on {len(images)} frames...")
repo = "KieDani/UpliftingTableTennis"

t0 = time.time()
ball_det = torch.hub.load(repo, "ball_detection", model_name="segformerpp_b2", trust_repo=True)
ball_det_aux = torch.hub.load(repo, "ball_detection", model_name="wasb", trust_repo=True)

ball_positions = predict_ball_positions_chunked(ball_det, images, args.detector_chunk_size)
ball_positions_aux = predict_ball_positions_chunked(ball_det_aux, images, args.detector_chunk_size)

if len(ball_positions) != len(ball_positions_aux):
    print("ERROR: Primary and auxiliary detector outputs have mismatched lengths.")
    sys.exit(1)
if len(ball_positions) == 0:
    print("ERROR: Ball detector produced no outputs. Check the input video and model setup.")
    sys.exit(1)

# Confidence proxy from detector agreement. This is robust to always-on visibility flags.
disagreement = np.linalg.norm(ball_positions[:, :2] - ball_positions_aux[:, :2], axis=1)
agreement_conf = np.clip(1.0 - (disagreement / 40.0), 0.0, 1.0)
visible_mask = np.logical_and(ball_positions[:, 2] > 0.5, ball_positions_aux[:, 2] > 0.5)
agreement_conf *= visible_mask.astype(np.float32)

print(f"  Ball detection done in {time.time() - t0:.1f}s")
print(f"  Detections: {len(ball_positions)} (frames 1..{len(images)-2})")
if len(agreement_conf) > 0:
    print(f"  Confidence (agreement) mean={agreement_conf.mean():.3f}, "
          f"p10={np.percentile(agreement_conf, 10):.3f}, "
          f"p90={np.percentile(agreement_conf, 90):.3f}")
else:
    print("  Confidence (agreement): no valid detections")

# ball_positions is (N, 3) where N = len(images)-2, columns = [x, y, confidence]
# Index i in ball_positions corresponds to frame i+1 in images
# We pad to align with frame indices
padded_positions = np.zeros((len(images), 3), dtype=np.float32)
padded_positions[1:-1, :2] = ball_positions[:, :2]
padded_positions[1:-1, 2] = agreement_conf
# Keep edge coordinates but leave confidence at zero (no temporal context at boundaries).
padded_positions[0, :2] = ball_positions[0, :2]
padded_positions[-1, :2] = ball_positions[-1, :2]


# ── 3. Segment into rallies and shots ─────────────────────────────
print(f"\n[3/6] Segmenting video into rallies and shots...")

confidences = padded_positions[:, 2]
x_positions = padded_positions[:, 0]

# --- Phase A: Find rally boundaries (gaps in detection) ---
low_conf = confidences < args.confidence_threshold
rally_segments = []  # list of (start_frame, end_frame)
in_rally = False
rally_start = 0
gap_count = 0

for i in range(len(images)):
    if low_conf[i]:
        gap_count += 1
        if in_rally and gap_count >= args.gap_frames:
            # End the rally at the frame before the gap started
            rally_end = i - gap_count + 1
            if rally_end - rally_start >= args.min_shot_frames:
                rally_segments.append((rally_start, rally_end))
            in_rally = False
    else:
        if not in_rally:
            rally_start = i
            in_rally = True
        gap_count = 0

# Close final rally
if in_rally:
    rally_end = len(images)
    if rally_end - rally_start >= args.min_shot_frames:
        rally_segments.append((rally_start, rally_end))

print(f"  Found {len(rally_segments)} rallies")
for idx, (s, e) in enumerate(rally_segments):
    print(f"    Rally {idx+1}: frames {s}-{e} ({(e-s)/fps:.1f}s)")


# --- Phase B: Split each rally into shots (direction reversals) ---
def find_shots_in_rally(x_pos, conf, rally_start, rally_end, min_frames, window):
    """
    Detect individual shots within a rally by finding horizontal direction reversals.

    Returns list of (start_frame, end_frame) tuples in global frame indices.
    """
    length = rally_end - rally_start
    if length < min_frames:
        return []

    segment_x = x_pos[rally_start:rally_end].copy()
    segment_conf = conf[rally_start:rally_end].copy()

    # Replace low-confidence positions with interpolated values
    valid = segment_conf >= args.confidence_threshold
    if valid.sum() < min_frames:
        return []

    valid_indices = np.where(valid)[0]
    all_indices = np.arange(length)
    segment_x = np.interp(all_indices, valid_indices, segment_x[valid_indices])

    # Smooth x-positions
    kernel = np.ones(window) / window
    if len(segment_x) >= window:
        smoothed_x = np.convolve(segment_x, kernel, mode='same')
    else:
        smoothed_x = segment_x

    # Compute x-velocity
    dx = np.diff(smoothed_x)

    # Find sign changes in velocity (direction reversals)
    # Use a threshold to avoid noise-induced reversals
    sign_dx = np.sign(dx)
    # Remove near-zero velocities (stationary ball)
    sign_dx[np.abs(dx) < 2.0] = 0

    # Find actual sign changes (skip zeros)
    reversals = []
    last_sign = 0
    for i, s in enumerate(sign_dx):
        if s != 0:
            if last_sign != 0 and s != last_sign:
                reversals.append(i)
            last_sign = s

    # Create shot segments from reversals
    if len(reversals) == 0:
        # No reversals: entire rally is one shot
        return [(rally_start, rally_end)]

    shots = []
    # First shot: rally_start to first reversal
    prev = 0
    for rev in reversals:
        if rev - prev >= min_frames:
            shots.append((rally_start + prev, rally_start + rev))
        prev = rev

    # Last shot: last reversal to rally_end
    if length - prev >= min_frames:
        shots.append((rally_start + prev, rally_end))

    return shots


all_shots = []
for rally_idx, (rs, re) in enumerate(rally_segments):
    shots = find_shots_in_rally(x_positions, confidences, rs, re,
                                args.min_shot_frames, args.smoothing_window)
    for shot in shots:
        all_shots.append((rally_idx, shot[0], shot[1]))

print(f"\n  Found {len(all_shots)} individual shots across {len(rally_segments)} rallies:")
for idx, (rally, s, e) in enumerate(all_shots):
    print(f"    Shot {idx+1} (rally {rally+1}): frames {s}-{e} "
          f"({e-s} frames, {(e-s)/fps:.2f}s)")


# ── 4. Run full pipeline on each shot ─────────────────────────────
print(f"\n[4/6] Running 3D uplifting on {len(all_shots)} shots...")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pipeline = torch.hub.load(repo, "full_pipeline", trust_repo=True)

output_dir = "output"
shots_dir = os.path.join(output_dir, "shots")
os.makedirs(shots_dir, exist_ok=True)

# Color palette for different shots
SHOT_COLORS = [
    (0, 255, 255),   # cyan
    (255, 128, 0),   # orange
    (0, 255, 0),     # green
    (255, 0, 255),   # magenta
    (255, 255, 0),   # yellow
    (128, 0, 255),   # purple
    (0, 128, 255),   # light blue
    (255, 0, 128),   # pink
]

shot_results = []

for idx, (rally_idx, start, end) in enumerate(all_shots):
    shot_images = images[start:end]
    print(f"\n  Shot {idx+1}/{len(all_shots)} "
          f"(rally {rally_idx+1}, frames {start}-{end}, {len(shot_images)} frames)")

    if len(shot_images) < 3:
        print(f"    SKIP: too few frames")
        shot_results.append(None)
        continue

    try:
        t0 = time.time()
        pred_spin, pred_pos_3d = pipeline.predict(shot_images, fps)
        elapsed = time.time() - t0

        spin_mag = pred_spin[1].item() / (2 * np.pi)
        if spin_mag > 2:
            spin_type = "Topspin"
        elif spin_mag < -2:
            spin_type = "Backspin"
        else:
            spin_type = "Flat"

        # Camera calibration for reprojection
        # Use BOTH table detection models + filtering (matches pipeline internals)
        table_kps, _ = pipeline.table_detector.predict(shot_images)
        table_kps_aux, _ = pipeline.table_detector_aux.predict(shot_images)
        filtered_table_kps = pipeline.table_detector_aux.filter_trajectory(
            table_kps, table_kps_aux)
        Mint, Mext = pipeline.calibrate_camera(filtered_table_kps)
        reprojected = pipeline.reproject(pred_pos_3d, Mint, Mext)

        # Ball detections for this shot (from the full-video detection pass)
        shot_ball_positions = padded_positions[start:end]

        result = {
            "shot_index": idx,
            "rally_index": rally_idx,
            "start_frame": start,
            "end_frame": end,
            "num_frames": end - start,
            "duration_sec": (end - start) / fps,
            "inference_time_sec": elapsed,
            "spin_vector": [float(pred_spin[0]), float(pred_spin[1]), float(pred_spin[2])],
            "spin_type": spin_type,
            "spin_magnitude_hz": float(spin_mag),
            "spin_magnitude_rpm": float(abs(spin_mag) * 60),
            "trajectory_3d": pred_pos_3d.tolist(),
            "trajectory_2d_reprojected": reprojected.tolist(),
        }
        shot_results.append(result)

        # Save per-shot JSON
        shot_path = os.path.join(shots_dir, f"shot_{idx:03d}.json")
        with open(shot_path, "w") as f:
            json.dump(result, f, indent=2)

        # ── Per-shot snapshot image ──
        mid = len(shot_images) // 2
        img_snap = shot_images[mid].copy()
        shot_color = SHOT_COLORS[idx % len(SHOT_COLORS)]

        # Draw full reprojected 3D trajectory (scale from model space)
        for pt in reprojected:
            dx, dy = scale_xy(pt[0], pt[1])
            cv2.circle(img_snap, (dx, dy), 6, shot_color, -1)

        # Draw all 2D ball detections as trail (scale from model space)
        for i in range(len(shot_ball_positions)):
            bx, by, bc = shot_ball_positions[i]
            if bc >= args.confidence_threshold:
                dx, dy = scale_xy(bx, by)
                cv2.circle(img_snap, (dx, dy), 4, (0, 255, 0), -1)

        # Draw table keypoints on mid frame (scale from model space)
        if mid < len(table_kps):
            for kx, ky, kv in table_kps[mid]:
                if kv > 0.5:
                    dx, dy = scale_xy(kx, ky)
                    cv2.circle(img_snap, (dx, dy), 5, (255, 0, 255), -1)

        # Text overlay
        cv2.rectangle(img_snap, (10, 10), (700, 130), (0, 0, 0), -1)
        cv2.rectangle(img_snap, (10, 10), (700, 130), (255, 255, 255), 2)
        cv2.putText(img_snap,
                    f"Shot {idx+1} | Rally {rally_idx+1} | Frames {start}-{end}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img_snap,
                    f"Spin: {spin_type} ({spin_mag:.1f} Hz / {abs(spin_mag)*60:.0f} RPM)",
                    (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, shot_color, 2)
        cv2.putText(img_snap,
                    f"3D points: {pred_pos_3d.shape[0]} | Inference: {elapsed:.1f}s",
                    (20, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        snap_path = os.path.join(shots_dir, f"shot_{idx:03d}.png")
        cv2.imwrite(snap_path, img_snap)

        # ── Per-shot annotated video ──
        shot_video_path = os.path.join(shots_dir, f"shot_{idx:03d}_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        shot_writer = cv2.VideoWriter(shot_video_path, fourcc, fps, (width, height))

        for local_idx, frame in enumerate(shot_images):
            img = frame.copy()
            global_idx = start + local_idx

            # Draw ball detection trail up to this frame
            trail_start = max(0, local_idx - 15)
            for t in range(trail_start, local_idx):
                bx, by, bc = shot_ball_positions[t]
                if bc >= args.confidence_threshold:
                    alpha = (t - trail_start) / max(1, local_idx - trail_start)
                    dx, dy = scale_xy(bx, by)
                    cv2.circle(img, (dx, dy), 4, (0, int(200 * alpha), 0), -1)

            # Current ball detection highlight
            if local_idx < len(shot_ball_positions):
                bx, by, bc = shot_ball_positions[local_idx]
                if bc >= args.confidence_threshold:
                    dx, dy = scale_xy(bx, by)
                    cv2.circle(img, (dx, dy), 12, (0, 255, 0), 3)
                    cv2.putText(img, f"{bc:.2f}", (dx + 15, dy - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Reprojected 3D trajectory (full, always visible)
            for pt in reprojected:
                dx, dy = scale_xy(pt[0], pt[1])
                cv2.circle(img, (dx, dy), 5, shot_color, -1)

            # Table keypoints
            if local_idx < len(table_kps):
                for kx, ky, kv in table_kps[local_idx]:
                    if kv > 0.5:
                        dx, dy = scale_xy(kx, ky)
                        cv2.circle(img, (dx, dy), 4, (255, 0, 255), -1)

            # Info overlay
            cv2.rectangle(img, (10, 10), (width - 10, 130), (0, 0, 0), -1)
            cv2.rectangle(img, (10, 10), (width - 10, 130), (255, 255, 255), 2)
            cv2.putText(img,
                        f"Shot {idx+1} | Frame {local_idx+1}/{len(shot_images)} "
                        f"(global: {global_idx})",
                        (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img,
                        f"Spin: {spin_type} ({spin_mag:.1f} Hz / {abs(spin_mag)*60:.0f} RPM)",
                        (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, shot_color, 2)
            cv2.putText(img,
                        "Green: Ball Detection | Colored: 3D Trajectory | Magenta: Table",
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(img,
                        f"Rally {rally_idx+1} | Frames {start}-{end} | {end-start} frames total",
                        (20, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            shot_writer.write(img)

        shot_writer.release()

        print(f"    {spin_type} ({spin_mag:.1f} Hz / {abs(spin_mag)*60:.0f} RPM) "
              f"| {pred_pos_3d.shape[0]} 3D points | {elapsed:.1f}s")
        print(f"    -> {snap_path}")
        print(f"    -> {shot_video_path}")

    except Exception as e:
        print(f"    FAILED: {e}")
        shot_results.append(None)


# ── 5. Generate per-shot 3D trajectory plots ──────────────────────
print(f"\n[5/6] Generating 3D trajectory plots...")

table_length, table_width, table_height = 2.74, 1.525, 0.76
table_corners = np.array([
    [-table_length/2, -table_width/2, table_height],
    [-table_length/2, table_width/2, table_height],
    [table_length/2, table_width/2, table_height],
    [table_length/2, -table_width/2, table_height],
])

for idx, result in enumerate(shot_results):
    if result is None:
        continue
    traj = np.array(result["trajectory_3d"])
    shot_color_rgb = tuple(c / 255 for c in reversed(SHOT_COLORS[idx % len(SHOT_COLORS)]))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "o-",
            color=shot_color_rgb, markersize=4, linewidth=2, label="Ball trajectory")

    for i in range(4):
        ni = (i + 1) % 4
        ax.plot([table_corners[i, 0], table_corners[ni, 0]],
                [table_corners[i, 1], table_corners[ni, 1]],
                [table_corners[i, 2], table_corners[ni, 2]], "k-", linewidth=2)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Shot {idx+1} — {result['spin_type']} "
                 f"({result['spin_magnitude_hz']:.1f} Hz / "
                 f"{result['spin_magnitude_rpm']:.0f} RPM)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(shots_dir, f"shot_{idx:03d}_3d.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Shot {idx+1}: {plot_path}")


# ── 6. Generate full annotated video ──────────────────────────────
print(f"\n[6/6] Creating full annotated video...")

# Build lookup: frame_idx -> shot indices active at that frame
frame_to_shots = defaultdict(list)
for idx, result in enumerate(shot_results):
    if result is None:
        continue
    for frame_idx in range(result["start_frame"], result["end_frame"]):
        frame_to_shots[frame_idx].append(idx)

video_path = os.path.join(output_dir, "annotated_full_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

for frame_idx in range(len(images)):
    img = images[frame_idx].copy()

    # Draw ball detection trail (last 15 frames)
    trail_start = max(0, frame_idx - 15)
    for t in range(trail_start, min(frame_idx, len(padded_positions))):
        bx, by, bc = padded_positions[t]
        if bc >= args.confidence_threshold:
            alpha = (t - trail_start) / max(1, frame_idx - trail_start)
            color = (0, int(180 * alpha), 0)
            dx, dy = scale_xy(bx, by)
            cv2.circle(img, (dx, dy), 3, color, -1)

    # Draw current ball detection
    if frame_idx < len(padded_positions):
        bx, by, bc = padded_positions[frame_idx]
        if bc >= args.confidence_threshold:
            dx, dy = scale_xy(bx, by)
            cv2.circle(img, (dx, dy), 10, (0, 255, 0), 3)

    # Draw reprojected 3D trajectory for active shots
    active_shot_indices = frame_to_shots.get(frame_idx, [])
    for shot_idx in active_shot_indices:
        result = shot_results[shot_idx]
        if result is None:
            continue
        color = SHOT_COLORS[shot_idx % len(SHOT_COLORS)]
        for pt in result["trajectory_2d_reprojected"]:
            dx, dy = scale_xy(pt[0], pt[1])
            cv2.circle(img, (dx, dy), 5, color, -1)

    # Info overlay
    overlay_h = 100
    cv2.rectangle(img, (10, 10), (width - 10, 10 + overlay_h), (0, 0, 0), -1)
    cv2.rectangle(img, (10, 10), (width - 10, 10 + overlay_h), (255, 255, 255), 2)

    cv2.putText(img, f"Frame {frame_idx+1}/{len(images)}", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show active shot info
    if active_shot_indices:
        shot_idx = active_shot_indices[-1]  # most recent shot
        r = shot_results[shot_idx]
        if r is not None:
            color = SHOT_COLORS[shot_idx % len(SHOT_COLORS)]
            cv2.putText(img,
                        f"Shot {shot_idx+1}: {r['spin_type']} "
                        f"({r['spin_magnitude_hz']:.1f} Hz)",
                        (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        cv2.putText(img, "Between shots", (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

    cv2.putText(img, "Green: Ball | Colored: 3D Trajectory per shot",
                (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    writer.write(img)

writer.release()
print(f"  Saved: {video_path}")

# ── Save summary ───────────────────────────────────────────────────
summary = {
    "input_video": args.video,
    "fps": fps,
    "total_frames": len(images),
    "total_duration_sec": len(images) / fps,
    "num_rallies": len(rally_segments),
    "num_shots": len(all_shots),
    "num_successful": sum(1 for r in shot_results if r is not None),
    "shots": [],
}
for idx, result in enumerate(shot_results):
    if result is None:
        summary["shots"].append({"shot_index": idx, "status": "failed"})
    else:
        summary["shots"].append({
            "shot_index": idx,
            "rally_index": result["rally_index"],
            "frames": f"{result['start_frame']}-{result['end_frame']}",
            "duration_sec": result["duration_sec"],
            "spin_type": result["spin_type"],
            "spin_hz": result["spin_magnitude_hz"],
            "spin_rpm": result["spin_magnitude_rpm"],
        })

summary_path = os.path.join(output_dir, "summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print(f"  Rallies detected:  {len(rally_segments)}")
print(f"  Shots detected:    {len(all_shots)}")
print(f"  Shots processed:   {sum(1 for r in shot_results if r is not None)}")
print()
for idx, result in enumerate(shot_results):
    if result is None:
        print(f"  Shot {idx+1}: FAILED")
    else:
        print(f"  Shot {idx+1}: {result['spin_type']:>8s} "
              f"| {result['spin_magnitude_hz']:+.1f} Hz "
              f"| {result['spin_magnitude_rpm']:.0f} RPM "
              f"| frames {result['start_frame']}-{result['end_frame']}")
print(f"\n{'=' * 70}")
print("Outputs:")
print(f"  {video_path}")
print(f"  {summary_path}")
print(f"  {shots_dir}/")
print(f"    shot_NNN.json        - per-shot spin + trajectory data")
print(f"    shot_NNN.png         - snapshot image with overlays")
print(f"    shot_NNN_3d.png      - 3D trajectory plot")
print(f"    shot_NNN_video.mp4   - annotated video clip for that shot")
print(f"{'=' * 70}\n")
