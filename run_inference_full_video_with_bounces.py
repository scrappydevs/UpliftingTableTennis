"""
Run full-video inference and generate a bounce-location map.

This script wraps `run_inference_full_video.py` (same analysis pipeline), then
post-processes the per-shot 3D trajectories to estimate bounce points and saves:

  - output/bounces.json
  - output/bounce_map.png

Usage:
  python run_inference_full_video_with_bounces.py --video match.mp4
  python run_inference_full_video_with_bounces.py --video match.mp4 --max_frames 900
  python run_inference_full_video_with_bounces.py --skip_inference
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


TABLE_LENGTH_M = 2.74
TABLE_WIDTH_M = 1.525
TABLE_HEIGHT_M = 0.76


def run_full_video_inference(base_script, passthrough_args):
    cmd = [sys.executable, base_script] + passthrough_args
    print(f"[1/3] Running full-video inference: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def local_quadratic_refine(series, idx, fps):
    """
    Refine value around index idx with a local quadratic fit on (idx-1, idx, idx+1).
    Returns:
      refined_index_float, refined_value
    """
    dt = np.array([-1.0, 0.0, 1.0], dtype=np.float64) / float(fps)
    window = np.array([series[idx - 1], series[idx], series[idx + 1]], dtype=np.float64)
    coeff = np.polyfit(dt, window, 2)  # a, b, c
    a, b, c = coeff
    if abs(a) < 1e-10:
        return float(idx), float(series[idx])
    dt_star = -b / (2.0 * a)
    dt_star = float(np.clip(dt_star, dt[0], dt[-1]))
    val_star = float(np.polyval(coeff, dt_star))
    idx_star = float(idx) + dt_star * float(fps)
    return idx_star, val_star


def estimate_bounces_for_shot(
    trajectory_3d,
    fps,
    start_frame,
    table_height,
    z_tolerance,
    table_margin,
    min_vertical_change,
    min_separation_frames,
    max_bounces_per_shot,
):
    points = np.asarray(trajectory_3d, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 3:
        return []

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    local_minima = []
    for i in range(1, len(z) - 1):
        if not (z[i - 1] > z[i] and z[i + 1] > z[i]):
            continue
        dz_before = z[i] - z[i - 1]
        dz_after = z[i + 1] - z[i]
        if dz_before < -min_vertical_change and dz_after > min_vertical_change:
            local_minima.append(i)

    # Enforce a minimum distance between local minima.
    merged_minima = []
    for i in local_minima:
        if not merged_minima:
            merged_minima.append(i)
            continue
        if i - merged_minima[-1] < min_separation_frames:
            if z[i] < z[merged_minima[-1]]:
                merged_minima[-1] = i
        else:
            merged_minima.append(i)

    candidates = []
    half_len = TABLE_LENGTH_M / 2.0
    half_wid = TABLE_WIDTH_M / 2.0
    for i in merged_minima:
        i_star, z_star = local_quadratic_refine(z, i, fps)
        _, x_star = local_quadratic_refine(x, i, fps)
        _, y_star = local_quadratic_refine(y, i, fps)
        on_table = (
            (-half_len - table_margin) <= x_star <= (half_len + table_margin) and
            (-half_wid - table_margin) <= y_star <= (half_wid + table_margin)
        )
        near_table = abs(z_star - table_height) <= z_tolerance
        if on_table and near_table:
            candidates.append({
                "frame_local_float": i_star,
                "frame_local_index": int(round(i_star)),
                "frame_global_estimate": int(round(start_frame + i_star)),
                "x": float(x_star),
                "y": float(y_star),
                "z": float(z_star),
                "z_error_to_table": float(abs(z_star - table_height)),
            })

    # Fallback: if no minima matched, try global min if it is plausible.
    if not candidates:
        i = int(np.argmin(z))
        if 0 < i < len(z) - 1:
            i_star, z_star = local_quadratic_refine(z, i, fps)
            _, x_star = local_quadratic_refine(x, i, fps)
            _, y_star = local_quadratic_refine(y, i, fps)
            on_table = (
                (-half_len - table_margin) <= x_star <= (half_len + table_margin) and
                (-half_wid - table_margin) <= y_star <= (half_wid + table_margin)
            )
            near_table = abs(z_star - table_height) <= (z_tolerance * 1.5)
            if on_table and near_table:
                candidates.append({
                    "frame_local_float": i_star,
                    "frame_local_index": int(round(i_star)),
                    "frame_global_estimate": int(round(start_frame + i_star)),
                    "x": float(x_star),
                    "y": float(y_star),
                    "z": float(z_star),
                    "z_error_to_table": float(abs(z_star - table_height)),
                })

    candidates.sort(key=lambda item: item["z_error_to_table"])
    return candidates[:max_bounces_per_shot]


def draw_bounce_map(bounces, out_path):
    half_len = TABLE_LENGTH_M / 2.0
    half_wid = TABLE_WIDTH_M / 2.0

    fig, ax = plt.subplots(figsize=(10, 6))
    table_patch = Rectangle(
        (-half_len, -half_wid),
        TABLE_LENGTH_M,
        TABLE_WIDTH_M,
        facecolor="#dbeee0",
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(table_patch)
    ax.plot([0, 0], [-half_wid, half_wid], linestyle="--", color="gray", linewidth=1.5)

    if bounces:
        color_map = plt.get_cmap("tab20")
        for i, bounce in enumerate(bounces):
            color = color_map(i % color_map.N)
            ax.scatter(
                bounce["x"],
                bounce["y"],
                s=85,
                color=color,
                edgecolors="black",
                linewidths=0.7,
                zorder=3,
            )
            ax.text(
                bounce["x"] + 0.02,
                bounce["y"] + 0.02,
                f"S{bounce['shot_index'] + 1}",
                fontsize=8,
                color="black",
            )
    else:
        ax.text(
            0.0,
            0.0,
            "No bounce points detected",
            ha="center",
            va="center",
            fontsize=12,
            color="#b00020",
        )

    ax.set_xlim(-half_len - 0.15, half_len + 0.15)
    ax.set_ylim(-half_wid - 0.15, half_wid + 0.15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Estimated Ball Bounce Locations ({len(bounces)} total)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_shot_results(shots_dir):
    shot_files = sorted(glob.glob(os.path.join(shots_dir, "shot_*.json")))
    results = []
    for path in shot_files:
        with open(path, "r") as f:
            data = json.load(f)
        results.append(data)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run full-video analysis and output table bounce map."
    )
    parser.add_argument(
        "--base_script",
        type=str,
        default="run_inference_full_video.py",
        help="Base full-video inference script to run (default: run_inference_full_video.py)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory used by the base script (default: output)",
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Skip running base inference and only generate bounce outputs from existing JSON files",
    )
    parser.add_argument(
        "--bounces_json_name",
        type=str,
        default="bounces.json",
        help="Output JSON filename for bounce points (default: bounces.json)",
    )
    parser.add_argument(
        "--bounce_map_name",
        type=str,
        default="bounce_map.png",
        help="Output image filename for bounce map (default: bounce_map.png)",
    )
    parser.add_argument(
        "--table_height",
        type=float,
        default=TABLE_HEIGHT_M,
        help="Expected table height in meters (default: 0.76)",
    )
    parser.add_argument(
        "--z_tolerance",
        type=float,
        default=0.12,
        help="Allowed |z - table_height| for bounce acceptance (default: 0.12)",
    )
    parser.add_argument(
        "--table_margin",
        type=float,
        default=0.03,
        help="Allowed boundary margin outside table edges in meters (default: 0.03)",
    )
    parser.add_argument(
        "--min_vertical_change",
        type=float,
        default=0.0015,
        help="Min vertical step magnitude for local-min bounce detection (default: 0.0015)",
    )
    parser.add_argument(
        "--min_separation_frames",
        type=int,
        default=3,
        help="Min frame distance between bounce candidates (default: 3)",
    )
    parser.add_argument(
        "--max_bounces_per_shot",
        type=int,
        default=2,
        help="Max bounce points retained per shot (default: 2)",
    )

    args, passthrough_args = parser.parse_known_args()

    if args.z_tolerance <= 0:
        parser.error("--z_tolerance must be > 0")
    if args.table_margin < 0:
        parser.error("--table_margin must be >= 0")
    if args.min_vertical_change <= 0:
        parser.error("--min_vertical_change must be > 0")
    if args.min_separation_frames < 1:
        parser.error("--min_separation_frames must be >= 1")
    if args.max_bounces_per_shot < 1:
        parser.error("--max_bounces_per_shot must be >= 1")

    if not args.skip_inference:
        if "--video" not in passthrough_args:
            parser.error(
                "Missing --video in passthrough arguments. Example:\n"
                "  python run_inference_full_video_with_bounces.py --video match.mp4"
            )
        if not os.path.exists(args.base_script):
            parser.error(f"Base script not found: {args.base_script}")
        run_full_video_inference(args.base_script, passthrough_args)
    else:
        print("[1/3] Skipping inference (--skip_inference).")

    print("[2/3] Loading shot outputs and estimating bounce points...")
    summary_path = os.path.join(args.output_dir, "summary.json")
    shots_dir = os.path.join(args.output_dir, "shots")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    if not os.path.isdir(shots_dir):
        raise FileNotFoundError(f"Missing shots directory: {shots_dir}")

    with open(summary_path, "r") as f:
        summary = json.load(f)
    fps = float(summary.get("fps", 60.0))
    if not np.isfinite(fps) or fps <= 0:
        fps = 60.0

    shot_results = load_shot_results(shots_dir)
    if not shot_results:
        raise RuntimeError(f"No shot JSON files found in {shots_dir}")

    bounces = []
    for shot in shot_results:
        traj = shot.get("trajectory_3d", [])
        start_frame = int(shot.get("start_frame", 0))
        shot_index = int(shot.get("shot_index", -1))
        rally_index = int(shot.get("rally_index", -1))
        detected = estimate_bounces_for_shot(
            trajectory_3d=traj,
            fps=fps,
            start_frame=start_frame,
            table_height=args.table_height,
            z_tolerance=args.z_tolerance,
            table_margin=args.table_margin,
            min_vertical_change=args.min_vertical_change,
            min_separation_frames=args.min_separation_frames,
            max_bounces_per_shot=args.max_bounces_per_shot,
        )
        for bounce in detected:
            bounce["shot_index"] = shot_index
            bounce["rally_index"] = rally_index
            bounces.append(bounce)

    bounces_payload = {
        "source_summary": summary_path,
        "fps": fps,
        "table": {
            "length_m": TABLE_LENGTH_M,
            "width_m": TABLE_WIDTH_M,
            "height_m": args.table_height,
        },
        "settings": {
            "z_tolerance": args.z_tolerance,
            "table_margin": args.table_margin,
            "min_vertical_change": args.min_vertical_change,
            "min_separation_frames": args.min_separation_frames,
            "max_bounces_per_shot": args.max_bounces_per_shot,
        },
        "num_shots": len(shot_results),
        "num_bounces": len(bounces),
        "bounces": bounces,
    }

    bounces_json_path = os.path.join(args.output_dir, args.bounces_json_name)
    with open(bounces_json_path, "w") as f:
        json.dump(bounces_payload, f, indent=2)

    print("[3/3] Writing bounce map image...")
    bounce_map_path = os.path.join(args.output_dir, args.bounce_map_name)
    draw_bounce_map(bounces, bounce_map_path)

    print("\nDone.")
    print(f"  Bounce JSON:  {bounces_json_path}")
    print(f"  Bounce image: {bounce_map_path}")
    print(f"  Bounce count: {len(bounces)}")


if __name__ == "__main__":
    main()
