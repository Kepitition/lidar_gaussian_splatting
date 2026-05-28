"""
Record3D iPhone Import — converts a Record3D EXR+JPG export to GS training format.

COMPLETELY BYPASSES COLMAP:
  - Camera poses come from ARKit (accurate, metric-scale)
  - Initial point cloud comes from LiDAR depth maps (dense, not sparse SIFT)

Expected export layout (Record3D → EXR + JPG):
  <export_dir>/
    rgb/          — JPEG frames  (0.jpg, 1.jpg, ...)
    depth/        — EXR depth maps in metres (0.exr, 1.exr, ...)
    metadata.json — ARKit poses, per-frame intrinsics, timestamps

Output (ready for GS training without COLMAP):
  <output_dir>/
    cameras.json  — camera poses + intrinsics
    input.ply     — initial point cloud from LiDAR
    images/       — selected RGB frames (copied)

Usage:
  python scripts/iphone_import.py --input R3_exports --output output/room
  python scripts/iphone_import.py --input R3_exports --output output/room --max-frames 80 --max-depth 1.0

Then train (no COLMAP needed):
  python main.py --scan output/room --quality high
"""

# IMPORTANT: must be set before cv2 is imported anywhere
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse
import json
import logging
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coordinate system helpers
# ---------------------------------------------------------------------------
# Record3D stores poses in ARKit convention:
#   - quaternion [qx, qy, qz, qw] — camera-to-world rotation
#   - [tx, ty, tz] — camera centre in world space
#   - ARKit axes: X right, Y up, Z toward viewer
#
# Our cameras.json uses OpenCV / COLMAP convention:
#   - rotation R — world-to-camera (3×3)
#   - translation T = −R @ camera_centre
#   - OpenCV axes: X right, Y down, Z into scene
#
# Conversion: flip Y and Z  →  FLIP = diag([1, −1, −1])

_FLIP = np.diag([1., -1., -1.])   # ARKit ↔ OpenCV camera axes


def _quat_to_rotmat(q):
    """[qx, qy, qz, qw] → 3×3 rotation matrix (float64)."""
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qw*qz),   2*(qx*qz + qw*qy)],
        [  2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz),   2*(qy*qz - qw*qx)],
        [  2*(qx*qz - qw*qy),   2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


def _arkit_to_world_to_cam(pose7):
    """
    Convert Record3D pose [qx,qy,qz,qw, tx,ty,tz] (ARKit cam-to-world)
    → (R_wc, T_wc, C) in OpenCV convention.

    R_wc : (3,3) world-to-camera rotation
    T_wc : (3,)  world-to-camera translation  = −R_wc @ C
    C    : (3,)  camera centre in world space
    """
    qx, qy, qz, qw, tx, ty, tz = pose7
    R_cw_arkit = _quat_to_rotmat([qx, qy, qz, qw])   # ARKit cam → world
    C = np.array([tx, ty, tz], dtype=np.float64)

    # Camera-to-world in OpenCV convention
    # pt_world = R_cw_arkit @ pt_arkit_cam + C
    # pt_arkit_cam = _FLIP @ pt_opencv_cam   (flip Y and Z)
    # → R_cw_opencv = R_cw_arkit @ _FLIP
    R_cw_opencv = R_cw_arkit @ _FLIP

    # World-to-camera (OpenCV)
    R_wc = R_cw_opencv.T
    T_wc = -R_wc @ C
    return R_wc, T_wc, C


# ---------------------------------------------------------------------------
# Frame selection (motion-based)
# ---------------------------------------------------------------------------

def select_frames(poses, max_frames=100, min_trans_m=0.004, min_angle_deg=1.0):
    """
    Select up to max_frames cameras with guaranteed spatial coverage.

    Two-pass approach:
      Pass 1 — motion filter: discard near-duplicate frames where the camera
               barely moved (< min_trans_m metres AND < min_angle_deg degrees).
               Produces a candidate list of meaningfully distinct viewpoints.

      Pass 2 — Farthest Point Sampling (FPS) on 3D camera positions:
               iteratively pick the candidate whose position is farthest from
               all already-selected cameras.  This guarantees spatial spread
               regardless of how fast or slow the user moved through each area —
               a wall you walked slowly past won't be over-represented just
               because it produced more frames.

    Example: 1800 frames → 900 candidates → 200 spatially spread selections.
    Even if 400 candidates are from one wall, FPS will only pick ~50 of them
    before the other walls become the "farthest" options.
    """
    n = len(poses)
    if n == 0:
        return []

    # ── Pass 1: motion filter ──────────────────────────────────────────────
    candidates = [0]
    last_pos  = np.array(poses[0][4:7], dtype=np.float64)
    last_quat = np.array(poses[0][:4],  dtype=np.float64)

    for i in range(1, n):
        pos  = np.array(poses[i][4:7], dtype=np.float64)
        quat = np.array(poses[i][:4],  dtype=np.float64)

        trans = np.linalg.norm(pos - last_pos)
        dot   = float(np.clip(abs(np.dot(quat, last_quat)), 0.0, 1.0))
        angle = np.degrees(2.0 * np.arccos(dot))

        if trans >= min_trans_m or angle >= min_angle_deg:
            candidates.append(i)
            last_pos  = pos
            last_quat = quat

    # Always include the last frame
    if candidates[-1] != n - 1:
        candidates.append(n - 1)

    logger.info(f"  Motion filter: {n} frames → {len(candidates)} candidates")

    if len(candidates) <= max_frames:
        return candidates

    # ── Pass 2: Farthest Point Sampling on camera positions ────────────────
    # Extract (x, y, z) positions for all candidates
    positions = np.array([poses[i][4:7] for i in candidates], dtype=np.float64)
    n_cands = len(candidates)

    selected_idx = [0]  # start with first candidate
    # min distance from each candidate to the nearest already-selected camera
    min_dists = np.full(n_cands, np.inf)

    for _ in range(max_frames - 1):
        last = selected_idx[-1]
        # Update minimum distances using the newly added point
        dists_to_last = np.linalg.norm(positions - positions[last], axis=1)
        np.minimum(min_dists, dists_to_last, out=min_dists)
        # Pick the candidate farthest from all selected so far
        next_idx = int(np.argmax(min_dists))
        selected_idx.append(next_idx)
        min_dists[next_idx] = 0.0  # mark as selected

    selected_idx.sort()  # restore temporal order for deterministic depth loading
    selected = [candidates[i] for i in selected_idx]
    logger.info(f"  Farthest-point sampling: {len(candidates)} candidates → {len(selected)} frames")
    return selected


# ---------------------------------------------------------------------------
# Depth map reading  (cv2 with OPENCV_IO_ENABLE_OPENEXR flag)
# ---------------------------------------------------------------------------

def read_depth_exr(path: Path) -> np.ndarray:
    """
    Read a Record3D EXR depth map.
    Returns float32 array (H, W) in metres.

    Note: cv2 reads EXR in BGR channel order; depth is in the R channel (index 2).
    """
    import cv2
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise IOError(f"cv2 could not open: {path}")
    # EXR has 3 channels read as BGR; actual depth lives in R = channel index 2
    depth = raw[:, :, 2].astype(np.float32)
    return depth


# ---------------------------------------------------------------------------
# Depth unprojection  →  3-D world points
# ---------------------------------------------------------------------------

def depth_to_world_points(
    depth:   np.ndarray,    # (dh, dw) float32, metres
    rgb:     np.ndarray,    # (h,  w,  3) uint8
    fx_d: float, fy_d: float, cx_d: float, cy_d: float,
    R_wc:    np.ndarray,    # (3, 3)
    C:       np.ndarray,    # (3,)  camera centre
    max_depth: float = 2.0,
    subsample: int   = 2,
) -> tuple:
    """
    Unproject depth pixels to 3-D world points and sample their RGB colour.

    Returns:
        positions (N, 3) float32
        colors    (N, 3) uint8
    """
    dh, dw = depth.shape
    h,  w  = rgb.shape[:2]

    # Pixel grid (subsampled to save memory & time)
    v_idx, u_idx = np.meshgrid(
        np.arange(0, dh, subsample),
        np.arange(0, dw, subsample),
        indexing='ij',
    )
    u = u_idx.ravel()
    v = v_idx.ravel()
    d = depth[v, u]

    # Keep only pixels with plausible depth
    valid = (d > 0.02) & (d < max_depth)
    u, v, d = u[valid], v[valid], d[valid]

    if len(d) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    # Unproject to OpenCV camera space
    pts_cam = np.stack([
        (u - cx_d) / fx_d * d,   # X (right)
        (v - cy_d) / fy_d * d,   # Y (down)
        d,                         # Z (forward)
    ], axis=1)                     # (N, 3)

    # Camera → world  (R_cw = R_wc.T)
    R_cw = R_wc.T
    pts_world = (R_cw @ pts_cam.T).T + C   # (N, 3)

    # Sample colour from RGB at corresponding pixel
    u_rgb = np.clip((u / dw * w ).astype(np.int32), 0, w  - 1)
    v_rgb = np.clip((v / dh * h ).astype(np.int32), 0, h  - 1)
    colors = rgb[v_rgb, u_rgb]   # (N, 3) uint8

    return pts_world.astype(np.float32), colors


# ---------------------------------------------------------------------------
# PLY writer
# ---------------------------------------------------------------------------

def write_ply(positions: np.ndarray, colors: np.ndarray, out_path: Path):
    """Write coloured point cloud as binary little-endian PLY."""
    N = len(positions)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        hdr = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {N}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property float nx\nproperty float ny\nproperty float nz\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        f.write(hdr.encode("utf-8"))
        colors_u8 = np.clip(colors, 0, 255).astype(np.uint8)
        for i in range(N):
            f.write(struct.pack("<3f", *positions[i]))
            f.write(struct.pack("<3f", 0.0, 0.0, 0.0))   # normals zero
            f.write(struct.pack("<3B", *colors_u8[i]))
    logger.info(f"Wrote {N:,} points → {out_path}")


# ---------------------------------------------------------------------------
# Main import routine
# ---------------------------------------------------------------------------

def import_record3d(
    input_dir:      Path,
    output_dir:     Path,
    max_frames:     int   = 100,
    max_depth:      float = 2.0,
    depth_subsample: int  = 2,
    min_trans_m:    float = 0.004,
    min_angle_deg:  float = 1.0,
) -> Path:
    """
    Convert a Record3D EXR+JPG export to GS training format.

    Returns the output directory path.
    """
    # ── Load metadata ───────────────────────────────────────────────────
    meta_path = input_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {input_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    poses      = meta["poses"]                    # list of 1106 × [qx,qy,qz,qw,tx,ty,tz]
    intrinsics = meta["perFrameIntrinsicCoeffs"]  # list of 1106 × [fx,fy,cx,cy]  (RGB res)
    w,  h  = meta["w"],  meta["h"]               # RGB  image size (720 × 960)
    dw, dh = meta["dw"], meta["dh"]              # Depth image size (192 × 256)

    logger.info(f"Total frames: {len(poses)}")
    logger.info(f"RGB  resolution: {w} × {h}")
    logger.info(f"Depth resolution: {dw} × {dh}")

    # ── Select frames ────────────────────────────────────────────────────
    selected = select_frames(
        poses,
        max_frames=max_frames,
        min_trans_m=min_trans_m,
        min_angle_deg=min_angle_deg,
    )
    logger.info(f"Selected {len(selected)} frames from {len(poses)} total")

    # ── Prepare output directories ───────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    # Always clear images/ and cameras.json before writing — ensures no stale
    # frames from a previous import (different --max-frames) remain.
    if images_dir.exists():
        shutil.rmtree(images_dir)
        logger.info("Cleared old images/ folder")
    images_dir.mkdir(exist_ok=True)

    rgb_dir   = input_dir / "rgb"
    depth_dir = input_dir / "depth"

    cameras_json  = {}
    all_positions = []
    all_colors    = []
    skipped       = 0

    # ── Process each selected frame ──────────────────────────────────────
    for cam_id, frame_idx in enumerate(selected):

        # RGB
        rgb_src = rgb_dir / f"{frame_idx}.jpg"
        if not rgb_src.exists():
            logger.warning(f"  Missing rgb/{frame_idx}.jpg — skipping")
            skipped += 1
            continue

        rgb = np.array(Image.open(rgb_src).convert("RGB"))
        frame_name = f"{frame_idx:05d}.jpg"

        # Per-frame intrinsics (RGB resolution)
        fx, fy, cx, cy = intrinsics[frame_idx]

        # Pose → OpenCV world-to-camera
        R_wc, T_wc, C = _arkit_to_world_to_cam(poses[frame_idx])

        # Record camera
        cameras_json[frame_name] = {
            "id":        cam_id,
            "camera_id": 0,
            "model":     "PINHOLE",
            "width":     w,
            "height":    h,
            "params":    [fx, fy, cx, cy],
            "rotation":  R_wc.tolist(),
            "translation": T_wc.tolist(),
            "position":  C.tolist(),
        }

        # Copy image to output
        shutil.copy2(rgb_src, images_dir / frame_name)

        # ── Depth → 3-D points ──────────────────────────────────────────
        depth_src = depth_dir / f"{frame_idx}.exr"
        if depth_src.exists():
            try:
                depth = read_depth_exr(depth_src)

                # Scale RGB intrinsics to depth image resolution
                scale_x = dw / w
                scale_y = dh / h
                fx_d = fx * scale_x;  cx_d = cx * scale_x
                fy_d = fy * scale_y;  cy_d = cy * scale_y

                pts, cols = depth_to_world_points(
                    depth, rgb,
                    fx_d, fy_d, cx_d, cy_d,
                    R_wc, C,
                    max_depth=max_depth,
                    subsample=depth_subsample,
                )
                if len(pts):
                    all_positions.append(pts)
                    all_colors.append(cols)

            except Exception as e:
                logger.warning(f"  Depth error for frame {frame_idx}: {e}")

        if (cam_id + 1) % 10 == 0:
            logger.info(f"  Processed {cam_id + 1}/{len(selected)} frames ...")

    n_cams = len(cameras_json)
    logger.info(f"Cameras processed: {n_cams}  (skipped {skipped})")

    # ── Write cameras.json ───────────────────────────────────────────────
    cam_json_path = output_dir / "cameras.json"
    with open(cam_json_path, "w") as f:
        json.dump(cameras_json, f, indent=2)
    logger.info(f"Wrote {cam_json_path}")

    # ── Merge + subsample point cloud ────────────────────────────────────
    if all_positions:
        positions = np.concatenate(all_positions, axis=0)
        colors    = np.concatenate(all_colors,    axis=0)
        logger.info(f"Raw LiDAR point cloud: {len(positions):,} points")

        # Cap at 500 K to keep GS init fast and stay within 8 GB VRAM
        if len(positions) > 500_000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(positions), 500_000, replace=False)
            positions, colors = positions[idx], colors[idx]
            logger.info(f"Subsampled to 500,000 points")

        write_ply(positions, colors, output_dir / "input.ply")
    else:
        logger.warning("No depth data — writing a dummy single-point PLY (training may be poor)")
        write_ply(np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8),
                  output_dir / "input.ply")

    # ── Done ─────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("═" * 60)
    logger.info("Import complete!")
    logger.info(f"  Cameras : {output_dir / 'cameras.json'}  ({n_cams} views)")
    logger.info(f"  Images  : {images_dir}")
    logger.info(f"  Points  : {output_dir / 'input.ply'}")
    logger.info("")
    logger.info("Next step — train:")
    logger.info(f"  python main.py --scan {output_dir} --quality high")
    logger.info("═" * 60)

    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Import Record3D EXR+JPG export for Gaussian Splatting training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Keyboard / small object (close range)
  python scripts/iphone_import.py --input R3_exports --output output/gs_output --max-depth 1.0

  # Room / large space
  python scripts/iphone_import.py --input R3_exports/R3_room --output output/room --max-depth 4.0 --max-frames 400
        """,
    )
    parser.add_argument("--input",  required=True,
                        help="Record3D export directory (contains rgb/, depth/, metadata.json)")
    parser.add_argument("--output", default="output/gs_output",
                        help="Output directory for GS training (default: output/gs_output)")
    parser.add_argument("--max-frames", type=int, default=200,
                        help="Maximum frames to use (default: 200; use 400 for large rooms)")
    parser.add_argument("--max-depth", type=float, default=2.0,
                        help="Maximum depth in metres to include as 3D points "
                             "(default: 2.0; use 1.0 for keyboards/objects, 4.0 for rooms)")
    parser.add_argument("--depth-subsample", type=int, default=2,
                        help="Depth pixel subsampling step (default: 2 = every other pixel; "
                             "use 1 for maximum detail, 4 for faster import)")
    parser.add_argument("--min-trans", type=float, default=0.004,
                        help="Minimum camera translation (metres) between selected frames (default: 0.004)")
    parser.add_argument("--min-angle", type=float, default=1.0,
                        help="Minimum camera rotation (degrees) between selected frames (default: 1.0)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    import_record3d(
        input_dir       = Path(args.input).resolve(),
        output_dir      = Path(args.output).resolve(),
        max_frames      = args.max_frames,
        max_depth       = args.max_depth,
        depth_subsample = args.depth_subsample,
        min_trans_m     = args.min_trans,
        min_angle_deg   = args.min_angle,
    )


if __name__ == "__main__":
    main()
