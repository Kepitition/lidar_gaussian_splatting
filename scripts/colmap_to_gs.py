"""
COLMAP-to-Gaussian Splatting data converter.
Author: Ayberk Tunca

Converts COLMAP sparse reconstruction output into the format expected
by 3D Gaussian Splatting training:

  output/
    cameras.json          — camera intrinsics
    images/               — undistorted images
    sparse/
      points3D.ply        — initial point cloud
    input.ply             — alias to points3D.ply

This follows the dataset format used by the original 3DGS implementation
(https://github.com/graphdeco-inria/gaussian-splatting).
"""

import json
import logging
import struct
import shutil
from pathlib import Path
from collections import namedtuple
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PipelineConfig

logger = logging.getLogger(__name__)

# COLMAP binary format structures
CameraModel = namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
CAMERA_MODELS = {
    0: CameraModel(0, "SIMPLE_PINHOLE", 3),
    1: CameraModel(1, "PINHOLE", 4),
    2: CameraModel(2, "SIMPLE_RADIAL", 4),
    3: CameraModel(3, "RADIAL", 5),
    4: CameraModel(4, "OPENCV", 8),
    5: CameraModel(5, "OPENCV_FISHEYE", 8),
    6: CameraModel(6, "FULL_OPENCV", 12),
    7: CameraModel(7, "FOV", 5),
    8: CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    9: CameraModel(9, "RADIAL_FISHEYE", 5),
    10: CameraModel(10, "THIN_PRISM_FISHEYE", 12),
}


def read_cameras_binary(path: Path) -> dict:
    """Read cameras.bin from COLMAP sparse model."""
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            num_params = CAMERA_MODELS[model_id].num_params
            params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
            cameras[cam_id] = {
                "id": cam_id,
                "model": CAMERA_MODELS[model_id].model_name,
                "width": width,
                "height": height,
                "params": list(params),
            }
    return cameras


def read_images_binary(path: Path) -> dict:
    """Read images.bin from COLMAP sparse model."""
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]

            # Read image name (null-terminated string)
            name_chars = []
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name_chars.append(ch.decode("utf-8"))
            name = "".join(name_chars)

            # Read 2D points (we skip them but need to advance the file pointer)
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            # Each point2D: x(double), y(double), point3D_id(long long)
            f.read(num_points2D * 24)

            images[image_id] = {
                "id": image_id,
                "qvec": [qw, qx, qy, qz],
                "tvec": [tx, ty, tz],
                "camera_id": camera_id,
                "name": name,
            }
    return images


def read_points3D_binary(path: Path) -> dict:
    """Read points3D.bin from COLMAP sparse model."""
    points = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            point_id = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            error = struct.unpack("<d", f.read(8))[0]

            # Track length
            track_length = struct.unpack("<Q", f.read(8))[0]
            # Each track element: image_id(uint32) + point2D_idx(uint32)
            f.read(track_length * 8)

            points[point_id] = {
                "id": point_id,
                "xyz": list(xyz),
                "rgb": list(rgb),
                "error": error,
            }
    return points


def qvec_to_rotmat(qvec):
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])
    return R


def write_points3D_ply(points: dict, output_path: Path, max_reproj_error: float = 2.0):
    """
    Write sparse points to PLY format for Gaussian Splatting input.

    Filters out points with high reprojection error — COLMAP marks these with
    large error values and they represent triangulation noise, not real geometry.
    A threshold of 2.0 px is conservative; 1.0 px gives a cleaner but sparser cloud.
    """
    good_points = {k: v for k, v in points.items() if v["error"] <= max_reproj_error}
    n_total, n_good = len(points), len(good_points)
    if n_good < n_total:
        logger.info(f"  Point quality filter: {n_total} → {n_good} points "
                    f"(removed {n_total - n_good} with reprojection error > {max_reproj_error}px)")
    logger.info(f"Writing {n_good} points to {output_path}")

    with open(output_path, "wb") as f:
        # PLY header
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n_good}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float nx\n"
            "property float ny\n"
            "property float nz\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("utf-8"))

        for pt in good_points.values():
            x, y, z = pt["xyz"]
            r, g, b = pt["rgb"]
            # Write position, zero normals, and color
            f.write(struct.pack("<3f", x, y, z))
            f.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            f.write(struct.pack("<3B", r, g, b))


def _find_best_sparse_model(sparse_root: Path) -> Path:
    """
    Find the sparse model sub-directory with the most registered images.
    COLMAP can produce multiple models (sparse/0, sparse/1, …).
    We pick the one with the most images — that's the most complete reconstruction.
    """
    best_path = None
    best_count = -1

    for model_dir in sorted(sparse_root.iterdir()):
        images_bin = model_dir / "images.bin"
        if not images_bin.exists():
            continue
        # Read just the image count (first 8 bytes)
        with open(images_bin, "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
        logger.info(f"  Sparse model {model_dir.name}: {num_images} images")
        if num_images > best_count:
            best_count = num_images
            best_path = model_dir

    if best_path is None:
        raise FileNotFoundError(f"No valid sparse models found in {sparse_root}")

    logger.info(f"  Selected model {best_path.name} ({best_count} images)")
    return best_path


def convert_colmap_to_gs(config: PipelineConfig, sparse_model_id: int = None):
    """
    Convert COLMAP output to Gaussian Splatting input format.

    Creates the directory structure expected by 3DGS:
      gs_output/
        input.ply              — initial sparse point cloud
        images/                — input images (copied or symlinked)
        sparse/0/              — COLMAP sparse model (cameras, images, points3D)
        cameras.json           — camera parameters in JSON

    If sparse_model_id is None (default), automatically picks the model with
    the most registered images.
    """
    sparse_root = config.colmap_output_dir / "sparse"

    if sparse_model_id is not None:
        sparse_path = sparse_root / str(sparse_model_id)
    else:
        sparse_path = _find_best_sparse_model(sparse_root)

    gs_dir = config.gs_output_dir
    gs_dir.mkdir(parents=True, exist_ok=True)

    # Validate sparse model exists
    required_files = ["cameras.bin", "images.bin", "points3D.bin"]
    for fname in required_files:
        if not (sparse_path / fname).exists():
            raise FileNotFoundError(
                f"Missing {fname} in {sparse_path}. "
                "Run COLMAP sparse reconstruction first."
            )

    logger.info(f"Converting COLMAP model from {sparse_path}")

    # Read COLMAP data
    cameras = read_cameras_binary(sparse_path / "cameras.bin")
    images = read_images_binary(sparse_path / "images.bin")
    points3D = read_points3D_binary(sparse_path / "points3D.bin")

    logger.info(f"  Cameras: {len(cameras)}, Images: {len(images)}, Points: {len(points3D)}")

    # 1. Copy sparse model
    gs_sparse = gs_dir / "sparse" / "0"
    gs_sparse.mkdir(parents=True, exist_ok=True)
    for fname in required_files:
        shutil.copy2(sparse_path / fname, gs_sparse / fname)
    logger.info(f"  Copied sparse model to {gs_sparse}")

    # 2. Write initial point cloud as PLY
    ply_path = gs_dir / "input.ply"
    write_points3D_ply(points3D, ply_path)

    # 3. Copy images
    gs_images = gs_dir / "images"
    gs_images.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_data in images.values():
        src = config.image_dir / img_data["name"]
        dst = gs_images / img_data["name"]
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        else:
            logger.warning(f"  Image not found: {img_data['name']}")
    logger.info(f"  Copied {copied} images to {gs_images}")

    # 4. Write cameras.json — scale intrinsics to match actual image dimensions
    #    (undistorted images may be resized by COLMAP's --max_image_size)
    from PIL import Image as PILImage

    cameras_json = {}
    for img_id, img_data in images.items():
        cam = cameras[img_data["camera_id"]]
        R = qvec_to_rotmat(img_data["qvec"])
        T = np.array(img_data["tvec"])

        # Camera-to-world transform
        R_inv = R.T
        C = -R_inv @ T  # Camera center in world coordinates

        # Check actual image size on disk (may differ from COLMAP's cam width/height)
        actual_img = gs_images / img_data["name"]
        colmap_w, colmap_h = cam["width"], cam["height"]
        params = list(cam["params"])  # Make a copy

        if actual_img.exists():
            with PILImage.open(actual_img) as im:
                actual_w, actual_h = im.size
            if actual_w != colmap_w or actual_h != colmap_h:
                # Scale intrinsics proportionally
                sx = actual_w / colmap_w
                sy = actual_h / colmap_h
                logger.info(f"  {img_data['name']}: scaling intrinsics "
                            f"{colmap_w}x{colmap_h} → {actual_w}x{actual_h} "
                            f"(sx={sx:.4f}, sy={sy:.4f})")
                # For OPENCV model: params = [fx, fy, cx, cy, k1, k2, p1, p2]
                # For PINHOLE: params = [fx, fy, cx, cy]
                # Scale fx, cx by sx; fy, cy by sy
                params[0] *= sx  # fx
                params[1] *= sy  # fy
                params[2] *= sx  # cx
                params[3] *= sy  # cy
                colmap_w, colmap_h = actual_w, actual_h

        cameras_json[img_data["name"]] = {
            "id": img_data["id"],
            "camera_id": img_data["camera_id"],
            "model": cam["model"],
            "width": colmap_w,
            "height": colmap_h,
            "params": params,
            "rotation": R.tolist(),
            "translation": img_data["tvec"],
            "position": C.tolist(),
        }

    json_path = gs_dir / "cameras.json"
    with open(json_path, "w") as f:
        json.dump(cameras_json, f, indent=2)
    logger.info(f"  Wrote camera data to {json_path}")

    logger.info(f"Conversion complete. GS input directory: {gs_dir}")
    return gs_dir
