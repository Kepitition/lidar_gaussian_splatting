"""
COLMAP SfM Pipeline (sparse reconstruction only).
Author: Ayberk Tunca

Runs COLMAP sparse reconstruction:
  1. Feature extraction (SIFT)
  2. Feature matching (exhaustive / sequential / vocab_tree)
  3. Sparse reconstruction (incremental mapper)

Supports both CLI-based COLMAP and pycolmap Python bindings.
"""

import logging
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PipelineConfig

logger = logging.getLogger(__name__)


class ColmapPipeline:
    """Orchestrates COLMAP feature extraction, matching, and reconstruction."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.cc = config.colmap  # ColmapConfig shortcut

        self.workspace = config.colmap_output_dir
        self.database_path = self.workspace / "database.db"
        self.sparse_dir = self.workspace / "sparse"
        self.image_dir = config.image_dir

        # Determine execution mode
        self.use_pycolmap = config.colmap_binary is None
        if self.use_pycolmap:
            try:
                import pycolmap
                self.pycolmap = pycolmap
                logger.info("Using pycolmap Python bindings.")
            except ImportError:
                raise RuntimeError(
                    "pycolmap is not installed and no COLMAP binary path provided. "
                    "Install pycolmap: pip install pycolmap  "
                    "OR set config.colmap_binary to your COLMAP executable path."
                )
        else:
            if not Path(config.colmap_binary).exists():
                raise FileNotFoundError(f"COLMAP binary not found: {config.colmap_binary}")
            logger.info(f"Using COLMAP binary: {config.colmap_binary}")

    def setup_workspace(self):
        """Create output directories."""
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Workspace ready: {self.workspace}")

    # ------------------------------------------------------------------
    # Step 1: Feature Extraction
    # ------------------------------------------------------------------
    def extract_features(self):
        """Extract SIFT features from all images."""
        logger.info("Step 1/3: Extracting features...")

        if self.use_pycolmap:
            self.pycolmap.extract_features(
                database_path=str(self.database_path),
                image_path=str(self.image_dir),
                camera_mode="SINGLE" if self.cc.single_camera else "AUTO",
                camera_model=self.cc.camera_model,
                sift_options={
                    "max_num_features": self.cc.sift_max_num_features,
                    "max_image_size": self.cc.max_image_size,
                },
            )
        else:
            cmd = [
                self.config.colmap_binary, "feature_extractor",
                "--database_path", str(self.database_path),
                "--image_path", str(self.image_dir),
                "--ImageReader.camera_model", self.cc.camera_model,
                "--ImageReader.single_camera", str(int(self.cc.single_camera)),
                "--SiftExtraction.max_image_size", str(self.cc.max_image_size),
                "--SiftExtraction.max_num_features", str(self.cc.sift_max_num_features),
            ]
            if self.config.use_gpu:
                cmd += ["--FeatureExtraction.use_gpu", "1"]
            self._run_cmd(cmd, "Feature extraction")

        logger.info("Feature extraction complete.")

    # ------------------------------------------------------------------
    # Step 2: Feature Matching
    # ------------------------------------------------------------------
    def match_features(self):
        """Match features across image pairs."""
        matcher = self.cc.matcher_type
        logger.info(f"Step 2/3: Matching features ({matcher})...")

        if self.use_pycolmap:
            if matcher == "exhaustive":
                self.pycolmap.match_exhaustive(
                    database_path=str(self.database_path),
                )
            elif matcher == "sequential":
                self.pycolmap.match_sequential(
                    database_path=str(self.database_path),
                )
            elif matcher == "vocab_tree":
                if not self.cc.vocab_tree_path:
                    raise ValueError("vocab_tree matcher requires vocab_tree_path in config.")
                self.pycolmap.match_vocab_tree(
                    database_path=str(self.database_path),
                    vocab_tree_path=self.cc.vocab_tree_path,
                )
            else:
                raise ValueError(f"Unknown matcher type: {matcher}")
        else:
            matcher_map = {
                "exhaustive": "exhaustive_matcher",
                "sequential": "sequential_matcher",
                "vocab_tree": "vocab_tree_matcher",
            }
            cmd_name = matcher_map.get(matcher)
            if not cmd_name:
                raise ValueError(f"Unknown matcher type: {matcher}")

            cmd = [
                self.config.colmap_binary, cmd_name,
                "--database_path", str(self.database_path),
            ]
            if matcher == "vocab_tree" and self.cc.vocab_tree_path:
                cmd += ["--VocabTreeMatching.vocab_tree_path", self.cc.vocab_tree_path]
            if self.config.use_gpu:
                cmd += ["--FeatureMatching.use_gpu", "1"]
            self._run_cmd(cmd, "Feature matching")

        logger.info("Feature matching complete.")

    # ------------------------------------------------------------------
    # Step 3: Sparse Reconstruction (Incremental Mapper)
    # ------------------------------------------------------------------
    def reconstruct_sparse(self):
        """Run incremental SfM to compute camera poses and sparse point cloud."""
        logger.info("Step 3/3: Running sparse reconstruction (incremental mapper)...")

        if self.use_pycolmap:
            maps = self.pycolmap.incremental_mapping(
                database_path=str(self.database_path),
                image_path=str(self.image_dir),
                output_path=str(self.sparse_dir),
                options={
                    "ba_refine_focal_length": self.cc.ba_refine_focal_length,
                    "ba_refine_extra_params": self.cc.ba_refine_extra_params,
                    "min_num_matches": self.cc.min_num_matches,
                },
            )
            if not maps:
                raise RuntimeError("Sparse reconstruction failed — no models produced.")
            logger.info(f"Sparse reconstruction produced {len(maps)} model(s).")
        else:
            cmd = [
                self.config.colmap_binary, "mapper",
                "--database_path", str(self.database_path),
                "--image_path", str(self.image_dir),
                "--output_path", str(self.sparse_dir),
                "--Mapper.ba_refine_focal_length", str(int(self.cc.ba_refine_focal_length)),
                "--Mapper.ba_refine_extra_params", str(int(self.cc.ba_refine_extra_params)),
                "--Mapper.min_num_matches", str(self.cc.min_num_matches),
                "--Mapper.multiple_models", str(int(self.cc.multiple_models)),
                "--Mapper.abs_pose_min_num_inliers", str(self.cc.abs_pose_min_num_inliers),
                "--Mapper.abs_pose_min_inlier_ratio", str(self.cc.abs_pose_min_inlier_ratio),
            ]
            self._run_cmd(cmd, "Sparse reconstruction")

        logger.info("Sparse reconstruction complete.")
        return self.sparse_dir

    # ------------------------------------------------------------------
    # Find best sparse model (most registered images)
    # ------------------------------------------------------------------
    def find_best_sparse_model(self) -> int:
        """Find the sparse model with the most registered images."""
        best_id = 0
        best_count = -1

        for model_dir in sorted(self.sparse_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            images_bin = model_dir / "images.bin"
            if not images_bin.exists():
                continue
            with open(images_bin, "rb") as f:
                num_images = struct.unpack("<Q", f.read(8))[0]
            logger.info(f"  Sparse model {model_dir.name}: {num_images} registered images")
            if num_images > best_count:
                best_count = num_images
                best_id = int(model_dir.name)

        logger.info(f"  Best model: {best_id} ({best_count} images)")
        return best_id

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------
    def run(self) -> Path:
        """Execute COLMAP sparse reconstruction."""
        self.setup_workspace()
        self.extract_features()
        self.match_features()
        sparse_path = self.reconstruct_sparse()
        self.best_sparse_model_id = self.find_best_sparse_model()
        return sparse_path

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _run_cmd(self, cmd: list, step_name: str):
        """Run a COLMAP CLI command with logging."""
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200,
        )
        if result.returncode != 0:
            logger.error(f"{step_name} failed:\n{result.stderr}")
            raise RuntimeError(f"{step_name} failed with return code {result.returncode}")
        if result.stdout:
            logger.debug(result.stdout[-2000:])  # Tail of output

    def get_reconstruction_stats(self, sparse_model_id: int = 0) -> dict:
        """Read basic stats from the sparse reconstruction."""
        model_path = self.sparse_dir / str(sparse_model_id)
        stats = {"model_path": str(model_path)}

        if self.use_pycolmap:
            reconstruction = self.pycolmap.Reconstruction()
            reconstruction.read(str(model_path))
            stats["num_cameras"] = reconstruction.num_cameras()
            stats["num_images"] = reconstruction.num_images()
            stats["num_registered_images"] = reconstruction.num_reg_images()
            stats["num_points3D"] = reconstruction.num_points3D()
        else:
            # Count files as rough indicator
            for fname in ["cameras.bin", "images.bin", "points3D.bin"]:
                fpath = model_path / fname
                stats[fname] = fpath.exists()

        return stats
