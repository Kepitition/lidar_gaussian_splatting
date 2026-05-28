"""
Configuration for iPhone → Gaussian Splatting scanning pipeline.
Author: Ayberk Tunca

Defines paths, COLMAP parameters, and Gaussian Splatting training settings.
Use quality presets for quick setup, then override individual parameters as needed.
"""

from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class QualityPreset(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


@dataclass
class ColmapConfig:
    """COLMAP SfM parameters (sparse reconstruction only)."""

    camera_model: str = "OPENCV"
    single_camera: bool = True
    max_image_size: int = 3200
    sift_max_num_features: int = 8192

    matcher_type: str = "exhaustive"
    vocab_tree_path: Optional[str] = None

    ba_refine_focal_length: bool = True
    ba_refine_extra_params: bool = True
    min_num_matches: int = 10
    multiple_models: bool = False
    abs_pose_min_num_inliers: int = 10
    abs_pose_min_inlier_ratio: float = 0.1


@dataclass
class GaussianSplattingConfig:
    """3D Gaussian Splatting training parameters."""

    iterations: int = 30_000
    learning_rate_position: float = 0.00016
    learning_rate_color: float = 0.0025
    learning_rate_opacity: float = 0.05
    learning_rate_scaling: float = 0.005
    learning_rate_rotation: float = 0.001

    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    # Threshold for clone/split: 0.0001 is calibrated for sparse COLMAP init.
    # LiDAR init already places Gaussians near optimal positions → smaller gradients
    # → lower threshold needed to trigger densification.
    densify_grad_threshold: float = 0.000001
    densification_interval: int = 100
    opacity_reset_interval: int = 5000
    prune_opacity_threshold: float = 0.005
    max_gaussians: int = 800_000

    sh_degree: int = 3
    white_background: bool = False

    save_iterations: list = field(default_factory=lambda: [7_000, 15_000, 30_000])


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    # Project paths
    project_root: Path = Path(__file__).parent
    image_dir: Path = Path("output/scan/images")
    colmap_output_dir: Path = Path("output/scan/colmap")
    gs_output_dir: Path = Path("output/scan")

    # Quality preset (apply with apply_quality_preset())
    quality: QualityPreset = QualityPreset.HIGH

    # Sub-configs — override any field after construction
    colmap: ColmapConfig = field(default_factory=ColmapConfig)
    gs: GaussianSplattingConfig = field(default_factory=GaussianSplattingConfig)

    # COLMAP binary path (set to None to use pycolmap)
    colmap_binary: Optional[str] = None     # e.g. "C:/COLMAP/COLMAP.bat"
    use_gpu: bool = True

    def __post_init__(self):
        # Resolve paths relative to project root
        self.image_dir = self.project_root / self.image_dir
        self.colmap_output_dir = self.project_root / self.colmap_output_dir
        self.gs_output_dir = self.project_root / self.gs_output_dir

    def apply_quality_preset(self):
        """Apply sensible defaults based on the selected quality level."""
        if self.quality == QualityPreset.LOW:
            self.colmap.sift_max_num_features = 4096
            self.colmap.max_image_size = 1600
            self.colmap.min_num_matches = 10
            self.gs.max_gaussians = 150_000
            self.gs.iterations = 10_000
            self.gs.densify_from_iter = 500
            self.gs.densify_until_iter = 5_000
            self.gs.sh_degree = 1
            self.gs.save_iterations = [5_000, 10_000]

        elif self.quality == QualityPreset.MEDIUM:
            self.colmap.sift_max_num_features = 8192
            self.colmap.max_image_size = 2400
            self.gs.max_gaussians = 400_000
            self.gs.iterations = 20_000
            self.gs.densify_from_iter = 500
            self.gs.densify_until_iter = 10_000
            self.gs.sh_degree = 2
            self.gs.save_iterations = [5_000, 10_000, 20_000]

        elif self.quality == QualityPreset.HIGH:
            # Default: 800K cap, 30K iters, grad_threshold=0.00005 (from base config)
            self.gs.max_gaussians = 800_000
            self.gs.iterations = 30_000
            self.gs.densify_from_iter = 500
            self.gs.densify_until_iter = 15_000
            self.gs.sh_degree = 3
            self.gs.save_iterations = [5_000, 10_000, 20_000, 30_000]

        elif self.quality == QualityPreset.ULTRA:
            self.colmap.sift_max_num_features = 16384
            self.colmap.max_image_size = 4000
            self.gs.max_gaussians = 1_500_000
            self.gs.iterations = 50_000
            self.gs.densify_from_iter = 500
            self.gs.densify_until_iter = 25_000
            self.gs.densification_interval = 50
            self.gs.sh_degree = 3   # gsplat supports max degree 3; degree 4 would error
            self.gs.save_iterations = [5_000, 15_000, 30_000, 50_000]
