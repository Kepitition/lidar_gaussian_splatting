"""
Gaussian Splatting training using gsplat.
Author: Ayberk Tunca

Self-contained 3D Gaussian Splatting trainer built on gsplat (Apache 2.0).
No external repo needed — just `pip install gsplat`.

Pipeline:
  1. Load COLMAP cameras + sparse points from the converted GS dataset
  2. Initialize Gaussians from the sparse point cloud
  3. Train via differentiable rasterization (gsplat)
  4. Save trained model as PLY

References:
  - gsplat: https://github.com/nerfstudio-project/gsplat
  - 3DGS paper: Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", 2023
"""

import json
import logging
import math
import random
import struct
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cameras_and_images(gs_dir: Path) -> dict:
    """
    Load camera parameters and images from the converted GS dataset.

    Returns dict with:
        cameras: list of dicts with keys:
            image_name, width, height, fx, fy, cx, cy,
            R (3x3 world-to-cam rotation), T (3, translation),
            image (H,W,3 float32 tensor, 0-1 range)
    """
    cameras_json = gs_dir / "cameras.json"
    images_dir = gs_dir / "images"

    with open(cameras_json, "r") as f:
        cam_data = json.load(f)

    cameras = []
    for name, cam in cam_data.items():
        img_path = images_dir / name
        if not img_path.exists():
            logger.warning(f"Image not found, skipping: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0  # (H, W, 3)

        params = cam["params"]
        w, h = cam["width"], cam["height"]
        model = cam["model"]

        # Extract focal length and principal point based on camera model
        if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        elif model in ("PINHOLE", "RADIAL", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
            fx, fy = params[0], params[1]
            cx, cy = params[2], params[3]
        else:
            fx = fy = params[0]
            cx, cy = w / 2, h / 2

        R = np.array(cam["rotation"])       # 3x3 world-to-camera
        T = np.array(cam["translation"])     # 3, world-to-camera translation

        cameras.append({
            "image_name": name,
            "width": w,
            "height": h,
            "fx": fx, "fy": fy,
            "cx": cx, "cy": cy,
            "R": torch.from_numpy(R).float(),
            "T": torch.from_numpy(T).float(),
            "image": img_tensor,
        })

    logger.info(f"Loaded {len(cameras)} cameras from {gs_dir}")
    return cameras


def load_initial_points(gs_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load initial sparse points from input.ply (binary little-endian)."""
    ply_path = gs_dir / "input.ply"

    positions = []
    colors = []

    with open(ply_path, "rb") as f:
        # Parse PLY header
        header_lines = []
        num_vertices = 0
        while True:
            line = f.readline().decode("utf-8").strip()
            header_lines.append(line)
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
            if line == "end_header":
                break

        # Read binary data: 3 floats (xyz) + 3 floats (normals) + 3 bytes (rgb)
        for _ in range(num_vertices):
            xyz = struct.unpack("<3f", f.read(12))
            _normals = struct.unpack("<3f", f.read(12))
            rgb = struct.unpack("<3B", f.read(3))
            positions.append(xyz)
            colors.append([c / 255.0 for c in rgb])

    positions = np.array(positions, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)
    logger.info(f"Loaded {len(positions)} initial points from {ply_path}")
    return positions, colors


# ---------------------------------------------------------------------------
# Gaussian model
# ---------------------------------------------------------------------------

class GaussianModel(nn.Module):
    """
    Stores the parameters of all 3D Gaussians:
      - means (positions): (N, 3)
      - scales: (N, 3) in log-space
      - quats (rotations): (N, 4) quaternions
      - opacities: (N,) in logit-space
      - sh_coeffs: (N, K, 3) spherical harmonics for view-dependent color
    """

    def __init__(self, positions: np.ndarray, colors: np.ndarray, sh_degree: int = 3):
        super().__init__()
        N = len(positions)
        num_sh = (sh_degree + 1) ** 2

        # Positions
        self.means = nn.Parameter(torch.from_numpy(positions))

        # Scales — initialize from nearest-neighbor distances
        dists = self._compute_nn_distances(positions)
        log_scales = np.log(np.clip(dists, 1e-7, None))
        self.scales = nn.Parameter(
            torch.from_numpy(
                np.tile(log_scales[:, None], (1, 3)).astype(np.float32)
            )
        )

        # Rotations (quaternions, identity init)
        quats = np.zeros((N, 4), dtype=np.float32)
        quats[:, 0] = 1.0  # w=1, x=y=z=0
        self.quats = nn.Parameter(torch.from_numpy(quats))

        # Opacity (logit-space, init ~0.1 opacity → logit ≈ -2.2)
        self.opacities = nn.Parameter(torch.full((N,), -2.2))

        # Spherical harmonics: DC term from colors, rest zero
        sh_coeffs = torch.zeros(N, num_sh, 3)
        # rendered_rgb = 0.5 + C0 * sh_dc  →  sh_dc = (rgb - 0.5) / C0
        C0 = 0.28209479177387814  # 1 / (2*sqrt(pi))
        sh_coeffs[:, 0, :] = (torch.from_numpy(colors) - 0.5) / C0
        self.sh_coeffs = nn.Parameter(sh_coeffs)

        self.sh_degree = sh_degree
        self.active_sh_degree = 0  # Gradually increase during training

    @staticmethod
    def _compute_nn_distances(points: np.ndarray) -> np.ndarray:
        """Compute distance to nearest neighbor for each point."""
        from scipy.spatial import KDTree
        tree = KDTree(points)
        dists, _ = tree.query(points, k=2)  # k=2 because closest is self
        return dists[:, 1].astype(np.float32)

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _position_lr_at_step(step: int, lr_init: float, max_steps: int) -> float:
    """
    Exponential position learning rate decay.
    Matches the original 3DGS paper: lr_init → lr_init/100 over max_steps.
    Prevents late-training drift where Gaussians keep repositioning uselessly.
    """
    t = min(step / max(max_steps, 1), 1.0)
    lr_final = lr_init * 0.01
    return lr_init * (lr_final / lr_init) ** t


def render_view(
    model: GaussianModel,
    camera: dict,
    device: torch.device,
    bg_color: torch.Tensor,
) -> tuple:
    """
    Render a single view using gsplat rasterization.

    Returns:
        (rendered_image, info)
          rendered_image : (H, W, 3) float tensor
          info           : dict from gsplat — contains means2d (C, N, 2) for 2D grad tracking
    """
    from gsplat import rasterization

    H, W = camera["height"], camera["width"]
    fx, fy = camera["fx"], camera["fy"]
    cx, cy = camera["cx"], camera["cy"]

    # Camera intrinsics matrix (3x3)
    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ], dtype=torch.float32, device=device)

    # World-to-camera: viewmat is 4x4
    R = camera["R"].to(device)  # (3, 3)
    T = camera["T"].to(device)  # (3,)
    viewmat = torch.eye(4, device=device)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = T

    # Gaussian parameters
    means = model.means                                         # (N, 3)
    scales = torch.exp(model.scales)                            # (N, 3)
    quats = torch.nn.functional.normalize(model.quats, dim=-1) # (N, 4)
    opacities = torch.sigmoid(model.opacities)                  # (N,)
    sh_coeffs = model.sh_coeffs                                 # (N, K, 3)

    # packed=False → info["means2d"] has shape (C, N, 2), enabling proper 2D grad tracking
    renders, alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=sh_coeffs,
        viewmats=viewmat[None],             # (1, 4, 4)
        Ks=K[None],                         # (1, 3, 3)
        width=W,
        height=H,
        sh_degree=model.active_sh_degree,
        packed=False,
        backgrounds=bg_color[None],         # (1, 3)
        render_mode="RGB",
    )

    return renders[0], info  # (H, W, 3), dict


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.abs(pred - target).mean()


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Simplified SSIM loss (1 - SSIM) on image tensors (H, W, C)."""
    # Convert to (1, C, H, W) for conv2d
    pred = pred.permute(2, 0, 1).unsqueeze(0)
    target = target.permute(2, 0, 1).unsqueeze(0)
    C = pred.shape[1]

    # Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) * g.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)

    pad = window_size // 2
    mu1 = torch.nn.functional.conv2d(pred, window, padding=pad, groups=C)
    mu2 = torch.nn.functional.conv2d(target, window, padding=pad, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(pred ** 2, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(target ** 2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(pred * target, window, padding=pad, groups=C) - mu12

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return 1.0 - ssim_map.mean()


def densify_and_prune(
    model: GaussianModel,
    grad_accum: torch.Tensor,
    grad_count: torch.Tensor,
    config,
    optimizers: dict,
):
    """
    Adaptive density control:
      - Split Gaussians with large positional gradients and large scale
      - Clone Gaussians with large positional gradients and small scale
      - Prune Gaussians with very low opacity
    """
    avg_grad = grad_accum / grad_count.clamp(min=1)
    grad_mask = avg_grad > config.densify_grad_threshold

    scales_exp = torch.exp(model.scales.data)
    scale_max = scales_exp.max(dim=1).values
    scene_extent = (model.means.data.max(dim=0).values - model.means.data.min(dim=0).values).norm()
    # Use percentile-based threshold: Gaussians bigger than median are "large"
    scale_threshold = scene_extent.item() * 0.01

    # Clone: high gradient, small scale
    clone_mask = grad_mask & (scale_max < scale_threshold)
    # Split: high gradient, large scale
    split_mask = grad_mask & (scale_max >= scale_threshold)

    new_means = []
    new_scales = []
    new_quats = []
    new_opacities = []
    new_sh = []

    # Clone — duplicate
    if clone_mask.any():
        new_means.append(model.means.data[clone_mask])
        new_scales.append(model.scales.data[clone_mask])
        new_quats.append(model.quats.data[clone_mask])
        new_opacities.append(model.opacities.data[clone_mask])
        new_sh.append(model.sh_coeffs.data[clone_mask])

    # Split — create two smaller copies
    if split_mask.any():
        n_split = split_mask.sum().item()
        for _ in range(2):
            stds = scales_exp[split_mask]
            samples = torch.randn_like(stds) * stds
            new_means.append(model.means.data[split_mask] + samples)
            new_scales.append(model.scales.data[split_mask] - math.log(1.6))
            new_quats.append(model.quats.data[split_mask])
            new_opacities.append(model.opacities.data[split_mask])
            new_sh.append(model.sh_coeffs.data[split_mask])

    # Prune: low opacity
    opacities_sigmoid = torch.sigmoid(model.opacities.data)
    prune_mask = opacities_sigmoid < config.prune_opacity_threshold

    # Prune: elongated floaters (max_scale / min_scale > threshold)
    scale_min = scales_exp.min(dim=1).values.clamp(min=1e-8)
    elongation = scale_max / scale_min
    prune_mask = prune_mask | (elongation > 50.0)  # Remove needle-like Gaussians

    # Prune: world-space too large (absolute scale cap)
    prune_mask = prune_mask | (scale_max > scene_extent.item() * 0.5)

    # Also prune the split originals
    if split_mask.any():
        prune_mask = prune_mask | split_mask

    keep_mask = ~prune_mask

    # Apply
    with torch.no_grad():
        kept_means = model.means.data[keep_mask]
        kept_scales = model.scales.data[keep_mask]
        kept_quats = model.quats.data[keep_mask]
        kept_opacities = model.opacities.data[keep_mask]
        kept_sh = model.sh_coeffs.data[keep_mask]

        if new_means:
            all_means = torch.cat([kept_means] + new_means, dim=0)
            all_scales = torch.cat([kept_scales] + new_scales, dim=0)
            all_quats = torch.cat([kept_quats] + new_quats, dim=0)
            all_opacities = torch.cat([kept_opacities] + new_opacities, dim=0)
            all_sh = torch.cat([kept_sh] + new_sh, dim=0)
        else:
            all_means = kept_means
            all_scales = kept_scales
            all_quats = kept_quats
            all_opacities = kept_opacities
            all_sh = kept_sh

        N_new = all_means.shape[0]
        model.means = nn.Parameter(all_means)
        model.scales = nn.Parameter(all_scales)
        model.quats = nn.Parameter(all_quats)
        model.opacities = nn.Parameter(all_opacities)
        model.sh_coeffs = nn.Parameter(all_sh)

    # Rebuild optimizers with new parameters
    _rebuild_optimizers(model, optimizers, config)

    n_before = keep_mask.shape[0]
    n_cloned = clone_mask.sum().item()
    n_split_pts = split_mask.sum().item() * 2
    n_pruned = prune_mask.sum().item()
    logger.info(
        f"Densify: {n_before} -> {N_new} "
        f"(+{n_cloned} cloned, +{n_split_pts} split, -{n_pruned} pruned)"
    )

    return N_new


def _rebuild_optimizers(model: GaussianModel, optimizers: dict, config):
    """Rebuild Adam optimizers after densification changes parameter sizes."""
    optimizers["means"] = torch.optim.Adam([model.means], lr=config.learning_rate_position)
    optimizers["scales"] = torch.optim.Adam([model.scales], lr=config.learning_rate_scaling)
    optimizers["quats"] = torch.optim.Adam([model.quats], lr=config.learning_rate_rotation)
    optimizers["opacities"] = torch.optim.Adam([model.opacities], lr=config.learning_rate_opacity)
    optimizers["sh_coeffs"] = torch.optim.Adam([model.sh_coeffs], lr=config.learning_rate_color)


def save_gaussians_ply(model: GaussianModel, output_path: Path):
    """Save trained Gaussians as a PLY file."""
    N = model.num_gaussians
    means = model.means.detach().cpu().numpy()
    scales = torch.exp(model.scales).detach().cpu().numpy()
    quats = torch.nn.functional.normalize(model.quats, dim=-1).detach().cpu().numpy()
    opacities = torch.sigmoid(model.opacities).detach().cpu().numpy()
    sh = model.sh_coeffs.detach().cpu().numpy()  # (N, K, 3)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        # Build header
        num_sh = sh.shape[1]
        header = "ply\nformat binary_little_endian 1.0\n"
        header += f"element vertex {N}\n"
        header += "property float x\nproperty float y\nproperty float z\n"
        header += "property float nx\nproperty float ny\nproperty float nz\n"
        # SH coefficients
        for i in range(num_sh):
            for c in range(3):
                header += f"property float f_sh_{i}_{c}\n"
        header += "property float opacity\n"
        header += "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
        header += "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
        header += "end_header\n"
        f.write(header.encode("utf-8"))

        for i in range(N):
            # Position
            f.write(struct.pack("<3f", *means[i]))
            # Normals (zero)
            f.write(struct.pack("<3f", 0, 0, 0))
            # SH coefficients
            for j in range(num_sh):
                f.write(struct.pack("<3f", *sh[i, j]))
            # Opacity
            f.write(struct.pack("<f", opacities[i]))
            # Scale (log-space for compatibility)
            f.write(struct.pack("<3f", *np.log(scales[i])))
            # Rotation quaternion
            f.write(struct.pack("<4f", *quats[i]))

    logger.info(f"Saved {N} Gaussians to {output_path}")


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(config: PipelineConfig) -> Path:
    """
    Train 3D Gaussian Splatting model using gsplat.

    Args:
        config: Pipeline configuration with GS training params.

    Returns:
        Path to the trained model output directory.
    """
    try:
        import gsplat
    except ImportError:
        raise ImportError(
            "gsplat is not installed. Install it with:\n"
            "  pip install gsplat\n"
            "Requires PyTorch with CUDA support."
        )

    gs = config.gs
    gs_dir = config.gs_output_dir
    model_dir = gs_dir / "trained_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("CUDA not available. Training on CPU will be extremely slow.")

    # Load data
    cameras = load_cameras_and_images(gs_dir)
    if len(cameras) == 0:
        raise RuntimeError("No camera/image data found. Run COLMAP + conversion first.")

    positions, colors = load_initial_points(gs_dir)

    # Initialize model
    model = GaussianModel(positions, colors, sh_degree=gs.sh_degree)
    model.to(device)

    # Move camera images to device
    for cam in cameras:
        cam["image"] = cam["image"].to(device)
        cam["R"] = cam["R"].to(device)
        cam["T"] = cam["T"].to(device)

    bg_color = torch.ones(3, device=device) if gs.white_background else torch.zeros(3, device=device)

    # Setup optimizers
    optimizers = {}
    _rebuild_optimizers(model, optimizers, gs)

    # Gradient accumulation for densification
    grad_accum = torch.zeros(model.num_gaussians, device=device)
    grad_count = torch.zeros(model.num_gaussians, device=device)

    logger.info(f"Starting training: {gs.iterations} iterations, {model.num_gaussians} initial Gaussians")
    logger.info(f"Device: {device}, SH degree: {gs.sh_degree}")

    num_cameras = len(cameras)

    try:
        for iteration in tqdm(range(1, gs.iterations + 1), desc="Training"):
            # Random camera selection
            cam = cameras[random.randrange(num_cameras)]

            # Gradually increase SH degree — scale milestones to total iterations
            sh_interval = max(1, gs.iterations // (gs.sh_degree + 1))
            model.active_sh_degree = min(gs.sh_degree, iteration // sh_interval)

            # Forward: render
            rendered, render_info = render_view(model, cam, device, bg_color)

            # Register 2D means for gradient tracking BEFORE backward
            # info["means2d"] shape: (C=1, N, 2) — 2D screen-space positions
            render_info["means2d"].retain_grad()

            gt_image = cam["image"]

            # Loss: 0.8 * L1 + 0.2 * SSIM
            loss = 0.8 * l1_loss(rendered, gt_image) + 0.2 * ssim_loss(rendered, gt_image)

            # Backward
            loss.backward()

            # Accumulate 2D screen-space gradients for densification (matches paper)
            if gs.densify_from_iter <= iteration <= gs.densify_until_iter:
                grads_2d = render_info["means2d"].grad
                if grads_2d is not None:
                    N = model.means.shape[0]
                    # grad shape: (C=1, N, 2) → (N,) magnitude
                    g = grads_2d[0] if grads_2d.dim() == 3 else grads_2d
                    grads_norm = g.norm(dim=-1)
                    with torch.no_grad():
                        grads_flat = grads_norm.reshape(-1)  # ensure 1D (N,)
                        if grad_accum.shape[0] == N:
                            # Only count iterations where each Gaussian was actually
                            # visible (radii > 0). Dividing by total iterations instead
                            # of visible iterations makes the effective threshold
                            # artificially hard to reach for partially-visible Gaussians.
                            radii = render_info.get("radii", None)
                            if radii is not None:
                                # Flatten to 1D (N,) regardless of gsplat output shape
                                r = radii[0] if radii.dim() >= 2 else radii
                                r = r.reshape(-1)
                                visible = (r > 0)
                                if visible.shape[0] == N:
                                    grad_accum[visible] += grads_flat[visible]
                                    grad_count[visible] += 1
                                else:
                                    grad_accum += grads_flat
                                    grad_count += 1
                            else:
                                grad_accum += grads_flat
                                grad_count += 1
                        else:
                            logger.warning(
                                f"Grad shape mismatch: accum={grad_accum.shape[0]} vs N={N} "
                                f"at iter {iteration} — skipping this step"
                            )

            # Decay position LR: lr_init → lr_init/100 (exponential, matches paper)
            pos_lr = _position_lr_at_step(iteration, gs.learning_rate_position, gs.iterations)
            optimizers["means"].param_groups[0]["lr"] = pos_lr

            # Optimizer step
            for opt in optimizers.values():
                opt.step()
                opt.zero_grad()

            # Densification (skip if already at max)
            if gs.densify_from_iter <= iteration <= gs.densify_until_iter:
                if iteration % gs.densification_interval == 0 and model.num_gaussians < gs.max_gaussians:
                    # Diagnostic: log actual gradient stats at first 3 densification steps
                    if iteration <= gs.densify_from_iter + 300:
                        avg_g = (grad_accum / grad_count.clamp(min=1))
                        logger.info(
                            f"  Grad stats — max: {avg_g.max().item():.2e}  "
                            f"p99: {avg_g.quantile(0.99).item():.2e}  "
                            f"threshold: {gs.densify_grad_threshold:.2e}"
                        )
                    N_new = densify_and_prune(model, grad_accum, grad_count, gs, optimizers)
                    grad_accum = torch.zeros(N_new, device=device)
                    grad_count = torch.zeros(N_new, device=device)
                    if N_new >= gs.max_gaussians:
                        logger.info(f"Reached max Gaussians cap ({gs.max_gaussians}), stopping densification")

            # Opacity reset — hard reset to ~0.01 so Gaussians must re-earn their opacity
            # (matches paper; aggressive pruning of floaters on next densification pass)
            if iteration % gs.opacity_reset_interval == 0 and iteration < gs.densify_until_iter:
                with torch.no_grad():
                    model.opacities.data.fill_(-4.595)  # sigmoid(-4.595) ≈ 0.01

            if iteration % 500 == 0:
                logger.info(f"  Iter {iteration}/{gs.iterations} | Loss: {loss.item():.5f} | "
                            f"Gaussians: {model.num_gaussians} | PosLR: {pos_lr:.2e}")

            # Save checkpoints
            if iteration in gs.save_iterations:
                ply_path = model_dir / f"point_cloud_iter_{iteration}.ply"
                save_gaussians_ply(model, ply_path)

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        logger.error(f"Training crashed at iteration {iteration}: {e}")
        logger.info("Saving emergency checkpoint...")
        emergency_ply = model_dir / f"point_cloud_emergency_iter_{iteration}.ply"
        save_gaussians_ply(model, emergency_ply)
        logger.info(f"Emergency checkpoint saved: {emergency_ply}")
        logger.info(f"TIP: Try again with --quality medium or fewer images if OOM persists")
        return model_dir

    # Final save
    final_ply = model_dir / "point_cloud_final.ply"
    save_gaussians_ply(model, final_ply)

    logger.info(f"Training complete. Final model: {final_ply}")
    logger.info(f"  Final Gaussians: {model.num_gaussians}")
    return model_dir


