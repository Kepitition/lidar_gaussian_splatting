"""
Interactive 3D Gaussian Splatting viewer (web-based).

Uses viser — opens in your browser, supports orbit and free-flight navigation.
No OpenGL/GLFW drivers needed.

Usage:
  python scripts/gs_viewer.py --model output/gs_output/trained_model/point_cloud_final.ply
  python scripts/gs_viewer.py --model output/gs_output/trained_model/point_cloud_final.ply --cameras output/gs_output/cameras.json
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PLY loader
# ---------------------------------------------------------------------------

def load_ply(ply_path: Path):
    """
    Load a 3DGS PLY file saved by gs_train.py.

    Returns:
        means       (N, 3)    float32 — Gaussian centers
        colors      (N, 3)    float32 — RGB in [0, 1] from SH DC term
        opacities   (N,)      float32 — alpha in [0, 1]
        covariances (N, 3, 3) float32
    """
    with open(ply_path, "rb") as f:
        properties = []
        num_vertices = 0
        while True:
            line = f.readline().decode("utf-8").strip()
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
            elif line.startswith("property float"):
                properties.append(line.split()[-1])
            if line == "end_header":
                break

        data = np.frombuffer(f.read(num_vertices * 4 * len(properties)), dtype=np.float32)
        data = data.reshape(num_vertices, len(properties))

    idx = {p: i for i, p in enumerate(properties)}

    means = data[:, [idx["x"], idx["y"], idx["z"]]].copy()

    sh_dc = data[:, [idx["f_sh_0_0"], idx["f_sh_0_1"], idx["f_sh_0_2"]]]
    SH_C0 = 0.28209479177387814
    colors = np.clip(0.5 + sh_dc * SH_C0, 0.0, 1.0).astype(np.float32)

    opacities = data[:, idx["opacity"]].copy().reshape(-1, 1)  # viser expects (N, 1)

    scales = np.exp(data[:, [idx["scale_0"], idx["scale_1"], idx["scale_2"]]])
    quats = data[:, [idx["rot_0"], idx["rot_1"], idx["rot_2"], idx["rot_3"]]]

    covariances = _build_covariances(quats, scales)

    return means, colors, opacities, covariances


def _build_covariances(quats: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """(N,4) quats [w,x,y,z] + (N,3) scales → (N,3,3) covariance matrices."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    R = np.zeros((len(quats), 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    S2 = scales**2
    return np.einsum("nij,nj,nkj->nik", R, S2, R)


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _mat_to_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → [w, x, y, z] quaternion."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25 / s, (R[2, 1] - R[1, 2]) * s,
                         (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s], dtype=np.float32)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s], dtype=np.float32)
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s], dtype=np.float32)
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                         (R[1, 2] + R[2, 1]) / s, 0.25 * s], dtype=np.float32)


def _get_fy(cam: dict) -> float:
    model = cam.get("model", "SIMPLE_PINHOLE")
    params = cam["params"]
    if model in ("PINHOLE", "RADIAL", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
        return float(params[1])
    return float(params[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="3DGS web viewer (viser)")

    # Convenient single-flag mode: --scan output/room
    parser.add_argument("--scan", default=None,
                        help="Scan folder (iphone_import output). Auto-finds model and cameras.")

    # Manual mode
    parser.add_argument("--model", default=None, help="Path to trained PLY file")
    parser.add_argument("--cameras", default=None,
                        help="Path to cameras.json (optional — shows training camera frustums)")

    parser.add_argument("--port", type=int, default=8080, help="Web server port (default: 8080)")
    parser.add_argument("--max-splats", type=int, default=None,
                        help="Cap number of Gaussians rendered (useful for quick preview on large scenes)")
    args = parser.parse_args()

    # Resolve --scan shortcut
    if args.scan:
        scan_dir = Path(args.scan)
        args.model   = str(scan_dir / "trained_model" / "point_cloud_final.ply")
        args.cameras = str(scan_dir / "cameras.json")
    elif not args.model:
        parser.error("Provide either --scan or --model")

    try:
        import viser
    except ImportError:
        raise SystemExit("viser not installed. Run: pip install viser")

    ply_path = Path(args.model)
    if not ply_path.exists():
        raise SystemExit(f"Model file not found: {ply_path}")

    logger.info(f"Loading: {ply_path}")
    means, colors, opacities, covs = load_ply(ply_path)
    logger.info(f"Loaded {len(means):,} Gaussians")

    if args.max_splats and args.max_splats < len(means):
        rng = np.random.default_rng(0)
        keep = rng.choice(len(means), args.max_splats, replace=False)
        means, colors, opacities, covs = means[keep], colors[keep], opacities[keep], covs[keep]
        logger.info(f"Showing {len(means):,} Gaussians (capped by --max-splats)")

    server = viser.ViserServer(port=args.port)
    server.scene.world_axes.visible = False

    # ARKit world has Y pointing up — set orbit axis accordingly
    server.scene.set_up_direction("+y")

    server.scene.add_gaussian_splats(
        "/scene",
        centers=means,
        covariances=covs,
        rgbs=colors,
        opacities=opacities,
    )

    # --- Load camera data (used for frustums + teleport GUI) ---
    cam_data = {}
    if args.cameras:
        cam_path = Path(args.cameras)
        if cam_path.exists():
            with open(cam_path) as f:
                cam_data = json.load(f)
        else:
            logger.warning(f"cameras.json not found: {cam_path}")

    # Compute scene centre from Gaussian means (used for initial view)
    scene_center = means.mean(axis=0).astype(np.float64)

    # Compute a sensible initial view distance: ~3× the point cloud radius
    scene_radius = float(np.percentile(np.linalg.norm(means - scene_center, axis=1), 90))
    view_distance = max(scene_radius * 3.0, 0.3)

    # If we have training cameras use their centroid as the starting look-from position,
    # otherwise fall back to a fixed offset in front of the scene.
    if cam_data:
        cam_positions = np.array([c["position"] for c in cam_data.values()], dtype=np.float64)
        cam_centroid = cam_positions.mean(axis=0)
        # Direction from scene centre toward camera centroid — keep user on the same side
        look_from_dir = cam_centroid - scene_center
        norm = np.linalg.norm(look_from_dir)
        if norm > 1e-6:
            look_from_dir /= norm
        else:
            look_from_dir = np.array([0.0, 0.0, 1.0])
        initial_position = (scene_center + look_from_dir * view_distance).astype(np.float32)
    else:
        initial_position = (scene_center + np.array([0.0, 0.0, view_distance])).astype(np.float32)

    # Point camera at scene centre, Y-up
    def _look_at_wxyz(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0., 1., 0.])) -> np.ndarray:
        """Return [w,x,y,z] quaternion for a camera at `eye` looking at `target`."""
        fwd = target - eye
        fwd_norm = np.linalg.norm(fwd)
        if fwd_norm < 1e-8:
            return np.array([1., 0., 0., 0.], dtype=np.float32)
        fwd = fwd / fwd_norm
        right = np.cross(fwd, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-8:
            up = np.array([0., 0., 1.])
            right = np.cross(fwd, up)
            right_norm = np.linalg.norm(right)
        right /= right_norm
        true_up = np.cross(right, fwd)
        # Build rotation matrix: columns are right, true_up, -fwd (OpenGL convention)
        R = np.stack([right, true_up, -fwd], axis=1)
        return _mat_to_wxyz(R).astype(np.float32)

    initial_wxyz = _look_at_wxyz(initial_position.astype(np.float64), scene_center)

    @server.on_client_connect
    def _on_connect(client):
        """Set initial view for every new browser tab."""
        client.camera.position = initial_position
        client.camera.wxyz = initial_wxyz
        # Orbit centre keeps the scene in view when the user drags
        client.camera.look_at = scene_center.astype(np.float32)

    # --- Training camera frustums + teleport GUI ---
    if cam_data:
        cam_names = list(cam_data.keys())

        for i, (name, cam) in enumerate(cam_data.items()):
            R_wc = np.array(cam["rotation"], dtype=np.float32)
            T_wc = np.array(cam["translation"], dtype=np.float32)
            R_cw = R_wc.T
            t_cw = -R_wc.T @ T_wc
            fy = _get_fy(cam)
            fov_y = 2.0 * np.arctan(cam["height"] / (2.0 * fy))
            server.scene.add_camera_frustum(
                f"/cameras/{i}",
                fov=fov_y,
                aspect=cam["width"] / cam["height"],
                scale=0.05,
                wxyz=_mat_to_wxyz(R_cw),
                position=t_cw,
            )
        logger.info(f"Added {len(cam_data)} camera frustums")

        # GUI panel — teleport to any training camera
        with server.gui.add_folder("Training cameras"):
            cam_slider = server.gui.add_slider(
                "Camera #",
                min=0,
                max=len(cam_names) - 1,
                step=1,
                initial_value=0,
            )
            cam_label = server.gui.add_text("Name", initial_value=cam_names[0], disabled=True)
            teleport_btn = server.gui.add_button("Jump to this camera")

        def _teleport_to(cam_idx: int):
            name = cam_names[int(cam_idx)]
            cam = cam_data[name]
            # cam["rotation"] is R_wc (world-to-cam), cam["position"] is camera centre C
            R_wc = np.array(cam["rotation"], dtype=np.float64)
            C = np.array(cam["position"], dtype=np.float64)
            R_cw = R_wc.T  # cam-to-world rotation

            # Build [w,x,y,z] from R_cw using scipy for numerical robustness
            try:
                from scipy.spatial.transform import Rotation as ScipyR
                xyzw = ScipyR.from_matrix(R_cw).as_quat()
                wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)
            except ImportError:
                wxyz = _mat_to_wxyz(R_cw).astype(np.float32)

            for client in server.get_clients().values():
                client.camera.wxyz = wxyz
                client.camera.position = C.astype(np.float32)
                client.camera.look_at = scene_center.astype(np.float32)

        @cam_slider.on_update
        def _on_slider(_):
            cam_label.value = cam_names[int(cam_slider.value)]

        @teleport_btn.on_click
        def _on_teleport(_):
            _teleport_to(cam_slider.value)

    print(f"\nViewer ready → open http://localhost:{args.port} in your browser")
    print("Controls: left-drag = orbit | right-drag = pan | scroll = zoom")
    print("          W/A/S/D = fly | Q = up | E = down  (click the scene first)")
    if cam_data:
        print("          Use the 'Training cameras' panel to jump to any training viewpoint")
    print("Press Ctrl+C to quit.\n")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
