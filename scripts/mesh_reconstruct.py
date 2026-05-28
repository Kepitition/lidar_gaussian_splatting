"""
Mesh reconstruction from LiDAR point cloud.
Author: Ayberk Tunca

Takes the input.ply already produced by iphone_import.py and converts it
into a watertight mesh using Poisson surface reconstruction.

Why Poisson instead of TSDF volume fusion:
  - input.ply is already correct (depth unprojection + pose transform tested)
  - Poisson is simpler, more robust, needs no extrinsic matrix conventions
  - Produces watertight mesh by design — ready for 3D printing

Outputs:
  mesh.ply              — mesh with vertex colours (open in MeshLab, Blender)
  mesh.obj              — same, for Windows 3D Viewer / any 3D app
  mesh_print.stl        — units in mm, drag into Cura / Bambu / PrusaSlicer

Usage:
  # After iphone_import.py ran:
  python scripts/mesh_reconstruct.py --input output/gs_output

  # Finer detail (slower):
  python scripts/mesh_reconstruct.py --input output/gs_output --poisson-depth 10

  # Room scan output:
  python scripts/mesh_reconstruct.py --input output/room_output --poisson-depth 8

Requirements:
  pip install open3d
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def reconstruct_mesh(
    gs_dir: Path,
    output_dir: Path,
    poisson_depth: int = 9,
    low_density_pct: float = 5.0,
    normal_radius: float = 0.015,
    normal_max_nn: int = 30,
):
    """
    Poisson surface reconstruction from input.ply.

    poisson_depth:    Controls mesh resolution. 8 = coarse/fast, 9 = good,
                      10 = fine/slow, 11 = very fine (needs lots of RAM).
    low_density_pct:  Remove faces with density below this percentile — cleans
                      up the 'skirt' Poisson adds around the point cloud edge.
    normal_radius:    Radius (metres) for normal estimation neighbourhood.
                      0.015 = 1.5cm, good for object scale. Use 0.05 for rooms.
    normal_max_nn:    Max neighbours for normal estimation.
    """
    try:
        import open3d as o3d
    except ImportError:
        raise SystemExit("open3d not installed.  Run:  pip install open3d")

    ply_path = gs_dir / "input.ply"
    if not ply_path.exists():
        raise FileNotFoundError(
            f"input.ply not found in {gs_dir}.\n"
            "Run iphone_import.py first to generate it."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load point cloud ───────────────────────────────────────────────────
    logger.info(f"Loading point cloud: {ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    n_pts = len(pcd.points)
    logger.info(f"  {n_pts:,} points loaded")

    if n_pts < 1000:
        raise RuntimeError(f"Too few points ({n_pts}). Check input.ply is valid.")

    # ── Downsample if very large (speeds up normal estimation) ────────────
    if n_pts > 300_000:
        voxel_size = 0.002  # 2mm voxel downsample
        pcd = pcd.voxel_down_sample(voxel_size)
        logger.info(f"  Downsampled to {len(pcd.points):,} points (voxel={voxel_size*1000:.0f}mm)")

    # ── Cluster-based crop: keep only the dominant object ────────────────
    # Point clouds from scene scans contain everything (desk, monitor, etc.).
    # DBSCAN clustering finds groups of nearby points; we keep the largest
    # cluster which is usually the primary subject.
    logger.info("Isolating main object via DBSCAN clustering...")
    labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=False))
    if labels.max() >= 0:
        # labels == -1 means noise; find the largest non-noise cluster
        unique, counts = np.unique(labels[labels >= 0], return_counts=True)
        main_cluster = unique[counts.argmax()]
        indices = np.where(labels == main_cluster)[0]
        pcd = pcd.select_by_index(indices)
        logger.info(f"  Kept cluster {main_cluster}: {len(pcd.points):,} points "
                    f"({len(unique)} clusters found)")
    else:
        logger.info("  No clusters found — using full point cloud")

    # ── Remove statistical outliers (LiDAR noise) ─────────────────────────
    logger.info("Removing outliers...")
    pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    n_removed = len(pcd.points) - len(pcd_clean.points)
    logger.info(f"  Removed {n_removed:,} outlier points → {len(pcd_clean.points):,} remain")
    pcd = pcd_clean

    # ── Estimate normals ───────────────────────────────────────────────────
    logger.info("Estimating surface normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=normal_max_nn
        )
    )
    # Orient normals toward the point cloud centroid — reliable for objects
    # scanned from the outside (camera always facing inward toward the object).
    # Much more stable than tangent plane on noisy LiDAR clouds.
    centroid = np.mean(np.asarray(pcd.points), axis=0)
    pcd.orient_normals_towards_camera_location(centroid)
    logger.info("  Normals estimated and oriented toward scene centroid")

    # ── Poisson surface reconstruction ────────────────────────────────────
    logger.info(f"Running Poisson reconstruction (depth={poisson_depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=poisson_depth,
        width=0,
        scale=1.1,
        linear_fit=False,
    )
    logger.info(f"  Raw mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    # ── Remove low-density faces (boundary artefacts) ──────────────────────
    densities = np.asarray(densities)
    threshold = np.percentile(densities, low_density_pct)
    verts_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(verts_to_remove)
    logger.info(f"  After density filter ({low_density_pct}th pct): "
                f"{len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    # ── Keep largest connected component ──────────────────────────────────
    logger.info("Keeping largest connected component...")
    triangle_clusters, cluster_n_tri, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_tri     = np.asarray(cluster_n_tri)
    largest           = int(cluster_n_tri.argmax())
    # remove_triangles_by_mask takes a per-TRIANGLE boolean — correct API
    remove_mask = triangle_clusters != largest
    mesh.remove_triangles_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    logger.info(f"  Final mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    if len(mesh.triangles) == 0:
        raise RuntimeError(
            "Final mesh has 0 triangles.\n"
            "Try: --poisson-depth 8, or check that input.ply has valid geometry."
        )

    # ── Export ─────────────────────────────────────────────────────────────
    ply_out = output_dir / "mesh.ply"
    obj_out = output_dir / "mesh.obj"
    stl_out = output_dir / "mesh_print.stl"

    o3d.io.write_triangle_mesh(str(ply_out), mesh)
    logger.info(f"Saved: {ply_out}")

    o3d.io.write_triangle_mesh(str(obj_out), mesh)
    logger.info(f"Saved: {obj_out}")

    # STL: scale metres → mm for slicers
    mesh_mm = o3d.geometry.TriangleMesh(mesh)
    verts_mm = np.asarray(mesh_mm.vertices) * 1000.0
    mesh_mm.vertices = o3d.utility.Vector3dVector(verts_mm)
    mesh_mm.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(stl_out), mesh_mm)
    logger.info(f"Saved: {stl_out}  (units: mm — ready for slicer)")

    logger.info("")
    logger.info("════════════════════════════════════════════════════════════")
    logger.info("Mesh reconstruction complete!")
    logger.info(f"  Vertices  : {len(mesh.vertices):,}")
    logger.info(f"  Triangles : {len(mesh.triangles):,}")
    logger.info(f"  View      : {obj_out}")
    logger.info(f"  Print     : {stl_out}  (mm scale)")
    logger.info("════════════════════════════════════════════════════════════")

    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Watertight mesh + STL from LiDAR point cloud",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Keyboard / small object
  python scripts/mesh_reconstruct.py --input output/gs_output

  # Room scan (coarser detail, faster)
  python scripts/mesh_reconstruct.py --input output/room_output --poisson-depth 8 --normal-radius 0.05

  # High detail (slower, more RAM)
  python scripts/mesh_reconstruct.py --input output/gs_output --poisson-depth 10
        """,
    )
    parser.add_argument("--input",         required=True,
                        help="Folder containing input.ply (gs_output or room_output)")
    parser.add_argument("--output",        default=None,
                        help="Output folder (default: <input>/mesh)")
    parser.add_argument("--poisson-depth", type=int,   default=9,
                        help="Poisson octree depth. 8=coarse, 9=good, 10=fine (default: 9)")
    parser.add_argument("--density-pct",   type=float, default=5.0,
                        help="Remove faces below this density percentile (default: 5)")
    parser.add_argument("--normal-radius", type=float, default=0.015,
                        help="Normal estimation radius in metres (default: 0.015 for objects, 0.05 for rooms)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    input_dir  = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir / "mesh"

    reconstruct_mesh(
        gs_dir        = input_dir,
        output_dir    = output_dir,
        poisson_depth = args.poisson_depth,
        low_density_pct = args.density_pct,
        normal_radius = args.normal_radius,
    )


if __name__ == "__main__":
    main()
