"""
iPhone → Gaussian Splatting 3D Scanning Pipeline.
Author: Ayberk Tunca

Scan rooms and objects with an iPhone camera (+ optional LiDAR depth),
then reconstruct and view them in 3D via Gaussian Splatting.

Usage:
  python main.py --images data/images --quality high
  python main.py --images data/images --quality ultra --colmap-binary "C:/COLMAP/COLMAP.bat"

Pipeline steps:
  1. COLMAP feature extraction + matching + sparse reconstruction
  2. Convert COLMAP output → Gaussian Splatting format
  3. (Optional) Train Gaussian Splatting model
"""

import argparse
import logging
import os
import sys

# ── MSVC / CUDA build environment setup ──────────────────────────────
_msvc_v143 = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.44.35207"
if os.path.isdir(_msvc_v143):
    os.environ.setdefault("VCToolsVersion", "14.44.35207")
    _cl_dir = os.path.join(_msvc_v143, "bin", "Hostx64", "x64")
    if _cl_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _cl_dir + ";" + os.environ.get("PATH", "")
os.environ.setdefault("CUDAFLAGS", "-allow-unsupported-compiler")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
# ─────────────────────────────────────────────────────────────────────
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import PipelineConfig, QualityPreset
from scripts.colmap_pipeline import ColmapPipeline
from scripts.colmap_to_gs import convert_colmap_to_gs
from scripts.gs_train import train as gs_train


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    try:
        import coloredlogs
        coloredlogs.install(level=level, fmt=fmt, datefmt=datefmt)
    except ImportError:
        logging.basicConfig(level=level, format=fmt, datefmt=datefmt)


def parse_args():
    parser = argparse.ArgumentParser(
        description="iPhone → Gaussian Splatting 3D Scanning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
── iPhone / Record3D workflow (no COLMAP needed) ──────────────────────────
  Step 1 — import:
    python scripts/iphone_import.py --input R3_exports --output output/room ...

  Step 2 — train:
    python main.py --scan output/room --quality high

  Step 3 — view:
    python scripts/gs_viewer.py --scan output/room

── COLMAP workflow (regular photos, no LiDAR) ─────────────────────────────
    python main.py --images data/images --output output/my_scene --quality high
        """,
    )

    # ── iPhone / Record3D workflow ─────────────────────────────────────
    parser.add_argument(
        "--scan", type=str, default=None,
        help="Scan folder produced by iphone_import.py  "
             "(contains cameras.json + input.ply + images/).  "
             "When given, --skip-colmap is implied and --images is not needed.",
    )

    # ── COLMAP workflow ────────────────────────────────────────────────
    parser.add_argument(
        "--images", type=str, default=None,
        help="(COLMAP workflow) Path to input images directory.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="(COLMAP workflow) Root output folder. "
             "Creates <output>/colmap_output and <output>/gs_output inside it.",
    )

    # ── Quality ────────────────────────────────────────────────────────
    parser.add_argument(
        "--quality", type=str, default="high",
        choices=["low", "medium", "high", "ultra"],
        help="Quality preset (default: high).",
    )

    # ── COLMAP options ─────────────────────────────────────────────────
    parser.add_argument("--colmap-binary",   type=str, default=None)
    parser.add_argument("--matcher",         type=str, default=None,
                        choices=["exhaustive", "sequential", "vocab_tree"])
    parser.add_argument("--camera-model",    type=str, default=None)
    parser.add_argument("--no-single-camera",action="store_true")
    parser.add_argument("--no-gpu",          action="store_true")

    # ── GS options ─────────────────────────────────────────────────────
    parser.add_argument(
        "--skip-colmap", action="store_true",
        help="Skip COLMAP (legacy flag — prefer --scan for iPhone workflow).",
    )
    parser.add_argument("--no-gs-train",     action="store_true")
    parser.add_argument("--gs-iterations",   type=int, default=None)
    parser.add_argument("--white-background",action="store_true")

    # ── Housekeeping ───────────────────────────────────────────────────
    parser.add_argument(
        "--clean", action="store_true",
        help="Delete previous trained_model/ before re-training.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("main")

    # ── Build configuration ──────────────────────────────────────────
    config = PipelineConfig()
    config.quality = QualityPreset(args.quality)
    config.apply_quality_preset()

    # ── Path resolution ───────────────────────────────────────────────
    # iPhone workflow: --scan points directly at the iphone_import output folder.
    #   output/room/
    #     cameras.json   ← iphone_import wrote this
    #     input.ply      ← iphone_import wrote this
    #     images/        ← iphone_import wrote this
    #     trained_model/ ← gs_train will write here
    #
    # COLMAP workflow: --images + --output, creates subfolders automatically.
    #   output/my_scene/
    #     colmap_output/ ← COLMAP writes here
    #     gs_output/     ← gs_train writes here

    if args.scan:
        # iPhone workflow — --scan IS the scan folder, everything lives inside it
        scan_dir = Path(args.scan).resolve()
        config.gs_output_dir = scan_dir
        config.image_dir     = scan_dir / "images"
        args.skip_colmap     = True   # implied
        logger.info(f"iPhone scan folder: {scan_dir}")
    elif args.images:
        # COLMAP workflow
        config.image_dir = Path(args.images).resolve()
        if args.output:
            base = Path(args.output).resolve()
            config.colmap_output_dir = base / "colmap_output"
            config.gs_output_dir     = base / "gs_output"
    else:
        logger.error("Provide either --scan (iPhone) or --images (COLMAP).")
        sys.exit(1)

    if args.colmap_binary:
        config.colmap_binary = args.colmap_binary
    if args.matcher:
        config.colmap.matcher_type = args.matcher
    if args.camera_model:
        config.colmap.camera_model = args.camera_model
    if args.no_single_camera:
        config.colmap.single_camera = False
    if args.no_gpu:
        config.use_gpu = False
    if args.gs_iterations:
        config.gs.iterations = args.gs_iterations
    if args.white_background:
        config.gs.white_background = True

    # ── Clean previous outputs ──────────────────────────────────────
    if args.clean:
        import shutil
        if args.skip_colmap:
            trained_dir = config.gs_output_dir / "trained_model"
            if trained_dir.exists():
                logger.info(f"Cleaning trained model: {trained_dir}")
                shutil.rmtree(trained_dir)
        else:
            for d in [config.colmap_output_dir, config.gs_output_dir]:
                if d.exists():
                    logger.info(f"Cleaning: {d}")
                    shutil.rmtree(d)
        torch_ext_dir = Path.home() / ".cache" / "torch_extensions"
        if torch_ext_dir.exists():
            shutil.rmtree(torch_ext_dir, ignore_errors=True)
        logger.info("Clean complete.")

    # ── Validate ─────────────────────────────────────────────────────
    if not config.image_dir.exists():
        logger.error(f"Image directory does not exist: {config.image_dir}")
        sys.exit(1)

    image_files = list(config.image_dir.glob("*.[jJ][pP][gG]")) + \
                  list(config.image_dir.glob("*.[pP][nN][gG]")) + \
                  list(config.image_dir.glob("*.[tT][iI][fF]")) + \
                  list(config.image_dir.glob("*.[tT][iI][fF][fF]"))
    if len(image_files) < 3:
        logger.error(f"Found only {len(image_files)} images. Need at least 3 for reconstruction.")
        sys.exit(1)

    logger.info(f"Found {len(image_files)} images in {config.image_dir}")
    logger.info(f"Quality: {config.quality.value} | Matcher: {config.colmap.matcher_type}")

    # ── Stage 1: COLMAP SfM ──────────────────────────────────────────
    if not args.skip_colmap:
        logger.info("=" * 60)
        logger.info("STAGE 1: COLMAP Structure-from-Motion")
        logger.info("=" * 60)

        colmap = ColmapPipeline(config)
        colmap.run()

        try:
            stats = colmap.get_reconstruction_stats()
            logger.info(f"Reconstruction stats: {stats}")
        except Exception as e:
            logger.warning(f"Could not read reconstruction stats: {e}")

        logger.info("=" * 60)
        logger.info("STAGE 2: Converting COLMAP → Gaussian Splatting format")
        logger.info("=" * 60)

        gs_dir = convert_colmap_to_gs(config)
    else:
        logger.info("Skipping COLMAP (--skip-colmap). Reusing existing data.")
        gs_dir = config.gs_output_dir
        if not (gs_dir / "cameras.json").exists():
            logger.error(f"No cameras.json in {gs_dir}. Run without --skip-colmap first.")
            sys.exit(1)

    # ── Stage 3: Gaussian Splatting Training ──────────────────────────
    if not args.no_gs_train:
        logger.info("=" * 60)
        logger.info("STAGE 3: Gaussian Splatting Training")
        logger.info("=" * 60)

        try:
            model_path = gs_train(config)
            logger.info(f"Trained model: {model_path}")
        except ImportError as e:
            logger.warning(f"GS training skipped: {e}")
            logger.info("Install gsplat: pip install gsplat")
    else:
        logger.info("Skipping GS training (--no-gs-train).")

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info(f"  GS output: {config.gs_output_dir}")
    logger.info("  View: python scripts/gs_viewer.py "
                f"--model {config.gs_output_dir}/trained_model/point_cloud_final.ply "
                f"--cameras {config.gs_output_dir}/cameras.json")
    logger.info("        → then open http://localhost:8080 in your browser")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
