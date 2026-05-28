# LiDAR → Gaussian Splatting Pipeline

**Scan rooms and objects with an iPhone, reconstruct them as interactive 3D Gaussian Splats — no COLMAP, no manual calibration.**

Camera poses come directly from ARKit (sub-millimetre accuracy). The initial point cloud comes from LiDAR depth maps. Gaussian Splatting training runs on a local GPU in ~15 minutes.

---

## Results

| Room scan | Object scan |
|-----------|-------------|
| ![Room scan result](assets/room_result.jpg) | ![Object scan result](assets/object_result.jpg) |

> *Add your screenshots here — replace the paths above with your own images*

<!-- VIDEO PLACEHOLDER
To embed a video, upload it to YouTube or GitHub and paste the link here:
[![Demo video](assets/video_thumbnail.jpg)](https://your-video-link)
-->

---

## How it works

```
iPhone (ARKit + LiDAR)
        │
        ▼
 Record3D export
  rgb/  depth/  metadata.json
        │
        ▼
 iphone_import.py
  • Selects frames with Farthest Point Sampling
  • Unprojects LiDAR depth → dense point cloud
  • Converts ARKit poses → OpenCV convention
        │
        ▼
 cameras.json + input.ply + images/
        │
        ▼
 gs_train.py (gsplat)
  • Initialises ~500K Gaussians from LiDAR point cloud
  • Optimises position, colour, opacity, scale, rotation
  • Clone / split / prune every 100 iterations
        │
        ▼
 point_cloud_final.ply
        │
        ▼
 gs_viewer.py (viser)
  • Web viewer — open in any browser
  • Orbit, pan, fly-through navigation
  • Jump to any training camera viewpoint
```

---

## Requirements

### Hardware
- **iPhone 12 Pro or later** (LiDAR required)
- **NVIDIA GPU** with CUDA — tested on RTX 4060 8 GB
- Windows 10/11 or Linux

### Software
- Python 3.10+
- CUDA 11.8+ and matching PyTorch
- [Record3D](https://record3d.app/) — free iOS app for capturing

---

## Installation

```bash
git clone https://github.com/Kepitition/lidar_gaussian_splatting
cd lidar_gaussian_splatting
```

Install PyTorch with CUDA (adjust your CUDA version):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Install remaining dependencies:
```bash
pip install gsplat viser open3d opencv-python imageio scipy coloredlogs
```

> **Windows note:** gsplat compiles CUDA kernels on first run. You need the [MSVC Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) installed (Desktop development with C++ workload).

---

## Full workflow

### Step 1 — Record with iPhone

Open **Record3D**, tap the settings gear and set:
- RGB resolution: **1440 × 1920** (highest quality)
- LiDAR: enabled
- Mode: **Video**

Record your scan:
- **Rooms:** Walk the perimeter slowly, pause at corners, tilt phone up toward ceiling and down toward floor. 1–2 minutes is enough.
- **Objects:** Orbit the object at 3–4 different heights. Keep it under 30 seconds.

Export from Record3D: **Share → EXR + JPG** → transfer to PC (AirDrop, cable, or Files app).

---

### Step 2 — Import

```bash
# Room scan
python scripts/iphone_import.py \
  --input R3_exports/R3_room \
  --output output/room \
  --max-frames 400 \
  --max-depth 4.0

# Small object
python scripts/iphone_import.py \
  --input R3_exports/R3_keyboard \
  --output output/keyboard \
  --max-frames 200 \
  --max-depth 1.0
```

**What this produces:**
```
output/room/
  cameras.json    ← 400 camera poses with intrinsics
  input.ply       ← ~500K 3D points from LiDAR
  images/         ← 400 selected RGB frames
```

**Import options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--max-frames` | 200 | Frames to select (uses Farthest Point Sampling for even coverage) |
| `--max-depth` | 2.0 | Depth cutoff in metres — use 1.0 for objects, 4.0 for rooms |
| `--depth-subsample` | 2 | Depth pixel stride — 1 = maximum points, 4 = faster import |

---

### Step 3 — Train

```bash
python main.py --scan output/room --quality high
```

**Quality presets:**

| Preset | Iterations | Max Gaussians | SH degree | Time (RTX 4060) |
|--------|-----------|---------------|-----------|-----------------|
| `low` | 10,000 | 150K | 1 | ~3 min |
| `medium` | 20,000 | 400K | 2 | ~7 min |
| `high` | 30,000 | 800K | 3 | ~15 min |
| `ultra` | 50,000 | 1.5M | 3 | ~25 min |

Training saves checkpoints at regular intervals so you can preview early results.

**Useful flags:**
```bash
--quality ultra             # best quality, slower
--gs-iterations 50000       # override iteration count
--clean                     # delete previous trained_model/ before retraining
```

---

### Step 4 — View

```bash
python scripts/gs_viewer.py --scan output/room
```

Open **http://localhost:8080** in your browser.

**Controls:**

| Input | Action |
|-------|--------|
| Left drag | Orbit |
| Right drag | Pan |
| Scroll | Zoom |
| W / A / S / D | Fly forward / left / back / right |
| Q / E | Fly up / down |
| Click scene first | Activate keyboard controls |

The **Training cameras** panel on the right lets you jump to any of the 400 training viewpoints instantly.

---

## Output files

```
output/room/
  cameras.json              ← camera poses (kept for viewer)
  input.ply                 ← LiDAR point cloud (initial Gaussians)
  images/                   ← training frames
  trained_model/
    point_cloud_iter_5000.ply    ← early checkpoint
    point_cloud_iter_15000.ply
    point_cloud_iter_30000.ply
    point_cloud_final.ply        ← final model (open this in viewer)
```

---

## Tuning for better results

### Blurry everywhere
More iterations and frames:
```bash
python scripts/iphone_import.py --input R3_exports/R3_room --output output/room --max-frames 400 --max-depth 4.0
python main.py --scan output/room --quality ultra --clean
```

### Missing fine detail
The densification threshold controls how aggressively Gaussians are cloned in high-gradient areas. Edit `config.py`:
```python
densify_grad_threshold: float = 0.000008  # raise for more cloning
```

### Floaty artifacts / foggy haze
Raise the opacity prune threshold in `config.py`:
```python
prune_opacity_threshold: float = 0.01   # was 0.005
```

### Best recording tips
- Scan slowly — motion blur degrades depth quality
- Overlap passes — revisit areas from different angles
- Avoid glass and mirrors (LiDAR reflects poorly)
- Good lighting helps RGB quality (LiDAR works in the dark but textures won't)

---

## Project structure

```
lidar_gaussian_splatting/
  main.py                     ← pipeline entry point
  config.py                   ← all training parameters and presets
  scripts/
    iphone_import.py          ← Record3D → training format
    gs_train.py               ← Gaussian Splatting training loop
    gs_viewer.py              ← web-based 3D viewer
```

---

## License

MIT — see [LICENSE](LICENSE)
