# Bad Apple!! in yt-project (AMR Visualizer)

> *"If it exists, Bad Apple!! can be rendered on it."*

Bad Apple!! rendered as an astrophysical simulation using the **[yt-project](https://yt-project.org/)** volumetric data analysis and visualization engine.

---

## Overview

This project converts the iconic *Bad Apple!!* music video into a scientific Adaptive Mesh Refinement (AMR) simulation dataset. Each frame is treated as a 2D astrophysical density slice, dynamically discretized into nested multi-level grid patches that adaptively cluster around character contours and silhouette shockfronts.

### Key Features

- **Multi-Level Adaptive Mesh Refinement (AMR):**
  - Up to 5 nested refinement levels constructed per frame (`refine_by=2`).
  - Coarse root grid ($16 \times 12$ cells across a $40 \times 30\ \text{kpc}$ domain) with dynamic subgrid refinement triggered by spatial Sobel gradient operators and silhouette boundaries.
  - Strict compliance with AMR grid cell-alignment constraints across parent-child hierarchies.

- **Astrophysical Density Field & Inverted Magma Aesthetics (`magma_r`):**
  - Preserves the classic Bad Apple aesthetic with an ambient light/white background ($\rho \sim 1\ \text{g/cm}^3$) and dense black silhouettes ($\rho \sim 10^3\ \text{g/cm}^3$).
  - Smooth flame transitions along the boundaries mapping through coral, crimson, magenta, and deep violet.

- **Scientific Annotations:**
  - Physical coordinate axes ($x, y$ in $\text{kpc}$).
  - Logarithmic gas density colorbar ($\text{g/cm}^3$).
  - Real-time simulation timestamps ($t = \dots\ \text{s}$) tracking video progression.

- **High-Performance Parallel Architecture:**
  - Multi-process worker pool chunking (utilizing all available CPU cores).
  - Memory-direct streaming into FFmpeg H.264 pipes (no intermediate uncompressed frame bloat on disk).
  - Seamless chunk concatenation and audio track synchronization via FFmpeg.

---

## Requirements

- Python 3.10+
- [yt](https://yt-project.org/) (`pip install yt`)
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- Matplotlib (`pip install matplotlib`)
- [FFmpeg](https://ffmpeg.org/) (must be available in `PATH`)

Install dependencies:
```bash
pip install yt opencv-python numpy matplotlib
```

---

## Usage

Place the source `bad_apple.mp4` video in the project root directory.

### Quick Preview
Render a short 60-frame preview:
```bash
python render.py --preview 60 --workers 4 --output preview.mp4
```

Render a specific frame range (e.g., frames 900 to 1200):
```bash
python render.py --start 900 --end 1200 --workers 8 --output scene.mp4
```

### Full Video Render
Render the entire video (6,572 frames) with audio synchronization across 20 parallel workers:
```bash
python render.py --workers 20 --output bad_apple_yt_amr.mp4
```

### CLI Options

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--input` | `bad_apple.mp4` | Path to input Bad Apple video file |
| `--output` | `bad_apple_yt_amr.mp4` | Path for the final rendered video |
| `--workers` | `20` | Number of parallel worker processes |
| `--start` | `0` | Starting frame index |
| `--end` | `None` (all) | Ending frame index |
| `--preview` | `None` | Number of frames to render for a test preview |
| `--cmap` | `magma_r` | Colormap for density field |
| `--max-lvl` | `5` | Maximum AMR refinement level |

---

## Technical Architecture

```
                       [ bad_apple.mp4 ]
                               │
               ┌───────────────┴───────────────┐
         (Worker 0)                      (Worker N)
      Frames [0..K]                   Frames [M..End]
             │                               │
    Sobel Edge Tensors              Sobel Edge Tensors
             │                               │
    Dynamic AMR Hierarchy           Dynamic AMR Hierarchy
      (Levels 0..5)                   (Levels 0..5)
             │                               │
       yt.SlicePlot                    yt.SlicePlot
      (magma_r, log ρ)                (magma_r, log ρ)
             │                               │
   Raw Video Pipe (RAM)            Raw Video Pipe (RAM)
             │                               │
      FFmpeg (chunk_0)                FFmpeg (chunk_N)
             └───────────────┬───────────────┘
                             │
                  FFmpeg Concat Demuxer
                  + Original AAC Audio Mux
                             │
                 [ bad_apple_yt_amr.mp4 ]
```

---

## License

MIT License.
Original *Bad Apple!!* music and video rights belong to Team Shanghai Alice / Alstroemeria Records.
