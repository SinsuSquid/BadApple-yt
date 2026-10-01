#!/usr/bin/env python3
"""
Bad Apple!! rendered using yt-project with Multi-Level AMR & Magma-R (White Background) Aesthetics
Option 2: Continuous Flame Contour (White Background, Black Silhouettes as High Densities, No Explicit Grid Lines)
"""

import os
import sys
import time
import argparse
import subprocess
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import yt

# Suppress yt's verbose logging
yt.set_log_level(50)

# Physical domain definition in kpc
X_MIN, X_MAX = -20.0, 20.0
Y_MIN, Y_MAX = -15.0, 15.0
Z_MIN, Z_MAX = -1.0, 1.0

# Multi-level AMR parameters
W_GRID, H_GRID = 512, 384
DOMAIN_DIMS = np.array([16, 12, 1])
DOMAIN_LEFT = np.array([X_MIN, Y_MIN, Z_MIN])
DOMAIN_RIGHT = np.array([X_MAX, Y_MAX, Z_MAX])
BBOX = np.column_stack([DOMAIN_LEFT, DOMAIN_RIGHT])
MAX_LEVEL = 5


def process_frame(frame_bgr, sim_time_sec, cmap='magma_r', max_lvl=MAX_LEVEL):
    """
    Construct multi-level AMR dataset for a video frame and render via yt SlicePlot.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (W_GRID, H_GRID))

    # Invert so black silhouette in video becomes high intensity (255)
    # and white background in video becomes 0
    sil_intensity = 255.0 - img.astype(np.float64)
    sil_up = np.flipud(sil_intensity)

    # Edge detection for AMR refinement criteria
    grad_x = cv2.Sobel(sil_intensity, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(sil_intensity, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    grad_up = np.flipud(grad)

    grid_data = []

    # Level 0 (root coarse grid: 16x12)
    l0_sil = cv2.resize(sil_up, (16, 12), interpolation=cv2.INTER_AREA).T[:, :, np.newaxis]
    l0_rho = (l0_sil / 255.0) * 2.0 + 1.0
    grid_data.append({
        'left_edge': DOMAIN_LEFT.tolist(),
        'right_edge': DOMAIN_RIGHT.tolist(),
        'dimensions': [16, 12, 1],
        'level': 0,
        'density': (l0_rho, 'g/cm**3')
    })

    # Levels 1 through max_lvl
    for lvl in range(1, max_lvl + 1):
        parent_dims = DOMAIN_DIMS * (2 ** (lvl - 1))
        bx_parent = 4 if parent_dims[0] >= 32 else 2
        by_parent = 4 if parent_dims[1] >= 24 else 2

        x_edges = np.linspace(DOMAIN_LEFT[0], DOMAIN_RIGHT[0], parent_dims[0], endpoint=False)
        y_edges = np.linspace(DOMAIN_LEFT[1], DOMAIN_RIGHT[1], parent_dims[1], endpoint=False)
        dx_p = (DOMAIN_RIGHT[0] - DOMAIN_LEFT[0]) / parent_dims[0]
        dy_p = (DOMAIN_RIGHT[1] - DOMAIN_LEFT[1]) / parent_dims[1]

        thresh = max(15.0, 80.0 / (lvl ** 0.8))

        for px_idx in range(0, parent_dims[0], bx_parent):
            for py_idx in range(0, parent_dims[1], by_parent):
                x0 = int(px_idx / parent_dims[0] * W_GRID)
                x1 = int((px_idx + bx_parent) / parent_dims[0] * W_GRID)
                y0 = int(py_idx / parent_dims[1] * H_GRID)
                y1 = int((py_idx + by_parent) / parent_dims[1] * H_GRID)

                patch_grad = grad_up[y0:y1, x0:x1]
                patch_sil = sil_up[y0:y1, x0:x1]

                if np.max(patch_grad) > thresh or np.mean(patch_sil) > 20.0:
                    lx = x_edges[px_idx]
                    rx = lx + bx_parent * dx_p
                    ly = y_edges[py_idx]
                    ry = ly + by_parent * dy_p

                    sub_nx = bx_parent * 2
                    sub_ny = by_parent * 2

                    patch_res = cv2.resize(patch_sil, (sub_nx, sub_ny)).T[:, :, np.newaxis].astype(np.float64)
                    norm_sil = patch_res / 255.0
                    base_rho = 2.5 ** lvl
                    sub_density = 1.0 + norm_sil * base_rho * 10.0

                    grid_data.append({
                        'left_edge': [lx, ly, -1.0],
                        'right_edge': [rx, ry, 1.0],
                        'dimensions': [sub_nx, sub_ny, 1],
                        'level': lvl,
                        'density': (sub_density, 'g/cm**3')
                    })

    ds = yt.load_amr_grids(
        grid_data,
        DOMAIN_DIMS,
        bbox=BBOX,
        length_unit='kpc',
        refine_by=2,
        periodicity=(False, False, False),
        sim_time=sim_time_sec,
        time_unit='s'
    )

    p = yt.SlicePlot(
        ds,
        'z',
        ('stream', 'density'),
        center=[0.0, 0.0, 0.0],
        width=((X_MAX - X_MIN, 'kpc'), (Y_MAX - Y_MIN, 'kpc'))
    )
    p.set_cmap(('stream', 'density'), cmap)
    p.set_zlim(('stream', 'density'), 1, 1000)
    p.annotate_timestamp(corner='upper_left', time_format='t = {time:.2f} {units}', text_args={'color': 'black'})

    p.render()
    fig = p.plots[('stream', 'density')].figure
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())

    h, w = rgba.shape[:2]
    h_even = h - (h % 2)
    w_even = w - (w % 2)

    bgr = cv2.cvtColor(rgba[:h_even, :w_even], cv2.COLOR_RGBA2BGR)

    plt.close(fig)
    del p, ds

    return bgr


def worker_render_chunk(worker_id, video_path, start_frame, end_frame, fps, temp_out_path, cmap, max_lvl):
    """
    Worker function executed in separate process:
    Reads frames start_frame to end_frame, renders via yt, and pipes frames to an ffmpeg subprocess.
    """
    yt.set_log_level(50)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ffmpeg_proc = None
    frames_count = end_frame - start_frame

    try:
        for idx in range(frames_count):
            curr_frame = start_frame + idx
            ret, frame = cap.read()
            if not ret:
                break

            sim_time = curr_frame / fps
            rendered_bgr = process_frame(
                frame,
                sim_time,
                cmap=cmap,
                max_lvl=max_lvl
            )

            if ffmpeg_proc is None:
                h, w = rendered_bgr.shape[:2]
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-s', f'{w}x{h}',
                    '-pix_fmt', 'bgr24',
                    '-r', str(fps),
                    '-i', '-',
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-crf', '18',
                    '-pix_fmt', 'yuv420p',
                    temp_out_path
                ]
                ffmpeg_proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

            ffmpeg_proc.stdin.write(rendered_bgr.tobytes())

            if (idx + 1) % 50 == 0 or (idx + 1) == frames_count:
                print(f"[Worker {worker_id:02d}] {idx + 1}/{frames_count} frames ({((idx + 1)/frames_count)*100:.1f}%)")
                sys.stdout.flush()

    finally:
        cap.release()
        if ffmpeg_proc is not None:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()

    return worker_id, temp_out_path


def main():
    parser = argparse.ArgumentParser(description="Render Bad Apple!! with yt Multi-Level AMR (Option 2)")
    parser.add_argument("--input", default="bad_apple.mp4", help="Input video path")
    parser.add_argument("--output", default="bad_apple_yt_amr.mp4", help="Output video path")
    parser.add_argument("--workers", type=int, default=20, help="Number of parallel worker processes")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--end", type=int, default=None, help="End frame index (default: all)")
    parser.add_argument("--preview", type=int, default=None, help="Render preview of N frames")
    parser.add_argument("--cmap", default="magma_r", help="Colormap name (default: magma_r)")
    parser.add_argument("--max-lvl", type=int, default=5, help="Maximum AMR refinement level (default: 5)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input video '{args.input}' not found.")
        sys.exit(1)

    cap = cv2.VideoCapture(args.input)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    start_frame = max(0, args.start)
    end_frame = min(total_frames, args.end if args.end is not None else total_frames)

    if args.preview is not None:
        end_frame = min(end_frame, start_frame + args.preview)

    total_to_render = end_frame - start_frame
    print("=" * 60)
    print(" Bad Apple!! -> yt Multi-Level AMR Visualizer (Option 2)")
    print("=" * 60)
    print(f"Input:        {args.input}")
    print(f"Output:       {args.output}")
    print(f"Total Frames: {total_to_render} (frames {start_frame} to {end_frame}) @ {fps:.2f} fps")
    print(f"Colormap:     {args.cmap} (White Background, Flame Contours)")
    print(f"Max Level:    {args.max_lvl}")
    print(f"Workers:      {args.workers}")
    print("=" * 60)

    temp_dir = os.path.abspath("temp_amr_chunks")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    num_workers = min(args.workers, total_to_render)
    frames_per_worker = total_to_render // num_workers
    remainder = total_to_render % num_workers

    chunks = []
    curr = start_frame
    for w_id in range(num_workers):
        count = frames_per_worker + (1 if w_id < remainder else 0)
        c_start = curr
        c_end = curr + count
        curr = c_end
        temp_chunk_file = os.path.join(temp_dir, f"chunk_{w_id:03d}.mp4")
        chunks.append((w_id, args.input, c_start, c_end, fps, temp_chunk_file, args.cmap, args.max_lvl))

    start_time = time.time()

    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_render_chunk, *chunk_args) for chunk_args in chunks]
        for f in concurrent.futures.as_completed(futures):
            w_id, path = f.result()

    render_duration = time.time() - start_time
    fps_achieved = total_to_render / render_duration if render_duration > 0 else 0
    print("-" * 60)
    print(f"Render completed in {render_duration:.1f}s ({fps_achieved:.2f} rendered fps)!")
    print("Concatenating chunks and muxing audio with FFmpeg...")

    concat_list_file = os.path.join(temp_dir, "concat_list.txt")
    with open(concat_list_file, "w") as f:
        for chunk in chunks:
            chunk_path = chunk[5].replace("\\", "/")
            f.write(f"file '{chunk_path}'\n")

    start_sec = start_frame / fps
    duration_sec = total_to_render / fps

    final_cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_list_file,
        '-ss', f'{start_sec:.3f}',
        '-t', f'{duration_sec:.3f}',
        '-i', args.input,
        '-map', '0:v:0',
        '-map', '1:a:0?',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-b:a', '192k',
        args.output
    ]

    res = subprocess.run(final_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print("FFmpeg concat error:", res.stderr.decode('utf-8', errors='ignore'))
    else:
        print(f"Successfully generated final video: {args.output}")

    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass

    print("=" * 60)
    print("All done!")


if __name__ == "__main__":
    main()
