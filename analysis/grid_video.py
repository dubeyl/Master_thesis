import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
cv2.setPreferableBackend(cv2.CAP_FFMPEG)

def get_video_min_max(video_path, max_frames=None):
    cap = cv2.VideoCapture(str(video_path))
    vmin, vmax = np.inf, -np.inf
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vmin = min(vmin, frame.min())
        vmax = max(vmax, frame.max())
        count += 1
        if max_frames and count >= max_frames:
            break
    cap.release()
    return vmin, vmax

def load_normalized_video(video_path, vmin, vmax, target_res, max_frames=None):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    frame_idx = 0
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 2 == 0:  # Keep every second frame (reduce 80 → 40 fps)
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            norm = np.clip((frame.astype(np.float32) - vmin) / (vmax - vmin) * 255, 0, 255)
            norm = cv2.resize(norm.astype(np.uint8), target_res)
            frames.append(norm)
            count += 1
            if max_frames and count >= max_frames:
                break
        frame_idx += 1
    cap.release()
    return frames

def create_grid_video(pupa_id, start_rec, end_rec, output_path, grid_size=(7, 7), target_res=(160, 160), max_frames=None):
    base_dir = Path(f"Z:/data/LD/experiments/imaging/output/Automated_experiment/{pupa_id}")
    recording_numbers = list(range(start_rec, end_rec + 1))

    all_videos = []
    max_len = 0

    print("🔍 Scanning videos for min/max...")

    for rec_num in tqdm(recording_numbers, desc="Analyzing"):
        rec_folder = list(base_dir.glob(f"recording{rec_num}_*/"))
        if not rec_folder:
            print(f"⚠️ Recording {rec_num} not found.")
            all_videos.append(None)
            continue

        video_file = rec_folder[0] / f"{rec_folder[0].name}.mkv"
        vmin, vmax = get_video_min_max(video_file, max_frames=max_frames)
        frames = load_normalized_video(video_file, vmin, vmax, target_res, max_frames=max_frames)
        all_videos.append(frames)
        max_len = max(max_len, len(frames))

    # Pad or fill in missing recordings
    for i in range(len(all_videos)):
        if all_videos[i] is None:
            all_videos[i] = [np.zeros(target_res, dtype=np.uint8)] * max_len
        else:
            last = all_videos[i][-1]
            while len(all_videos[i]) < max_len:
                all_videos[i].append(last)

    # Fill grid with blanks if fewer than grid slots
    while len(all_videos) < grid_size[0] * grid_size[1]:
        all_videos.append([np.zeros(target_res, dtype=np.uint8)] * max_len)

    # Prepare video writer
    grid_h = target_res[1] * grid_size[0]
    grid_w = target_res[0] * grid_size[1]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 40, (grid_w, grid_h), isColor=False)

    print("🎥 Writing grid video...")
    for f_idx in tqdm(range(max_len), desc="Frames"):
        rows = []
        for row in range(grid_size[0]):
            row_frames = []
            for col in range(grid_size[1]):
                idx = row * grid_size[1] + col
                row_frames.append(all_videos[idx][f_idx])
            rows.append(np.hstack(row_frames))
        grid_frame = np.vstack(rows)
        out.write(cv2.cvtColor(grid_frame, cv2.COLOR_GRAY2BGR))

    out.release()
    print(f"✅ Done! Grid video saved to: {output_path}")

# Example usage:
create_grid_video(
    pupa_id="pupa_1",
    start_rec=1,
    end_rec=50,
    output_path="grid_pupa1_1to50.mp4",
    grid_size=(5,10),
    target_res=(205,140),
    max_frames=None  # or set to something like 300 to limit video length
)
