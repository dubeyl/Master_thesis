## Script to overlay dynamic heatmap of heart muscle activity on a static background pupa video

import os
import re
import pickle
import cv2
import numpy as np
import subprocess
from tqdm import tqdm

def get_ordered_recording_paths(pupa_folder):
    pattern = re.compile(r"recording(\d+)_\d{8}_\d{4}")
    recordings = []
    for folder in os.listdir(pupa_folder):
        match = pattern.match(folder)
        if match:
            recordings.append((int(match.group(1)), folder))
    recordings.sort(key=lambda x: x[0])
    return recordings


def invert_crop_and_resize(mask, crop_off, orig_size, reduce_factor):
    up_h = mask.shape[0] * reduce_factor
    up_w = mask.shape[1] * reduce_factor
    up_mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_NEAREST)

    full_mask = np.zeros(orig_size, dtype=mask.dtype)
    x_off, y_off = crop_off
    full_mask[y_off:y_off+up_h, x_off:x_off+up_w] = up_mask
    return full_mask


def enlarge_mask(mask, scale=2.0):
    ring = np.zeros_like(mask)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = M['m10']/M['m00']
        cy = M['m01']/M['m00']
        scaled = []
        for [[x, y]] in cnt:
            x_new = cx + scale * (x - cx)
            y_new = cy + scale * (y - cy)
            scaled.append([[int(round(x_new)), int(round(y_new))]])
        cnt_scaled = np.array(scaled, dtype=np.int32)
        cv2.fillPoly(ring, [cnt_scaled], 255)
    return ring


def compute_background(cap, num_frames=10):
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.astype(np.float32))
    mean_frame = np.mean(frames, axis=0).astype(np.uint8)
    return mean_frame


def write_dynamic_heatmap_video(video_path, masks, out_path, crop_off, reduce_factor):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Cannot read video {video_path}")
    h_full, w_full = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # compute static background
    background = compute_background(cap, num_frames=10)

    # prepare FFmpeg pipe
    ffmpeg_cmd = [
        'ffmpeg','-y',
        '-f','rawvideo','-vcodec','rawvideo',
        '-pix_fmt','bgr24','-s',f'{w_full}x{h_full}',
        '-r',str(int(fps)),'-i','pipe:0',
        '-an','-c:v','libx264','-pix_fmt','yuv420p',out_path
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    # rewind to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        heat_full = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        # combine all masks
        combined = np.zeros((h_full, w_full), dtype=np.uint8)
        for m in masks:
            combined = cv2.max(combined, m)
        # overlay: replace masked regions with heatmap
        out_frame = background.copy()
        mask_bool = combined > 0
        out_frame[mask_bool] = heat_full[mask_bool]
        proc.stdin.write(out_frame.tobytes())

    proc.stdin.close()
    proc.wait()
    cap.release()

if __name__ == "__main__":
    pupa_folder = r"Z:/data/LD/experiments/imaging/output/Automated_experiment/pupa_10"
    crop_x, crop_y = 127, 724
    reduce_factor = 2
    rec_num = 40
    mask_file = os.path.join(pupa_folder, "muscle_masks.pkl")

    recs = get_ordered_recording_paths(pupa_folder)
    rec_folder = next((name for rn,name in recs if rn==rec_num), None)
    if rec_folder is None:
        raise ValueError(f"Recording {rec_num} not found")

    video_path = os.path.join(pupa_folder, rec_folder, f"output_video_40fps.mp4")
    out_path = os.path.join(pupa_folder, rec_folder, f"heatmap_mask_{rec_num}.mp4")

    # load and process masks
    with open(mask_file,'rb') as f:
        masks_reduced = pickle.load(f)
    cap_tmp = cv2.VideoCapture(video_path)
    ret, frame = cap_tmp.read()
    cap_tmp.release()
    h_full, w_full = frame.shape[:2]

    masks_full = []
    for m in masks_reduced:
        full = invert_crop_and_resize(m, (crop_x,crop_y), (h_full,w_full), reduce_factor)
        big = enlarge_mask(full, scale=2.0)
        masks_full.append(big)

    write_dynamic_heatmap_video(video_path, masks_full, out_path, (crop_x,crop_y), reduce_factor)
    print(f"Saved dynamic heatmap overlay video to {out_path}")
