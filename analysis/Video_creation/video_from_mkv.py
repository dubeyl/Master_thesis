##Create visualisation video from mkv file. You can either do a single video at a time or for a whole pupa or a whole batch of pupae

import os
import subprocess
import tifffile as tiff
import cv2
import numpy as np
from tqdm import tqdm
import datetime

def is_recent_mp4(file_path, skip_dates):
    """Check if the .mp4 was modified on a skip date."""
    if not os.path.exists(file_path):
        return False
    modified_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
    file_date = modified_time.strftime("%d.%m")
    return file_date in skip_dates

def extract_frames_from_mkv(mkv_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    command = [
        "ffmpeg", "-i", mkv_path,
        "-pix_fmt", "gray16le",
        os.path.join(output_folder, "frame_%04d.tif")
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    print(f"Extracted frames saved to {output_folder}")

def process_and_save_frames(output_folder, processed_folder, select_frames='all', brightness_increase=1):
    os.makedirs(processed_folder, exist_ok=True)
    frame_files = sorted(os.listdir(output_folder))
    if not frame_files:
        print("No frames found!")
        return

    min_val, max_val = 0, 860
    print(f"Intensity Range: Min = {min_val}, Max = {max_val}")

    if select_frames == 'odd':
        frame_files = frame_files[1::2]
    elif select_frames == 'even':
        frame_files = frame_files[0::2]

    for i, frame_file in enumerate(tqdm(frame_files, desc="Processing frames")):
        img = tiff.imread(os.path.join(output_folder, frame_file))
        normalized = ((img - min_val) / (max_val - min_val) * 255).clip(0, 255).astype(np.uint8)
        final_frame = (normalized * brightness_increase).clip(0, 255).astype(np.uint8)
        frame_path = os.path.join(processed_folder, f"frame_{i:06d}.png")
        cv2.imwrite(frame_path, final_frame)

def create_video_from_frames(processed_folder, output_video_path, target_fps=40):
    ffmpeg_cmd = [
        "ffmpeg",
        "-framerate", str(target_fps),
        "-i", os.path.join(processed_folder, "frame_%06d.png"),
        "-c:v", "libx264",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-y",
        output_video_path
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Video saved to {output_video_path}")

def create_video_from_mkv(mkv_file_path, output_video_path, target_fps=40, select_frames='all', brightness_increase=1):
    print(f"Processing {os.path.basename(mkv_file_path)}")
    temp_frame_dir = "extracted_frames"
    processed_frame_dir = "processed_frames"
    extract_frames_from_mkv(mkv_file_path, temp_frame_dir)
    process_and_save_frames(temp_frame_dir, processed_frame_dir, select_frames, brightness_increase)
    create_video_from_frames(processed_frame_dir, output_video_path, target_fps)

    for folder in [temp_frame_dir, processed_frame_dir]:
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))
        os.rmdir(folder)

    print(f"Completed processing for {mkv_file_path}")

def create_videos(
        root_folder,
        f_struct=None,
        number=None,
        fps=40,
        select_frames='all',
        output_video_name="output_video.mp4",
        skip_dates=None):

    if skip_dates is None:
        skip_dates = set()

    def should_skip_folder(recording_folder):
        for file in os.listdir(recording_folder):
            if file.endswith(".mp4"):
                full_path = os.path.join(recording_folder, file)
                # if is_recent_mp4(full_path, skip_dates):
                print(f"Skipping {full_path} due to recent modification date.")
                return True
        return False

    if f_struct == 'all':
        for pupa in os.listdir(root_folder):
            pupa_folder = os.path.join(root_folder, pupa)
            if os.path.isdir(pupa_folder):
                for folder in os.listdir(pupa_folder):
                    recording_folder = os.path.join(pupa_folder, folder)
                    if os.path.isdir(recording_folder) and not should_skip_folder(recording_folder):
                        for recording in os.listdir(recording_folder):
                            if recording.endswith(".mkv"):
                                mkv_file_path = os.path.join(recording_folder, recording)
                                output_video_path = os.path.join(recording_folder, output_video_name)
                                create_video_from_mkv(mkv_file_path, output_video_path, fps, select_frames)

    elif f_struct == 'pupa':
        for folder in os.listdir(root_folder):
            recording_folder = os.path.join(root_folder, folder)
            if os.path.isdir(recording_folder) and not should_skip_folder(recording_folder):
                for recording in os.listdir(recording_folder):
                    if recording.endswith(".mkv"):
                        mkv_file_path = os.path.join(recording_folder, recording)
                        output_video_path = os.path.join(recording_folder, output_video_name)
                        create_video_from_mkv(mkv_file_path, output_video_path, fps, select_frames)

    elif f_struct == 'recording':
        if not should_skip_folder(root_folder):
            for recording in os.listdir(root_folder):
                if recording.endswith("corrected.mkv"):
                    mkv_file_path = os.path.join(root_folder, recording)
                    output_video_path = os.path.join(root_folder, output_video_name)
                    create_video_from_mkv(mkv_file_path, output_video_path, fps, select_frames)

# --- Main execution ---

# Select root_folder and according f_struct: all, pupa, or recording.
# For downsampling, select 'odd'.
# Adapt output video name.
#change skip dates or set to none
root_folder = "Z:/data/LD/experiments/imaging/output/Automated_experiment"

create_videos(
    root_folder=root_folder,
    f_struct='all',
    fps=40,
    select_frames='odd',
    output_video_name="output_video_40fps.mp4",
    skip_dates={'15.04', '16.04'}
)
