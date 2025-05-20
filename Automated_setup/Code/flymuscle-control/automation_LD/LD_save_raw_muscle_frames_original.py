import os
import yaml
import json
from pathlib import Path
import cv2
import sys
import re
import numpy as np
import h5py
from tqdm import tqdm, trange
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import subprocess

import tifffile

def extract_recording_number(folder_name):
    match = re.search(r'recording(\d+)_', folder_name)  # Look for "recordingX_"
    return int(match.group(1)) if match else None  # Convert to int if found

def process_folder(data_folder, overwrite=False):
    print(f"Processing folder: {data_folder}")
    sorted_img_files = sorted(data_folder.glob("*.tif"), key=lambda x: int(x.stem))[1:]
    batch_size = 400  # Adjust based on your system's memory capacity
    total_files = len(sorted_img_files)
    batches = (total_files + batch_size - 1) // batch_size
    #print(batches)

    h5_file = data_folder / f"{data_folder.name}_lowcompchunks.h5"
    if False and h5_file.exists() and not overwrite:
        print(f"File {h5_file} already exists. Skipping...")
        return
   
    # get the shape by reading the first image
    try:
        img = tifffile.imread(str(sorted_img_files[0]))
    except Exception as e:  
        print(e)
    #print(len(sorted_img_files))
    try:
        has_timestamp = False
        has_stage_postions = False
        with h5py.File(h5_file, 'w') as f:
            ds_images = f.create_dataset("images", shape=(total_files, *img.shape), compression='gzip', compression_opts=3, dtype='uint16')
            ds_frames_num = f.create_dataset("frames_num", shape=(total_files,), dtype="int")
            if has_timestamp:
                ds_timestamps = f.create_dataset("timestamp", shape=(total_files,), dtype='float')
            if has_stage_postions:
                ds_stage_pos = f.create_dataset("stage_pos", shape=(total_files,), dtype='float')

            for i in range(batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, total_files)
                images = [cv2.imread(str(file), cv2.IMREAD_UNCHANGED) for file in sorted_img_files[start:end]]
                frames_num = [int(file.stem) for file in sorted_img_files[start:end]]
                if has_timestamp:
                    timestamps = [np.loadtxt(file.parent / (file.stem + "_timestamp.txt")) for file in sorted_img_files[start:end]]
                if has_stage_postions:
                    stage_positions = [np.loadtxt(file.parent / (file.stem + "_stage_pos.txt")) for file in sorted_img_files[start:end]]

                ds_images[start:end] = images
                ds_frames_num[start:end] = frames_num
                if has_timestamp:
                    ds_timestamps[start:end] = timestamps
                if has_stage_postions:
                    ds_stage_pos[start:end] = stage_positions
                print(f"Processed batch {i+1}/{batches} for {data_folder}")

    except Exception as e:
        print("Ohoh")
        print(e)

    #print("Moving stuf")
    ##mv the h5 to a the right folder
    #nas_base = Path("/mnt/upramdya_data/VAS/muscle_setup/")
    #parent_dest = nas_base / data_folder.parent.name
    #if not parent_dest.exists():
    #    parent_dest.mkdir()
    #dest_folder = parent_dest / data_folder.name
    #if not dest_folder .exists():
    #    dest_folder.mkdir()

    #subprocess.call(["mv", str(h5_file), str(dest_folder)])


def main():
    config = yaml.load(open("LD_config.yaml", 'r'), Loader=yaml.FullLoader)
    folders = config['folders_muscle']

    if len(sys.argv) > 1:  # Check if an argument was passed
        cnt = int(sys.argv[1])
    else: print("No argument passed, exiting"); sys.exit(1)
    print("Making h5 files for muscle data")

    folders = [Path(folder) for folder in folders if Path(folder).exists()]
    data_folders = []
    for folder in folders:
        for root, dirs, files in os.walk(folder):
            root_path = Path(root)  # Full path of the current directory
            
            # Extract recording number from subdirectories (not the parent directory)
            for subfolder in dirs:
                recording_number = extract_recording_number(subfolder)
                if recording_number is None or recording_number != cnt:  # `cnt` should be the expected recording number
                    print(f"Skipping {subfolder} (recording {recording_number}, expected {cnt})")
                    continue

                # If the recording number matches the expected `cnt`
                subfolder_path = root_path / subfolder
                for file in os.listdir(subfolder_path):
                    # look for any file name that has only numbers and ends with .tiff
                    if file.endswith(".tif") and file[:-4].isdigit() and "temp" not in subfolder_path.name:
                        print(subfolder_path, file)
                        data_folders.append(subfolder_path)  # Add the full subfolder path
                        break
    
    if len(data_folders) == 12:
        print(f"Found {len(data_folders)} data folders, proceeding")
    else: print(f"Expected 12 data folders, found {len(data_folders)}")

    # Create a partial function with common arguments filled
    process_folder_partial = partial(process_folder, overwrite=config['overwrite'])

    # Use ThreadPoolExecutor for I/O-bound tasks
    with ThreadPoolExecutor(max_workers=4) as executor:  # Number of workers can be tuned
        executor.map(process_folder_partial, data_folders)

    #for folder in data_folders:
    #    process_folder(folder, overwrite=config['overwrite'])

if __name__ == "__main__":
    main()
