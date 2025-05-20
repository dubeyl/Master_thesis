import time
import yaml
import datetime
import subprocess
import os
from zaber_motion import Units
from zaber_motion.ascii import Connection
import multiprocessing
from pypylon import pylon
from pathlib import Path

import LD_save_raw_muscle_frames
import LD_move_mkv

def update_config_file(config_file):
    with open(config_file) as f:
        list_doc = yaml.safe_load(f)

    list_doc["arduino"]["alternating"] = False
    list_doc["arduino"]["optogenetics"] = False
    list_doc["arduino"]["framerate"] = 80
    list_doc["arduino"]["duration"] = "120"
    list_doc["camera"]["exposure_time"] = 0.004
    list_doc["camera"]["offset"] = "864 224"
    list_doc["camera"]["resolution"] = "864 1920"
    list_doc["tracking"]['tracking'] = False

    with open(config_file, "w") as f:
        yaml.dump(list_doc, f)

def run_cpp_script(output_dir):
    cpp_executable = "/home/nely/flymuscle-control/cpp/record_parallel_no_track"
    if not os.path.exists(cpp_executable):
        raise FileNotFoundError(f"Executable not found: {cpp_executable}")

    command = ["strace", "-e", "openat", cpp_executable, output_dir]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ""

    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True, env=env, timeout=180)
        if result.returncode == 0:
            print(f"Recording completed and saved to {output_dir}")
        else:
            print(f"Error in running C++ script: {result.stderr}")
        return result.returncode
    except subprocess.SubprocessError as e:
        print(f"Error running C++ script: {e}")
        return 1

def camera_capture_loop(base_folder):
    file_path = os.path.join(base_folder, "eclosion_camera_output")
    os.makedirs(file_path, exist_ok=True)

    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    print("Basler camera operational")

    camera.ExposureAuto.SetValue('Off')
    camera.GainAuto.SetValue('Off')
    camera.ExposureTime.Value = 105.1
    camera.Gain.Value = 12.0
    camera.Width.SetValue(1920)
    camera.Height.SetValue(1200)

    def capture_image(index):
        if not camera.IsGrabbing():
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        
        if grab_result.GrabSucceeded():
            img = pylon.PylonImage()
            img.AttachGrabResultBuffer(grab_result)
            file_name = os.path.join(file_path, f'eclosion_picture_{index}.png')
            img.Save(pylon.ImageFileFormat_Png, file_name)
            print(f"Captured image {index}", end='\r')

    index = 0
    while True:
        capture_image(index)
        index += 1
        time.sleep(5)

import queue

def format_and_export(recording_queue):
    print("Starting format_and_export queue process.")
    while True:
        try:
            recording = recording_queue.get(timeout=120)  # Wait 2 minutes for new recordings
            if recording is None:
                print("Shutdown signal received. Processing remaining recordings before exiting...")
                break  # Stop processing when shutdown signal is received
            
            print(f"Processing recording: {recording}")
            subprocess.run(["python", "LD_save_raw_muscle_frames.py", str(recording)], check=True)
            subprocess.run(["python", "LD_move_mkv.py", str(recording)], check=True)
            
        except queue.Empty:
            print("No new recordings found. Checking again in 2 minutes...")
    
    # Process any remaining items in the queue before exiting
    while not recording_queue.empty():
        recording = recording_queue.get()
        if recording is None:
            break
        print(f"Processing remaining recording: {recording}")
        subprocess.run(["python", "LD_save_raw_muscle_frames.py", str(recording)], check=True)
        subprocess.run(["python", "LD_move_mkv.py", str(recording)], check=True)
    
    print("All recordings processed. Exiting format_and_export.")

def main():
    update_config_file("/home/nely/flymuscle-control/config/config.yaml")

    config = yaml.load(open("LD_config.yaml", 'r'), Loader=yaml.FullLoader)
    output_base_dir = str(config['base_folder'])

    recording_queue = multiprocessing.Queue()

    # Start camera capture
    camera_process = multiprocessing.Process(target=camera_capture_loop, args=(output_base_dir,))
    camera_process.daemon = True
    camera_process.start()

    # Start queue-based export process
    export_process = multiprocessing.Process(target=format_and_export, args=(recording_queue,))
    export_process.daemon = True
    export_process.start()

    with Connection.open_serial_port("/dev/zaber_device") as connection:
        connection.enable_alerts()
        device_list = connection.detect_devices()
        print("Found {} zaber devices".format(len(device_list)))
        device = device_list[0]
        axis = device.get_axis(1)

        cnt = 0
        while cnt < 60:
            if not axis.is_homed():
                axis.home()
            axis.move_absolute(43.5, Units.LENGTH_MILLIMETRES)

            start_time = time.time()

            for i in range(12):
                curr_time = datetime.datetime.now()
                time_stamp = curr_time.strftime("%d%m%Y_%H%M")
                output_dir = os.path.join(output_base_dir, f"pupa_{i+1}", f"recording{cnt}_{time_stamp}")
                os.makedirs(output_dir, exist_ok=True)

                print(f"Recording pupa {i+1}...")
                status = run_cpp_script(output_dir)

                if status != 0:
                    print("Error in recording, exiting...")
                    camera_process.terminate()
                    continue

                recording_queue.put(output_dir)  # Add to queue for processing

                if i < 11:
                    print(f"Moving stage to pupa {i+2}")
                    axis.move_relative(8, Units.LENGTH_MILLIMETRES)

            axis.move_absolute(0, Units.LENGTH_MILLIMETRES)

            end_time = time.time()
            time_to_wait = max(0, (60 * 60) - (end_time - start_time))
            if time_to_wait > 0:
                print(f"Waiting {time_to_wait / 60:.2f} minutes before next recording...")
                time.sleep(time_to_wait)
            else:
                print("Recording took longer than 1 hour. Skipping waiting period.")
            cnt += 1

    camera_process.terminate()
    print('Killing eclosion cam process')

    recording_queue.put(None)  # Signal to stop processing
    export_process.join()  # Ensure it finishes processing

if __name__ == "__main__":
    main()
