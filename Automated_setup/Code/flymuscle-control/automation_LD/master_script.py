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
    list_doc["arduino"]["framerate"] = 80
    list_doc["arduino"]["duration"] = "120"
    list_doc["camera"]["exposure_time"] = 0.004
    list_doc["camera"]["offset"] = "864 224" #"2752 224"
    list_doc["camera"]["resolution"] = "864 1920" #"1024 704"
    list_doc["tracking"]['tracking'] = False

    with open(config_file, "w") as f:
        yaml.dump(list_doc, f)

def run_cpp_script(output_dir):#, strobing_flag):
    """
    Runs the C++ script to capture images.

    :param output_dir: Directory to save the images.

    
    :param strobing_flag: Parmeter to decide if you want behavior frames and strobing light. ##not implemented yet
    :return: None
    """
    ###os.system version -> not the one I want
    #cpp_executable = f"cd /home/nely/flymuscle-control/cpp && ./record_parallel_no_track {output_dir}"
    #command = [cpp_executable, output_dir] ##,str(strobing_flag)]
    #os.system(cpp_executable)

    ###subprocess.run version --> the one I want
    cpp_executable = "/home/nely/flymuscle-control/cpp/record_parallel_no_track"
    if not os.path.exists(cpp_executable):
        raise FileNotFoundError(f"Executable not found: {cpp_executable}")

    # Build the command to run
    command = ["strace", "-e", "openat", cpp_executable, output_dir] ##,str(strobing_flag)] -> to add once the cpp has the flag option

    env = os.environ.copy()
    # OPENCV from conda env and c++ get in conflict !!!
    env["LD_LIBRARY_PATH"] = ""

    try:
        #run process
        result = subprocess.run(command, check=True, text=True, capture_output=True, env=env, timeout=180) #timeout 3min
        # Check for any output or errors
        if result.returncode == 0:
            print(f"Recording completed and saved to {output_dir}")
            print(result.stdout)
        else:
            print(f"Error in running C++ script: {result.stderr}")
            print(result.stdout)  
    except subprocess.CalledProcessError as e:
        print(f"Error while running the C++ script: {e}")
        print(e.stdout)
        print(e.stderr)
        return 1
    except subprocess.TimeoutExpired as e:
        print(f"Timeout error: {e}")
        print(e.stdout)
        print(e.stderr) 
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(e.stdout)
        print(e.stderr)  
        return 1
    
    return result.returncode


def camera_capture_loop(base_folder):
    #time.sleep(36000) #72000 (for 20) wait 10h before starting the camera capture
    file_path = os.path.join(base_folder,"eclosion_camera_output") 
    os.makedirs(file_path, exist_ok=True)
    # Initialize camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    print("Basler camera == operational")

    # Set camera parameters
    camera.ExposureAuto.SetValue('Off')
    # camera.AutoExposureTimeLowerLimit.SetValue(1000.0)
    # camera.AutoExposureTimeUpperLimit.SetValue(50000.0)
    camera.GainAuto.SetValue('Off')
    # camera.AutoGainLowerLimit.SetValue(0.0)
    # camera.AutoGainUpperLimit.SetValue(12.0)
    # camera.AutoTargetBrightness.SetValue(0.3)
    camera.ExposureTime.Value = 105.1
    camera.Gain.Value = 12.0

    camera.Width.SetValue(1920)          # Set resolution
    camera.Height.SetValue(1200)

    # Function to capture image
    def capture_image(index):
        if not camera.IsGrabbing():
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        
        if grab_result.GrabSucceeded():
            # Save imageraise ValueError(
            img = pylon.PylonImage()
            img.AttachGrabResultBuffer(grab_result)
            file_name = os.path.join(file_path, 'eclosion_picture_{}.png'.format(index))
            img.Save(pylon.ImageFileFormat_Png, file_name)
            print(f"Captured image {index}", end='\r')
    # Capture images every 10 seconds
    index = 0
    while True:
        capture_image(index)
        index += 1
        time.sleep(5)

def format_and_export():
    time.sleep(1680) #1680 wait 28 min (= roughly after first recordings) before starting

    config = yaml.load(open("LD_config.yaml", 'r'), Loader=yaml.FullLoader)
    folders = config['temp_folder']
    base_folder = config['base_folder']
    #permission_command = ["sudo", "chmod", "-R", "o+rwx", base_folder]
    
    cnt = 0
    #clear the log file
    with open("/home/nely/flymuscle-control/automation_LD/format_export_logs.txt", "w") as log_file:
        pass

    while True:
        #log time of start
        start_time = datetime.datetime.now()
        #subprocess.run(permission_command, check=True)
        #check if there are some .tif are "-pix_fmt", "gray16le",available
        folders = [Path(folder) for folder in folders if Path(folder).exists()]
        data_folders = []
        for folder in folders:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    # look for any file name that has only numbers and ends with .tiff
                    if file.endswith(".tif") and file[:-4].isdigit() and not "temp" in root:
                        data_folders.append(Path(root))
                        break
        ### if no .tif files, kill process
        if not data_folders:
            print("No more .tif files found, exiting format and export process")
            break
        print("New format and export process started")

        #File for logs
        with open("/home/nely/flymuscle-control/automation_LD/format_export_logs.txt", "a") as log_file:
            try:
                # Run the formatting to mkv script
                subprocess.run(
                    ["python", "LD_save_raw_muscle_frames.py", str(cnt)], 
                    check=True,
                    stdout=log_file,
                    stderr=log_file
                )
            except subprocess.CalledProcessError as e:
                print("mkv file creation failed with error:", e.stderr)

            #move the mkv files to the server
            try:
                # Run the second script and capture its output
                subprocess.run(
                    ["python", "LD_move_mkv.py"], 
                    check=True,
                    stdout= log_file,
                    stderr= log_file
                )
            except subprocess.CalledProcessError as e:
                print("Transfer to server failed with error:", e.stderr)
        print("Format and export process number {} completed".format(cnt))
        cnt+=1
        #log time of end
        end_time = datetime.datetime.now()
        time_gone = end_time - start_time
        if time_gone.total_seconds() < 3600:
            time_to_wait = 3600 - time_gone.total_seconds()
            print(f"Waiting {time_to_wait / 60:.2f} minutes before the next format and export process...")
            time.sleep(time_to_wait)

#def move_to_temp(source_folder, temp_folder):
#    # Move the folder to the temp folder
#    os.system(f"mv {source_folder} {temp_folder}")
#    # Remove the source folder
#    os.system(f"rm -rf {source_folder}")


def main():

    update_config_file("/home/nely/flymuscle-control/config/config.yaml")

    #strobing_flag = 0  # no strobing to feed to record_parallel_no_track
    config = yaml.load(open("LD_config.yaml", 'r'), Loader=yaml.FullLoader)
    output_base_dir = str(config['base_folder'])
    #temp_folder = str(config['temp_folder'])
    #os.makedirs(temp_folder, exist_ok=True)

    # Start eclosion camera side process
    camera_process = multiprocessing.Process(target=camera_capture_loop, args=(output_base_dir,))
    camera_process.daemon = True #kill camera process when main dies
    camera_process.start()

    format_and_export_process = multiprocessing.Process(target= format_and_export)
    format_and_export_process.start()

    #update serial port name, here "/dev/zaber_device"
    with Connection.open_serial_port("/dev/zaber_device") as connection:
        connection.enable_alerts()

        device_list = connection.detect_devices()
        print("Found {} zaber devices".format(len(device_list)))

        device = device_list[0]

        axis = device.get_axis(1)

        #number of recordings /change to wanted number
        cnt=0
        while cnt < 60:
            #prepare stage for first recording
            if not axis.is_homed():
                axis.home()
            #### Move to 2mm --> to update to correct value
            axis.move_absolute(44.5, Units.LENGTH_MILLIMETRES)

            start_time = time.time()

            for i in range(12):  # Loop over 12 pupas
                curr_time = datetime.datetime.now()
                time_stamp = curr_time.strftime("%d%m%Y_%H%M")
                output_dir = os.path.join(output_base_dir, f"pupa_{i+1}",f"recording{cnt}_"+str(time_stamp))
                os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
                #temporary_folder = os.path.join(temp_folder, f"pupa_{i+1}")
                #os.makedirs(temporary_folder, exist_ok=True)  # Ensure directory exists

                print(f"Recording pupa {i+1}...")
                status = run_cpp_script(output_dir)#, strobing_flag)
                if status != 0:
                    print("Error in recording, exiting...")
                    camera_process.terminate()
                    continue
                
                ##source_folder = os.path.join(output_base_dir, f"pupa_{i+1}")
                #multiprocessing.Process(target=move_to_temp, args=(output_dir, temporary_folder)).start()

                if i < 11:
                    #move stage for next recording
                    print(f"Moving stage to pupa {i+2}")
                    axis.move_relative(8, Units.LENGTH_MILLIMETRES)

            # Move back to startfolder, exist_ok=True)

            axis.move_absolute(0, Units.LENGTH_MILLIMETRES)

            # Wait so that 1 hours passed before starting the next loop
            end_time = time.time()
            time_gone = end_time - start_time
            time_to_wait = max(0, (60 * 60) - time_gone)
            if time_to_wait > 0:
                print(f"Waiting {time_to_wait / 60:.2f} minutes before the next recording...")
                time.sleep(time_to_wait)
            else:
                print("Recording took longer than 1 hour. Skipping waiting period.")
            cnt+=1

    #kill the eclosion camera side process (should not be needed though)
    camera_process.terminate()
    print('Killing eclosion cam process')
    #make sure the save and export process has finished
    format_and_export_process.join()

if __name__ == "__main__":
    main()
