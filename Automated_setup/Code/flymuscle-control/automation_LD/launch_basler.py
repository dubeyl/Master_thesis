
import time

import os
from pypylon import pylon
from pathlib import Path

import cv2

interframe_delay = 1.0

def camera_capture_loop():
    #time.sleep(36000) #72000 (for 20) wait 10h before starting the camera capture
    # Initialize camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    print("Basler camera == operational")

    # Set camera parameters
    camera.ExposureAuto.SetValue('Off')
    camera.AutoExposureTimeLowerLimit.SetValue(105.0)
    camera.AutoExposureTimeUpperLimit.SetValue(200)#1/interframe_delay*1000.0)
    camera.GainAuto.SetValue('Off')
    camera.AutoGainLowerLimit.SetValue(0.0)
    camera.AutoGainUpperLimit.SetValue(12.00920)
    camera.AutoTargetBrightness.SetValue(0.2)

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
            #img.Save(pylon.ImageFileFormat_Png, file_name)
            #camera.ExposureTime.Value = interframe_delay*1000.0
            camera.ExposureTime.Value = 105.1
            print(camera.Gain.Value)
            cv2.imshow("Basler live", img.Array)
            cv2.waitKey(1)
            print(f"Captured image {index}", end='\r')
    # Capture images every 10 seconds
    index = 0
    while True:
        capture_image(index)
        index += 1
        time.sleep(interframe_delay)

camera_capture_loop()