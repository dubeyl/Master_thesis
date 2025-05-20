// SimpleExample.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und
// endet die Ausführung des Programms.
//
//#pragma once

#include <chrono>
#include "opencv2/imgcodecs.hpp"
#include "stdafx.h"
#include "camera.h"
#include "cameraexception.h"
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string.h>


namespace fs = std::filesystem;

int countFiles(const std::string& folderPath) {
    int count = 0;
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (fs::is_regular_file(entry)) {
            count++;
        }
    }
    return count;
}

void setCameraConfig(pco::Camera &cam){

    cam.defaultConfiguration();
    pco::Configuration config = cam.getConfiguration();
    pco::Description desc = cam.getDescription();

    config.roi.x0 = 1+20 * 32;
    config.roi.y0 = 15*32;
    config.roi.x1 = 2048 - 16 * 32;
    config.roi.y1 = 2048-40*32;
    config.trigger_mode = TRIGGER_MODE_EXTERNALTRIGGER; //TRIGGER_MODE_AUTOTRIGGER //TRIGGER_MODE_EXTERNALTRIGGER
    config.acquire_mode = ACQUIRE_MODE_AUTO;
    //config.delay_time_s = 0;
    //config.noise_filter_mode = NOISE_FILTER_MODE_ON;

    cam.setConfiguration(config);
}

void setCameraLine4(pco::Camera &cam){
    //Setup line 4 to signal the exposure ON for all lines
    
    PCO_Signal io_signal_write;
    io_signal_write.wSize = sizeof(io_signal_write);
    WORD line = 3;
    int failed_read = PCO_GetHWIOSignal(cam.sdk(), line, &io_signal_write);
    assert(failed_read == 0);
    // Signal the exposure
    io_signal_write.dwSignalFunctionality[0] = HW_IO_SIGNAL_TIMING_TYPE_EXPOSURE_RS; //shows exposure
    // Signal exposure for all lines (rolling shutter)
    io_signal_write.dwParameter[0] = HW_IO_SIGNAL_TIMING_EXPOSURE_RS_ALLLINES; //all lines
    int failed_write = PCO_SetHWIOSignal(cam.sdk(), line, &io_signal_write);
    assert(failed_write == 0);
}

int main() {
  try {
    //setup the camera
    pco::Camera cam = pco::Camera();
    pco::Image img;
    int image_count = 20;
    cam.defaultConfiguration();

    pco::Configuration config = cam.getConfiguration(); 
    config.trigger_mode = TRIGGER_MODE_AUTOTRIGGER;

    cam.setConfiguration(config);
    cam.setExposureTime(0.01);
    
    //setCameraLine4(cam);

    //pco::Configuration config3 = cam.getConfiguration();
    //std::cout<<config3.trigger_mode<<std::endl;

    int width, height;
    //cam.setConfiguration(config);

    double image_timeout_s = 1.0;
    cv::namedWindow("Muscle", cv::WINDOW_AUTOSIZE);
    std::cout << "Recording" << std::endl;
    cam.record(image_count, pco::RecordMode::ring_buffer);
    std::cout << "Entering the main loop" << std::endl;
    cam.waitForFirstImage(true, image_timeout_s);

    auto start_t = std::chrono::high_resolution_clock::now();
    auto start_epoch = start_t.time_since_epoch();
    double last_checkpoint_time = std::chrono::duration<double>(start_epoch).count();
    int n_frames_fps_calc = 200;

    for (int i = 0; i < 99999; i++) {
      // Check real FPS
    if ((i > 0) && (i % n_frames_fps_calc == 0)) {
      auto now = std::chrono::high_resolution_clock::now();
      auto dur_since_epoch = now.time_since_epoch();
      double new_checkpoint_time = std::chrono::duration<double>(dur_since_epoch).count();
      auto real_fps = n_frames_fps_calc / (new_checkpoint_time - last_checkpoint_time);
      last_checkpoint_time = new_checkpoint_time;
      printf("  Muscles real FPS (frames %d-%d): %f\n", i - n_frames_fps_calc, i - 1,
             real_fps);
    }
      if (i==0){
        std::cout << "waiting for first frame" << std::endl;
      }
      cam.waitForNewImage(true, image_timeout_s);
      cam.image(img, PCO_RECORDER_LATEST_IMAGE, pco::DataFormat::Mono16);
      if (i==0){
        std::cout << "First frame aquired" << std::endl;  // 8
      }
      int width = img.width();
      int height = img.height();
      cv::Mat frame(height, width, CV_16UC1, img.raw_data().first);
      cv::Mat disp_img;
      //disp_img = frame;
      cv::resize(frame, disp_img, cv::Size(), 0.25, 0.25, cv::INTER_NEAREST);
      // cv::imshow("Muscle", disp_img * 1);
      cv::imshow("Muscle", disp_img * 100.0);
      std::string frame_id_str = std::to_string(i);
      std::string filename =
          "output/" + std::string(6 - std::min(6, (int)frame_id_str.length()), '0') + frame_id_str + ".tif";
      cv::imwrite(filename, frame);
      if (cv::waitKey(1) == 27) {
        break;
       }
    }
    std::cout<<"Recording done"<<std::endl;
    cam.stop();
    std::cout<<"Camera stopped"<<std::endl;
  } catch (pco::CameraException &err) {
    std::cout << "Error Code: " << (uint)err.error_code() << std::endl;
    std::cout << err.what() << std::endl;
  }

  cv::destroyAllWindows();
  std::cout << "All done" << std::endl;
  return 0;
}


// very interesting: PCO_GetCmosLineTiming