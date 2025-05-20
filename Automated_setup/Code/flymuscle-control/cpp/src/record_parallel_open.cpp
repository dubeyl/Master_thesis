/*
TODO:
1. Refactor config file; move magic numbers there
2. Refactor two translation stages as a single class
3. Get output directory from user input
*/

#include "utils.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "stdafx.h"
#include "camera.h"
#include "cameraexception.h"
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <yaml-cpp/yaml.h>
//#include <zaber/motion/ascii.h>

using namespace std;
using namespace pco;

struct ImageData {
  int frame_id;         // id of the frame
  cv::Mat image;        // 1-channel image
};

struct ImageSize {
  size_t height;
  size_t width;
};


int clipToMultipleOf32(int value) {
  int remainder = value % 32;
  if (remainder > 16) {
    return value + (32 - remainder);
  } else {
    return value - remainder;
  }
}

// ========== Global variables ==========
// Synchronization primatives
mutex recording_mtx;
mutex live_mtx;
condition_variable recording_condvar;
condition_variable live_condvar;
atomic<bool> is_finished(false);

const int live_stream_int = 1;

// Data containers
queue<ImageData> image_queue;
cv::Mat live_image;

void image_acquirer(Camera *cam_ptr,
                    int num_frames, int live_stream_int, size_t image_width,
                    size_t image_height, int image_count,
                    DataFormat format, double timeout_s) {
  try{
    Image img;

    cam_ptr->record(image_count, RecordMode::ring_buffer);
    cam_ptr->waitForFirstImage(true, timeout_s);
    cout << "Recorded the first image: Entering the main loop" << endl;
    cout << "Num frames " << to_string(num_frames) <<endl; 
    auto last_checkpoint_time = get_current_time();
    for (int frame_id = 1; frame_id < num_frames; ++frame_id) {
      // Check real FPS
      if ((frame_id > 0) && (frame_id % 100 == 0)) {
        auto new_checkpoint_time = get_current_time();
        auto real_fps = 100 / (new_checkpoint_time - last_checkpoint_time);
        last_checkpoint_time = new_checkpoint_time;
        printf("  Real FPS (frames %d-%d): %f\n", frame_id - 100, frame_id - 1,
              real_fps);
      }
      // Fetch new frame
      cam_ptr->waitForNewImage(false, 0.5);
      cam_ptr->image(img, PCO_RECORDER_LATEST_IMAGE, format);
      if (frame_id == 0) {
                  cout << "First frame acquired" << endl;
              }

      int width = img.width();
      int height = img.height();
      cv::Mat frame(height, width,
      CV_16UC1, img.raw_data().first);

      ImageData img_data;
      img_data.frame_id = frame_id;
      img_data.image = frame;
      unique_lock<mutex> ul(recording_mtx);
      image_queue.push(img_data);
      ul.unlock();
      recording_condvar.notify_one();

      // Update image for closed-loop control and live streaming
      if (frame_id % live_stream_int == 0) {
        unique_lock<mutex> ul(live_mtx);
        cv::resize(frame, live_image, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
        ul.unlock();
        live_condvar.notify_all();
        
      }
      if (is_finished.load()) {
        break;
      }
    }
    cam_ptr->stop();
    is_finished.store(true);
    recording_condvar.notify_all();
    cout << "Image acquirer thread exiting" << endl;
  }catch (CameraException& err) {
      cout << "Error Code: " << (uint)err.error_code() << endl;
      cout << err.what() << endl;
      is_finished.store(true);
      recording_condvar.notify_all();
      cout << "Image acquirer thread exiting" << endl;
  }
}

void image_saver(const string output_dir) {
  while (true) {
    // Get an image from the queue
    unique_lock<mutex> ul(recording_mtx);
    recording_condvar.wait(
        ul, []() { return !image_queue.empty() || is_finished.load(); });
    ImageData image_data;
    if (is_finished.load()) {
      ul.unlock();
      break;
    } else {
      image_data = image_queue.front();
      image_queue.pop();
      ul.unlock();
    }

    // Write image to disk
    string filename_stem = output_dir + to_string(image_data.frame_id);
    cv::imwrite(filename_stem + ".tif", image_data.image);
  }
  cout << "Image saver thread exiting" << endl;
}

void live_display() {

  while (true) {
    unique_lock<mutex> ul(live_mtx);
    live_condvar.wait(
        ul, []() { return !live_image.empty() || is_finished.load(); });
    cv::Mat my_live_image = live_image.clone();
    ul.unlock();
    cv::Mat display_image;
    cv::cvtColor(my_live_image, display_image, cv::COLOR_GRAY2BGR);
    // Live display
    cv::imshow("Behavior Camera", display_image*100.0);
    if (cv::waitKey(1) == 27) {
      is_finished.store(true);
      break;
    }
    if (is_finished.load()) {
      break;
    }
  }
  cv::destroyAllWindows();
  cout << "Live streamer thread exiting" << endl;
}

int main(int argc, char **argv) {
  cout<<"Program started"<<endl;
  // Parameters
  const float fps = 30; 
  const float record_time = 20;
  const int num_saver_threads = 4;
  const double exposure_time = 0.0005;
  const int image_count = 10; //number of images that can be stored in the ring buffer
  const double acquire_timeout_s = 0.1; // should be seconds from Pandas API but is not
  const DataFormat format = DataFormat::Mono16;

  //const float gain = 2.0;
  const string output_dir = (string)argv[1] + "/";
  const int num_frames = record_time * fps;

  size_t target_image_width = 0;
  size_t target_image_height = 0;  

  cout<<"Init cam object"<<endl;
  Camera cam;

  try{
    // Configure camera
    cout<<"Setting up camera configuration"<< endl;    
    cam.defaultConfiguration();

    Configuration config = cam.getConfiguration();
    config.roi.x0 = 1;
    config.roi.y0 = clipToMultipleOf32(500);
    config.roi.x1 = 2048;
    config.roi.y1 = clipToMultipleOf32(1500);
    config.trigger_mode = TRIGGER_MODE_AUTOTRIGGER; //TRIGGER_MODE_AUTOTRIGGER //TRIGGER_MODE_EXTERNALTRIGGER
    config.acquire_mode = ACQUIRE_MODE_AUTO;
    config.delay_time_s = 0;
    config.noise_filter_mode = NOISE_FILTER_MODE_ON;

    ImageSize image_size;
    image_size.width = config.roi.x1 - config.roi.x0;
    image_size.height = config.roi.y1 - config.roi.y0;

    cam.setConfiguration(config);
    cout<<"Done."<<endl;

    cout<<"Set exposure time"<<endl;
    cam.setExposureTime(0.0005);
    cout<<"Done."<<endl;
    
    //Setup line 4 to strobe the excitation light
    cout<<"Configure line 4 (exposure ON output)"<<endl;
    PCO_Signal io_signal_write;
    io_signal_write.wSize = sizeof(io_signal_write);
    WORD output_line = 3;
    int failed_read = PCO_GetHWIOSignal(cam.sdk(), output_line, &io_signal_write);
    assert(failed_read == 0);
    // Signal the exposure
    io_signal_write.dwSignalFunctionality[0] = HW_IO_SIGNAL_TIMING_TYPE_EXPOSURE_RS; //shows exposure
    // Signal exposure for all lines (rolling shutter)
    io_signal_write.dwParameter[0] = HW_IO_SIGNAL_TIMING_EXPOSURE_RS_ALLLINES; //all lines
    int failed_write = PCO_SetHWIOSignal(cam.sdk(), output_line, &io_signal_write);
    assert(failed_write == 0);
    cout<<"Done."<<endl;
    

    size_t set_image_width = config.roi.x1 - config.roi.x0;
    size_t set_image_height = config.roi.y1 - config.roi.y0;

    cout << "  Set resolution: H=" + to_string(set_image_height) +
                ", W=" + to_string(set_image_width)
        << endl;

    prepare_output_dir(output_dir);
    cout << "Writing data to " + output_dir << endl;
  } catch (CameraException& err) {
      cout << "Error Code: " << (uint)err.error_code() << endl;
      cout << err.what() << endl;
      return 1;
  }

  Camera* cam_ptr = &cam;
  // Initialize threads
  thread acquirer_thread(image_acquirer, cam_ptr,
                          num_frames, live_stream_int,
                          target_image_width, target_image_height,
                          image_count, format, acquire_timeout_s);
  vector<thread> saver_threads;
  for (int i = 0; i < num_saver_threads; ++i) {
    thread saver_thread(image_saver, output_dir);
    saver_threads.push_back(move(saver_thread));
  }
  thread live_display_thread(live_display);

  // Wait for threads to finish
  acquirer_thread.join();
  /*for (auto &saver_thread : saver_threads) {
    saver_thread.join();
  }
  */
  live_display_thread.join();

  cout << "All done." << endl;

  return 0;
}