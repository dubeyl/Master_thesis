/*
TODO:
1. Refactor config file; move magic numbers there
2. Refactor two translation stages as a single class
3. Get output directory from user input
*/

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "stdafx.h"
#include "camera.h"
#include "cameraexception.h"
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <boost/asio.hpp>
#include <csignal>
#include <unistd.h>

#include "utils/utils.h"
#include "utils/utils_serial.h"
#include "utils/utils_camera.h"
#include "utils/utils_tracking.h"

using namespace std;
using namespace pco;
using namespace boost::asio;
using namespace std::chrono_literals;

int n_channels = 4;

// ========== Global variables ==========
// Synchronization primatives
mutex recording_mtx;
mutex live_mtx;
condition_variable recording_condvar;
condition_variable live_condvar;
atomic<bool> is_finished(false);

const int live_stream_int = 1;

// Data containers
queue<ImageDataVec> image_queue;

cv::Mat live_image;
int live_image_frame_id;
bool live_image_is_kin;
bool new_live_image = false;

// Config
std::string config_path = "/home/nely/flymuscle-control/config/config.yaml";

void image_acquirer(Camera& cam, serial_port& port, io_service& io,
                    int num_frames, int live_stream_int, size_t image_width,
                    size_t image_height, int image_count,
                    DataFormat format, double timeout_s) {
  try{
    Image img;
    int queue_size = 0;

    bool is_kin = false;
    long long timestamp = 0;

    cam.record(image_count, RecordMode::ring_buffer);
    // Last handshake
    cout << "Num frames " << to_string(num_frames) <<endl; 
    
    waitForArduinoReady(io, port, "Ready");
    send_command(port, "launch");

    cout << "Arduino is ready" << endl;
  
    // Launch the triggering here
    ImageDataVec img_data(n_channels);
    is_kin = get_is_kinematic(port);
    cam.waitForFirstImage(true, timeout_s);
    cout << "Recorded the first image: Entering the main loop" << endl;
    timestamp = getHighResolutionTimestamp();
    auto last_checkpoint_time = get_current_time();
    cam.image(img, PCO_RECORDER_LATEST_IMAGE, format);
    cv::Mat frame(image_height, image_width, CV_16UC1, img.raw_vector_16bit().data());
    
    img_data.addData(
      0,
      timestamp,
      frame.clone(),
      is_kin,
      0.0
    );

    for (int frame_id = 1; frame_id < num_frames; ++frame_id) {
      // Check real FPS
      if ((frame_id > 0) && (frame_id % 300  == 0)) {
        auto new_checkpoint_time = get_current_time();
        auto real_fps = 300  / (new_checkpoint_time - last_checkpoint_time);
        last_checkpoint_time = new_checkpoint_time;
        printf("  Real FPS (frames %d-%d): %f\n", frame_id - 300 , frame_id - 1,
              real_fps);
      }
      is_kin = get_is_kinematic(port);
      // Fetch new frame
      cam.waitForNewImage(true, timeout_s);
      //get the timestamp
      timestamp = getHighResolutionTimestamp();
      cam.image(img, PCO_RECORDER_LATEST_IMAGE, format);

      cv::Mat frame(image_height, image_width, CV_16UC1, img.raw_vector_16bit().data());
      
      img_data.addData(
        frame_id,
        timestamp,
        frame.clone(),
        is_kin,
        0.0
      );

      if(img_data.isFull()) {
        unique_lock<mutex> ul(recording_mtx);
        image_queue.push(img_data);
        queue_size = image_queue.size();
        ul.unlock();
        recording_condvar.notify_one();
      }

      if (queue_size > 40){
        cout<<"Stopping because the image queue is getting too full"<<endl;
        break;
      }

      // Update image for closed-loop control and live streaming
      if (frame_id % live_stream_int == 0) {
          unique_lock<mutex> ul(live_mtx);
          //carefull here the resiying needs to be the same as in the calibration
          cv::resize(frame.clone(), live_image, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
          live_image_is_kin = is_kin;
          live_image_frame_id = frame_id;
          new_live_image = true;
          ul.unlock();
          live_condvar.notify_all();
      }
      if (is_finished.load()) {
        break;
      }
    }
    stopArduino(port);
    is_finished.store(true);
    recording_condvar.notify_all();
    live_condvar.notify_all();
  }catch (CameraException& err) {
      cout << "Error Code: " << (uint)err.error_code() << endl;
      cout << err.what() << endl;
      stopArduino(port);
      is_finished.store(true);
      recording_condvar.notify_all();
      live_condvar.notify_all();
  }
  cout << "Image acquirer thread exiting" << endl;
}

void image_saver(const string output_dir) {
  vector<int> compression_params = {cv::IMWRITE_TIFF_COMPRESSION, 1}; 
  vector<int> jpg_compression_params = {cv::IMWRITE_JPEG_QUALITY, 100};
  
  while (true) {

    ImageDataVec image_data(n_channels);

    // Get an image from the queue
    unique_lock<mutex> ul(recording_mtx);
    recording_condvar.wait(
        ul, []() { return !image_queue.empty() || (is_finished.load() && image_queue.empty()); });

    if (is_finished.load() && image_queue.empty()) {
      ul.unlock();
      break;
    } else {
      image_data = move(image_queue.front());
      image_queue.pop();
      ul.unlock();
    }

    // Write image to output_dir
    string filename_stem = output_dir + to_string(image_data.getFrameIds()[0]);
    cv::Mat merged_image;
    cv::merge(image_data.getImages(), merged_image);
    cv::imwrite(filename_stem + ".jpg", merged_image, jpg_compression_params);
    
    // cv::imwrite(filename_stem + ".tiff", merged_image, compression_params);

    ofstream metadata_file(filename_stem + "metadata.csv", ios::app);
    for (int i = 0; i < n_channels; ++i) {
      metadata_file << image_data.getFrameIds()[i] << ","
                    << image_data.getTimestamps()[i] << ","
                    << image_data.getIsKinematics()[i] << ","
                    << image_data.getStagePos()[i] << endl;
    }
    metadata_file.close();    
  }
  cout << "Image saver thread exiting" << endl;
}

void live_display(ImageParameters image_parameters) {

  // Create two windows with different names
  cv::namedWindow("Behavior frames", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Muscle frames", cv::WINDOW_AUTOSIZE);

  double display_width = image_parameters.width*0.5;

  // Move windows to specific positions to show them side by side
  cv::moveWindow("Behavior frames", 100, 100);   // Move "Window 1" to position (0, 100)
  cv::moveWindow("Muscle frames", display_width/2+100, 100); // Move "Window 2" to position (350, 100)

  while (true) {
    unique_lock<mutex> ul(live_mtx);
    live_condvar.wait_for(ul, std::chrono::milliseconds(1000), []() { 
      return (!live_image.empty() && new_live_image) || is_finished.load(); 
    });

    // Check if the loop should terminate
    if (is_finished.load()) {
      ul.unlock();
      break;
    }
    
    // Only process the image if it's not empty
    if (!live_image.empty()) {
      cv::Mat my_live_image = live_image.clone();
      bool my_is_kin = live_image_is_kin;
      new_live_image = false;
      int my_live_image_frame_id = live_image_frame_id;
      ul.unlock();
      //cout<<"Frame id: "<<my_live_image_frame_id<< " is kinematic: "<<my_is_kin<<endl;
      cv::Mat display_image;
      cv::cvtColor(my_live_image, display_image, cv::COLOR_GRAY2BGR);
      cv::resize(display_image, display_image, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
      // Live display
      if (my_is_kin){
        cv::imshow("Behavior frames", display_image*100.0);
      }else{
        cv::imshow("Muscle frames", display_image*100.0);
      }
      if (cv::waitKey(1) == 27) {
        is_finished.store(true);
        break;
      }
    } else {
      ul.unlock();
    }
  }

  cout << "destroying windows" << endl;
  cv::destroyAllWindows();
  cout << "Live streamer thread exiting" << endl;
}

int main(int argc, char **argv) {
  //printf("OpenCV: %s", cv::getBuildInformation().c_str());
  
  cout<<"Program started"<<endl;
  // Read the configuration file
  CameraParameters camera_parameters = read_camera_parameters(config_path);
  ArduinoParameters arduino_parameters = read_arduino_parameters(config_path);

  cout<<"FPS: "<<arduino_parameters.fps<<endl;

  ImageParameters image_parameters = {
    camera_parameters.height,
    camera_parameters.width,
    camera_parameters.x0,
    camera_parameters.y0
    };

  const string output_dir = (string)argv[1] + "/";
  const int num_frames = arduino_parameters.fps * arduino_parameters.record_time;
  arduino_parameters.record_time += 1000; //for Arduino on purpose longer than record time

  // Parameters
  const int num_saver_threads = 16;
  const int image_count = 10; //number of images that can be stored in the ring buffer
  const double acquire_timeout_s = 1.0; // should be seconds from Pandas API but is not
  const DataFormat format = DataFormat::Mono16;


  cout<<"### Camera ###" <<endl;
  cout<<"Init cam object"<<endl;
  PCO_InitializeLib();
  Camera cam = Camera();
  cout << "Done initializing the camera object" <<endl;
  try{
    setupCamera(cam, camera_parameters);
  } catch (const exception& e) {
    return 1;
  }

  cout<<"### Arduino ###" <<endl;
  cout<<"Oppening connection with the Arduino"<<endl;
  io_service io;
  serial_port port(io);
  port.open("/dev/arduino");
  port.set_option(serial_port_base::baud_rate(9600));
  setupArduino(io, port, arduino_parameters);
  cout<<"### Done ###" <<endl;

  prepare_output_dir(output_dir);

  cout<<"------ Record ------"<<endl;
  
  //Initialize the threads
  thread acquirer_thread(image_acquirer, std::ref(cam), std::ref(port), std::ref(io),
                         num_frames, live_stream_int,
                         image_parameters.width, image_parameters.height,
                         image_count, format, acquire_timeout_s);

  vector<thread> saver_threads;
  for (int i = 0; i < num_saver_threads; ++i) {
    //thread saver_thread(image_saver, output_dir);
    //saver_threads.push_back(move(saver_thread));
    saver_threads.push_back(thread(image_saver, output_dir));
  }
  thread live_display_thread(live_display, image_parameters);

  // image_acquirer(cam, port, io, num_frames, live_stream_int,
  //                 image_parameters.width, image_parameters.height,
  //                 image_count, format, acquire_timeout_s);

  // Wait for threads to finish
  acquirer_thread.join();
  cout << "acquirer thread joined" << endl;

  for (auto &saver_thread : saver_threads) {
    saver_thread.join();
    cout << "saver thread joined" << endl;
  }

  live_display_thread.join();
  cout << "streamer thread joined" << endl;
  cam.stop();

  cout << "All done." << endl;
  return 0;
  }