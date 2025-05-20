/*
TODO:
1. Refactor config file; move magic numbers there
2. Refactor two translation stages as a single class
3. Get output directory from user input
*/

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <mutex>
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

#include "utils/utils_serial.h"
#include "utils/utils_camera.h"
#include "utils/utils_tracking.h"

using namespace std;
using namespace pco;
using namespace boost::asio;
using namespace std::chrono_literals;
namespace motion = zaber::motion;


// ========== Global variables ==========
// Synchronization primatives
mutex recording_kin_mtx;
mutex recording_muscle_mtx;
mutex live_mtx;
condition_variable recording_kin_condvar;
condition_variable recording_muscle_condvar;
condition_variable live_condvar;
atomic<bool> is_finished(false);
atomic<bool> latest_is_kin(false);

const int live_stream_int = 3;
const int n_frames_fps_comp = 300;

// Data containers
const int num_muscle_channels = 4;
const int num_kin_channels = 3;
queue<ImageDataVec> image_queue_kin;
queue<ImageDataVec> image_queue_muscle;
cv::Mat live_image;
bool live_image_type;
bool new_live_image = false;
int live_image_frame_id = 0;
float live_image_stage_pos = 0;

//Hardware
unique_ptr<motion::ascii::Axis> main_axis;

//Fly detection
const bool debug_tracking = false;

//Paths
std::string calibration_path = "/home/nely/flymuscle-control/config/arena_limits.yaml";
std::string config_path = "/home/nely/flymuscle-control/config/config.yaml";

// tracking parameters 
TrackingParameters tracking_parameters = {
  .low_speed_dist = 200,
  .max_speed_dist = 600,
  .stop_zone = 50,
  .max_tracking_speed = 20,
  .max_tracking_accel = 200,
  .min_tracking_speed = 1,
  .min_tracking_accel = 10
};
int cropping_margin = 10;

void image_acquirer(Camera& cam, serial_port& port, io_service& io,
                    int num_frames, int live_stream_int, size_t image_width,
                    size_t image_height, int image_count,
                    DataFormat format, double timeout_s, float img_scale) {
  
  bool is_kin = false;
  long long timestamp = 0;
  double stage_pos = 0;

  try{
    Image img;

    cam.record(image_count, RecordMode::ring_buffer);
    // Last handshake
    cout << "Num frames " << to_string(num_frames) <<endl; 
    
    // Launch the triggering here
    ImageDataVec image_data_muscle(num_muscle_channels);
    ImageDataVec image_data_kin(num_kin_channels);

    waitForArduinoReady(io, port, "Ready");
    send_command(port, "launch");
    
    cam.waitForFirstImage(true, timeout_s);
    is_kin = latest_is_kin.load(memory_order_acquire);
    timestamp = getHighResolutionTimestamp();
    cout << "Recorded the first image: Entering the main loop" << endl;
    stage_pos = main_axis->getPosition(motion::Units::LENGTH_MILLIMETRES);
    cam.image(img, PCO_RECORDER_LATEST_IMAGE, format);
    cv::Mat frame(image_height, image_width, CV_16UC1, img.raw_data().first);
    
    if (is_kin){
      image_data_kin.addData(
        0,
        timestamp,
        frame.clone(),
        is_kin,
        stage_pos
      );
    }else{
      image_data_muscle.addData(
        0,
        timestamp,
        frame.clone(),
        is_kin,
        stage_pos
      );
    }

    int muscle_queue_size = 0;
    int kin_queue_size = 0;

    long long last_checkpoint_time = timestamp;

    for (int frame_id = 1; frame_id < num_frames; ++frame_id) {
      // Check real FPS
      if (frame_id % n_frames_fps_comp  == 0) {
        double real_fps = n_frames_fps_comp  / (timestamp - last_checkpoint_time)*1e6;
        last_checkpoint_time = timestamp;
        printf("  Real FPS (frames %d-%d): %f\n", frame_id - n_frames_fps_comp , frame_id - 1,
              real_fps);
      }

      cam.waitForNewImage(true, timeout_s);
      is_kin = latest_is_kin.load(memory_order_acquire);
      timestamp = getHighResolutionTimestamp();
      stage_pos = main_axis->getPosition(motion::Units::LENGTH_MILLIMETRES);
      cam.image(img, PCO_RECORDER_LATEST_IMAGE, format);
      cv::Mat frame(image_height, image_width, CV_16UC1, img.raw_data().first);
      
      if (is_kin){
        image_data_kin.addData(
          frame_id,
          timestamp,
          frame.clone(),
          is_kin,
          stage_pos
        );
         // Check if the image data is full
        if (image_data_kin.isFull()) {
          unique_lock<mutex> ul(recording_kin_mtx);
          image_queue_kin.push(image_data_kin);
          kin_queue_size = image_queue_kin.size();
          ul.unlock();
          recording_kin_condvar.notify_one();
        }
      }else{
        image_data_muscle.addData(
          frame_id,
          timestamp,
          frame.clone(),
          is_kin,
          stage_pos
        );
        if (image_data_muscle.isFull()) {
          unique_lock<mutex> ul(recording_muscle_mtx);
          image_queue_muscle.push(image_data_muscle);
          muscle_queue_size = image_queue_muscle.size();
          ul.unlock();
          recording_muscle_condvar.notify_one();
        }
      }

      if (muscle_queue_size > 40) {
        cout << "Quitting image acquirer thread because muscle queue is too big" << endl;
        cout << "Muscle queue size: " << muscle_queue_size << endl;
        break;
      }
      if (kin_queue_size > 40) {
        cout << "Quitting image acquirer thread because kin queue is too big" << endl;
        cout << "Kin queue size: " << kin_queue_size << endl;
        break;
      }

      // Update image for closed-loop control and live streaming
      if (frame_id % live_stream_int == 0) {
          unique_lock<mutex> ul(live_mtx);
          live_image = frame.clone();
          live_image_type = is_kin;
          new_live_image = true;
          live_image_stage_pos = stage_pos;
          live_image_frame_id = frame_id;
          ul.unlock();
          live_condvar.notify_all();
      }
      if (is_finished.load()) {
        break;
      }
    }
    stopArduino(port);
    is_finished.store(true);
    recording_kin_condvar.notify_all();
    recording_muscle_condvar.notify_all();
    live_condvar.notify_all();
  }catch (CameraException& err) {
      cout << "Error Code: " << (uint)err.error_code() << endl;
      cout << err.what() << endl;
      stopArduino(port);
      is_finished.store(true);
      recording_kin_condvar.notify_all();
      recording_muscle_condvar.notify_all();
      live_condvar.notify_all();
  }
  cout << "Image acquirer thread exiting" << endl;
}

void muscle_img_saver(const string output_dir){
  while (true) {

    ImageDataVec image_data_muscle(num_muscle_channels);
    cv::Mat merged_image;
    // Get an image from the queue
    unique_lock<mutex> ul(recording_muscle_mtx);
    recording_muscle_condvar.wait(
        ul, []() { return !image_queue_muscle.empty() || (is_finished.load() && image_queue_muscle.empty()); });

    if (is_finished.load() && image_queue_muscle.empty()) {
      ul.unlock();
      break;
    } else {
      image_data_muscle = move(image_queue_muscle.front());
      image_queue_muscle.pop();
      ul.unlock();
    }
    // Write image to output_dir
    string img_filename = "/mnt/tmpfs/" +
    to_string(image_data_muscle.getFrameIds()[0]) + ".tif";
    cv::merge(image_data_muscle.getImages(), merged_image);
    save_tiff_lib(img_filename, merged_image);

    if (image_data_muscle.getFrameIds()[0] == 0){
      //write stage pos
      cout << "Writing first muscle frame" << endl;
    }

    string metadata_filename = output_dir +
    to_string(image_data_muscle.getFrameIds()[0]) + "_metadata.csv";
    ofstream metadata_file(metadata_filename, ios::app);
    for (int i = 0; i < num_muscle_channels; ++i) {
      metadata_file << image_data_muscle.getFrameIds()[i] << ","
                    << image_data_muscle.getTimestamps()[i] << ","
                    << image_data_muscle.getIsKinematics()[i] << ","
                    << image_data_muscle.getStagePos()[i] << endl;
    }
    metadata_file.close();    
  }
  cout << "Muscle image saver thread exiting" << endl;
}

void kin_img_saver(const string output_dir){

  vector<int> jpeg_compression_params = {cv::IMWRITE_JPEG_QUALITY, 100};

  while (true) {

    ImageDataVec image_data_kin(num_kin_channels);
    cv::Mat merged_image;
    // Get an image from the queuemetadata_filename
    unique_lock<mutex> ul(recording_kin_mtx);
    recording_kin_condvar.wait(
        ul, []() { return !image_queue_kin.empty() || (is_finished.load() && image_queue_kin.empty()); });

    if (is_finished.load() && image_queue_kin.empty()) {
      ul.unlock();
      break;
    } else {
      image_data_kin = move(image_queue_kin.front());
      image_queue_kin.pop();
      ul.unlock();
    }
    // Write image to output_dir
    string filename_stem = output_dir +
      to_string(image_data_kin.getFrameIds()[0]);
    cv::merge(image_data_kin.getImages(), merged_image);
    cv::imwrite(filename_stem + ".jpg", merged_image, jpeg_compression_params);

    if (image_data_kin.getFrameIds()[0] == 0){
      //write stage pos
      cout << "Writing first kinematic frame" << endl;
    }

    ofstream metadata_file(filename_stem + "metadata.csv", ios::app);
    for (int i = 0; i < num_kin_channels; ++i) {
      metadata_file << image_data_kin.getFrameIds()[i] << ","
                    << image_data_kin.getTimestamps()[i] << ","
                    << image_data_kin.getIsKinematics()[i] << ","
                    << image_data_kin.getStagePos()[i] << endl;
    }
    metadata_file.close();    
  }
  cout << "Kinematic image saver thread exiting" << endl;
}

void closed_loop_controller(CalibrationParameters calibration_params, TrackingParameters tracking_params, int crop_margin) {
  
  // Create two windows with different names
  cv::namedWindow("Behavior frames", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("Muscle frames", cv::WINDOW_AUTOSIZE);
  // Move windows to specific positions to show them side by side
  cv::moveWindow("Behavior frames", 100, 100);   // Move "Window 1" to position (0, 100)
  cv::moveWindow("Muscle frames", calibration_params.img_width*calibration_params.img_scale+100, 100); // Move "Window 2" to position (350, 100)


  FlyDetector fly_detector = FlyDetector(calibration_params, cropping_margin, 0.0);

  TruncatedLinearRegression speed_tlr(tracking_params.low_speed_dist,
                                      tracking_params.min_tracking_speed,
                                      tracking_params.max_speed_dist,
                                      tracking_params.max_tracking_speed);
  TruncatedLinearRegression accel_tlr(tracking_params.low_speed_dist,
                                      tracking_params.min_tracking_accel,
                                      tracking_params.max_speed_dist,
                                      tracking_params.max_tracking_accel);

  calibration_params.left_limit += cropping_margin;
  calibration_params.right_limit -= cropping_margin;

  while (true) {
    unique_lock<mutex> ul(live_mtx);
    live_condvar.wait_for(ul, std::chrono::milliseconds(1000), []() { 
      return (!live_image.empty() && new_live_image) || is_finished.load(); 
    });
    cv::Mat my_live_image = live_image.clone()*50.0;
    bool my_live_image_type = live_image_type;
    int my_live_image_frameid = live_image_frame_id;
    float my_live_image_stage_pos = live_image_stage_pos;
    new_live_image = false;
    ul.unlock();

    // Update image for closed-loop control and live streaming
    cv::resize(my_live_image, my_live_image, cv::Size(), calibration_params.img_scale, calibration_params.img_scale, cv::INTER_NEAREST);

    cv::Mat my_live_image_uint8;
    my_live_image.convertTo(my_live_image_uint8, CV_8U, 0.00390625);

    if (my_live_image_type){
      fly_detector.update_edge_positions(my_live_image_stage_pos);
      int fly_pos = fly_detector.detect_fly_pos(my_live_image_uint8);

      if (my_live_image_stage_pos < calibration_params.home_pos){
        main_axis->moveVelocity(20, motion::Units::VELOCITY_MILLIMETRES_PER_SECOND, 100, motion::Units::ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED);
      }
      else if (my_live_image_stage_pos > calibration_params.far_pos){
        main_axis->moveVelocity(-20, motion::Units::VELOCITY_MILLIMETRES_PER_SECOND, 100, motion::Units::ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED);
      }
      else{
        if (fly_pos > 0){
          float fly_offset = fly_pos - calibration_params.img_height*calibration_params.img_scale/2;
          float abs_fly_offset = fabsf(fly_offset);
          float tracking_speed = speed_tlr.predict(abs_fly_offset);
          float tracking_accel = accel_tlr.predict(abs_fly_offset);
          if (fly_offset < -tracking_params.stop_zone){
            main_axis->moveVelocity(-tracking_speed, motion::Units::VELOCITY_MILLIMETRES_PER_SECOND, tracking_accel, motion::Units::ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED);
          }else if (fly_offset > tracking_params.stop_zone){
            main_axis->moveVelocity(tracking_speed, motion::Units::VELOCITY_MILLIMETRES_PER_SECOND, tracking_accel, motion::Units::ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED);
          }
          else{
            main_axis->stop(false);
          }
        }
        else{
          if (!main_axis->isBusy()){
            //If last state was stop and the fly magically disapeared
            // (very unlikely) intitiate the sweep movement
            main_axis->moveVelocity(-10, motion::Units::VELOCITY_MILLIMETRES_PER_SECOND, 100.0, motion::Units::ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED);
          }
        }
      }
      cv::Mat display_image = fly_detector.prepare_display_image(my_live_image_uint8);
      if (fly_pos>0){
        draw_horizontal_line(display_image, fly_pos, 50, calibration_params.img_height*calibration_params.img_scale, cv::Scalar(255, 255, 255));
      }
      cv::imshow("Behavior frames", display_image);
    }
    else{
      cv::imshow("Muscle frames", my_live_image_uint8);
    }
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

void kinematicReader(serial_port& port) {
  unsigned char latest_byte;
  while (!is_finished.load()) { 
      if (read(port, buffer(&latest_byte, 1)) > 0) {
        latest_is_kin.store(latest_byte == 0x01, std::memory_order_release);
      }
      std::this_thread::sleep_for(std::chrono::microseconds(10));  // Avoid busy-waiting
  }
  cout << "Kinematic reader thread exiting" << endl;
}

int main(int argc, char **argv) {
  cout<<"Program started"<<endl;
  // Read the configuration file
  CalibrationParameters calibration_parameters = load_calibration_parameters(calibration_path);
  CameraParameters camera_parameters = read_camera_parameters(config_path);
  ArduinoParameters arduino_parameters = read_arduino_parameters(config_path);

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
  const int num_muscle_saver_threads = 16;
  const int num_kin_saver_threads = 4;
  const int image_count = 10; //number of images that can be stored in the ring buffer
  const double acquire_timeout_s = 1.0; // should be seconds from Pandas API but is not
  const DataFormat format = DataFormat::Mono16;

  cout<<"----- Initialization -----" <<endl;

  cout<<"### Zaber translating stage ###" <<endl;
  cout<<"Opening connection with the Zaber translating stage"<<endl;
  motion::ascii::Connection connection =
        motion::ascii::Connection::openSerialPort("/dev/zaber_device");
  setupZaber(connection, main_axis);
  cout<<"### Done ###" <<endl;
  
  cout<<"### Camera ###" <<endl;
  cout<<"Init cam object"<<endl;
  Camera cam = Camera();

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

  cout<<"----- Done -----" <<endl;
  
  cout<<"----- Recording -----" <<endl;
  cout << "Writing data to " + output_dir << endl;

  //Initialize the threads
  thread is_kinematic_thread(kinematicReader, std::ref(port));

  vector<thread> muscle_saver_threads;
  for (int i = 0; i < num_muscle_saver_threads; ++i) {
    muscle_saver_threads.push_back(thread(muscle_img_saver, output_dir));
  }
  vector<thread> kin_saver_threads;
  for (int i = 0; i < num_kin_saver_threads; ++i) {
    kin_saver_threads.push_back(thread(kin_img_saver, output_dir));
  }

  thread control_thread(closed_loop_controller,
    calibration_parameters,
    tracking_parameters,
    cropping_margin);

  thread acquirer_thread(image_acquirer, std::ref(cam), std::ref(port), std::ref(io),
    num_frames, live_stream_int,
    image_parameters.width, image_parameters.height,
    image_count, format, acquire_timeout_s, calibration_parameters.img_scale);

  // Wait for threads to finish
  acquirer_thread.join();
  cout << "acquirer thread joined" << endl;

  is_kinematic_thread.join();
  cout << "is_kinematic thread joined" << endl;

  for (auto &saver_thread : muscle_saver_threads) {
    saver_thread.join();
    cout << "muscle saver thread joined" << endl;
  }
  for (auto &saver_thread : kin_saver_threads) {
    saver_thread.join();
    cout << "kin saver thread joined" << endl;
  }
  cout << "saver thread joined" << endl;
  
  control_thread.join();
  cout << "control thread joined" << endl;

  cout<<"### Camera ###" <<endl;
  cam.stop();
  cout<<"### Done ###" <<endl;
  cout<<"### Arduino ###" <<endl;
  cout<<"Closing connection with the Arduino"<<endl;
  port.close();
  cout<<"### Done ###" <<endl;
  cout<<"### Zaber translating stage ###" <<endl;
  cout<<"Closing connection with the Zaber translating stage"<<endl;
  main_axis->home(true);
  main_axis->stop(true);
  connection.close();
  cout<<"### Done ###" <<endl;
  cout<<"### Camera ###" <<endl;
  cout<<"Closing connection with the camera"<<endl;

  //moving files from tmpfs to the output directory
  string command = "mv /mnt/tmpfs/* " + output_dir;
  int result = system(command.c_str());
  if (result == 0) {
    cout << "Files moved successfully." << endl;
  } else {
    cout << "Error moving files." << endl;
  }

  cout<<"----- Done -----" <<endl;

  return 0;
}