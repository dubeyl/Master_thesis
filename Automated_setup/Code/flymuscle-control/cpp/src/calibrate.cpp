#include <chrono>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "opencv2/imgcodecs.hpp"
#include "stdafx.h"
#include "camera.h"
#include "cameraexception.h"
#include <stdio.h>
#include <string.h>

#include <thread>
#include <zaber/motion/ascii.h>


#include "utils/utils.h"
#include "utils/utils_serial.h"
#include "utils/utils_camera.h"

namespace motion = zaber::motion;

float img_scale = 0.25;

// Hardware
std::unique_ptr<motion::ascii::Axis> main_axis;

//Paths
std::string limits_path = "/home/nely/flymuscle-control/config/arena_limits.yaml";
std::string camera_params_path = "/home/nely/flymuscle-control/config/config.yaml";

int main() {

  motion::ascii::Connection connection =
        motion::ascii::Connection::openSerialPort("/dev/zaber_device");
  setupZaber(connection, main_axis);

  CameraParameters camera_parameters = read_camera_parameters(camera_params_path);
  camera_parameters.exposure_time = 0.003;
  camera_parameters.external_trigger = false;
  

  ImageParameters image_parameters = {
    camera_parameters.height,
    camera_parameters.width,
    camera_parameters.x0,
    camera_parameters.y0
    };
  pco::Camera cam = pco::Camera();

  try{
    setupCamera(cam, camera_parameters);
  } catch (pco::CameraException &err) {
    cam.stop();
    std::cout << "Error Code: " << (uint)err.error_code() << std::endl;
    std::cout << err.what() << std::endl;
  }

  double acquire_timeout_s = 1.0;
  pco::DataFormat format = pco::DataFormat::Mono8;
  pco::Image img;

  int calibration_step = 0;
  bool started_step = true;

  int n_extreme_pos = 0;

  int line_pos = 0;
  int n_lines_enterred = 0;
  int move_amount = 10;

  std::ofstream fout(limits_path);
  YAML::Emitter out(fout);
  out << YAML::BeginMap;
  //add width and height to yaml file
  out << YAML::Key << "Width" << YAML::Value << image_parameters.width;
  out << YAML::Key << "Height" << YAML::Value << image_parameters.height;

  vector<int> corridor_sides;
  vector<string> corridor_sides_names = {"LeftLimit", "RightLimit"};
  vector<float> corridor_ends;
  vector<string> corridor_ends_names = {"HomePos", "HomeDisappeared", "FarPos", "FarDisappeared"};

  int key = -1;

  try{
    std::string window_name = "Calibration";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    cam.record(10, pco::RecordMode::ring_buffer);
    std::cout << "Entering the main loop" << std::endl;

    cam.waitForFirstImage(false, acquire_timeout_s);
    cout << "First image aquired" << endl;
    while (true) {  
      cam.waitForNewImage(false, acquire_timeout_s);
      cam.image(img, PCO_RECORDER_LATEST_IMAGE, format);
      cv::Mat frame(image_parameters.height, image_parameters.width, CV_16UC1, img.raw_data().first);
      cv::Mat frame_small;
      cv::resize(frame, frame_small, cv::Size(), img_scale, img_scale, cv::INTER_NEAREST);
      cv::cvtColor(frame_small, frame_small, cv::COLOR_GRAY2BGR);
      switch (calibration_step){
        case 0:
          {
            if (started_step) {
              std::cout<<"Press enter when the line is aligned with:" <<std::endl;
              std::cout<<"- The home side of the arena (Right)"<<std::endl;
              std::cout<<"- When the home side just disappears from the right of the image"<<std::endl;
              std::cout<<"- The far side of the arena (Left)"<<std::endl;
              std::cout<<"- When the far side just disappears from the left side of the image"<<std::endl;
              started_step = false;
            }
            if (key == 13){
              float stage_pos = main_axis->getPosition(motion::Units::LENGTH_MILLIMETRES);
              if ((n_extreme_pos == 3) && (stage_pos > corridor_ends[n_extreme_pos-1])){
                std::cout<<"The far side disappear should be toward the inside of the corridor. Please try again."<<std::endl;
                break;
              }
              n_extreme_pos += 1;
              corridor_ends.push_back(stage_pos);
              std::cout<<corridor_ends_names[n_extreme_pos-1]<<": "<<stage_pos<<"mm"<<std::endl;
              if (n_extreme_pos == 4){
                  cout << "Done with vertical edges calibration going to horizontal edges" << endl;
                  calibration_step = 1;
                  started_step = true;
              }
            }
            draw_horizontal_line(frame_small, image_parameters.height*img_scale/2, image_parameters.width*img_scale,
                            image_parameters.height*img_scale, cv::Scalar(255, 0, 0)); //divide by four because of resize
            break;
          }
        case 1:
          {
            if (started_step) {
                std::cout<<"Use the up and down arrow to move the horizontal line"<<std::endl;
                std::cout<<"Press enter when the line is aligned with:"<<std::endl;
                std::cout<<"- The top of the corrdior"<<std::endl;
                std::cout<<"- The bottom of the corridor"<<std::endl;
                started_step = false;
            }
            switch (key){
              case 81:
                //left arrow
                line_pos = max(0, line_pos - move_amount);
                break; 
              case 83:
                //right arrow
                line_pos = min(image_parameters.width, line_pos + move_amount);
                break; 
              case 13:
                // User pressed enter
                corridor_sides.push_back(line_pos);
                n_lines_enterred += 1;
                std::cout<<corridor_sides_names[n_lines_enterred-1]<<": "<<line_pos<<"px"<<std::endl;
                if (n_lines_enterred == 2){
                  cout << "Done with horizontal edges calibration" << endl;
                  calibration_step = 2;
                  started_step = true;
                }
                break;
            }
            draw_vertical_line(frame_small, line_pos, image_parameters.height*img_scale, image_parameters.width*img_scale, cv::Scalar(255, 0, 0));
          break;
        }
      }
      cv::imshow(window_name, frame_small*100.0);
      key = cv::waitKey(1);
      if (key == 27) {
        break;
      }
      //if press backspace, restart current step
      if (key == 8){
        started_step = true;
        switch (calibration_step){
          case 1:
            //reset variables
            corridor_ends.clear();
            n_extreme_pos = 0;
            break;
          case 2:
            corridor_sides.clear();
            n_lines_enterred = 0;
            break;
        }
      }
      if (calibration_step == 2) {
        break;
      }
    }
    cam.stop();
  } catch (pco::CameraException &err) {
    cam.stop();
    std::cout << "Error Code: " << (uint)err.error_code() << std::endl;
    std::cout << err.what() << std::endl;
  }

  // Save limits
  for (int i = 0; i < 4; i++){
    out << YAML::Key << corridor_ends_names[i] << YAML::Value << corridor_ends[i];
  }
  for (int i = 0; i < 2; i++){
    out << YAML::Key << corridor_sides_names[i] << YAML::Value << corridor_sides[i];
  }
  out << YAML::Key << "ImgScale" << YAML::Value << img_scale;
  out << YAML::EndMap;
  fout.close();

  //return to home position
  std::cout<<"Returning to home position"<<std::endl;
  main_axis->home();
  connection.close();

  cv::destroyAllWindows();

  std::cout << "All done." << std::endl;
  return 0;
}
