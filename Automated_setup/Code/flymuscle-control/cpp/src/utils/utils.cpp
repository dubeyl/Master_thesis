#include "utils.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

using namespace std;

//file
void prepare_output_dir(string directory) {
  if (filesystem::exists(directory)) {
    // Remove all files and subdirectories from the directory
    for (auto &entry : filesystem::directory_iterator(directory)) {
      if (entry.is_directory()) {
        filesystem::remove_all(entry.path());
      } else {
        filesystem::remove(entry.path());
      }
    }
  } else {
    // Create the directory
    error_code err;
    mkdir_recursive(directory, err);
    if (err) {
      cout << "Error creating directory: " << err.message() << endl;
    }
  }
}

bool mkdir_recursive(string const &dirName, std::error_code &err) {
  err.clear();
  if (!filesystem::create_directories(dirName, err)) {
    if (filesystem::exists(dirName)) {
      // The folder already exists:
      err.clear();
      return true;
    }
    return false;
  }
  return true;
}

CameraParameters read_camera_parameters(string& yaml_path){
  std::ifstream params_file(yaml_path);
  YAML::Node yaml = YAML::Load(params_file);
  CameraParameters camera_parameters;
  vector<string> resolution = split(yaml["camera"]["resolution"].as<string>(), ' ');
  vector<string> offset = split(yaml["camera"]["offset"].as<string>(), ' ');
  camera_parameters.x0 = stoi(offset[0]);
  camera_parameters.y0 = stoi(offset[1]);
  camera_parameters.width = stoi(resolution[0]);
  camera_parameters.height = stoi(resolution[1]);

  camera_parameters.exposure_time = yaml["camera"]["exposure_time"].as<double>();
  camera_parameters.external_trigger = true;

  return camera_parameters;
}

ArduinoParameters read_arduino_parameters(string& yaml_path){
  std::ifstream params_file(yaml_path);
  YAML::Node yaml = YAML::Load(params_file);
  ArduinoParameters arduino_parameters;

  arduino_parameters.alternating = yaml["arduino"]["alternating"].as<bool>();
  arduino_parameters.record_time = yaml["arduino"]["duration"].as<int>();
  arduino_parameters.optogenetics = yaml["arduino"]["optogenetics"].as<bool>();
  arduino_parameters.off1 = yaml["arduino"]["off1"].as<double>();
  arduino_parameters.off2 = yaml["arduino"]["off2"].as<double>();
  arduino_parameters.on1 = yaml["arduino"]["on1"].as<double>();
  arduino_parameters.stimnum = yaml["arduino"]["stimnum"].as<int>();
  arduino_parameters.fps = yaml["arduino"]["framerate"].as<int>();
  
  return arduino_parameters;
}

//time
double get_current_time() {
  auto now = chrono::high_resolution_clock::now();
  auto dur_since_epoch = now.time_since_epoch();
  double seconds = chrono::duration<double>(dur_since_epoch).count();
  // auto milliseconds = chrono::time_point_cast<chrono::milliseconds>(now);
  return seconds;
}

long long getHighResolutionTimestamp() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

//miscellaneous
void draw_vertical_line(cv::Mat image, int position, int length, int img_width,
                        cv::Scalar color) {
  if ((0 <= position) && (position <= img_width)) {
    cv::Point line_start(position, 0);
    cv::Point line_end(position, length);
    cv::line(image, line_start, line_end, color);
  }
}

void draw_horizontal_line(cv::Mat image, int position, int length, int img_height,
                        cv::Scalar color) {
  if ((0 <= position) && (position <= img_height)) {
    cv::Point line_start(0, position);
    cv::Point line_end(length, position);
    cv::line(image, line_start, line_end, color);
  }
}

int clipToMultipleOf32(int value) {
  int remainder = value % 32;
  if (remainder > 16) {
    return value + (32 - remainder);
  } else {
    return value - remainder;
  }
}


vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> result;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    result.push_back(item);
  }
  return result;
}


void save_tiff_lib(const std::string& filename, const cv::Mat& image) {
  if (image.empty()) {
      std::cerr << "Error: Image is empty!" << std::endl;
      return;
  }

  // Ensure it's a 4-channel image
  if (image.channels() != 4) {
      std::cerr << "Error: Image does not have 4 channels!" << std::endl;
      return;
  }

  TIFF* tiff = TIFFOpen(filename.c_str(), "w");
  if (!tiff) {
      std::cerr << "Error: Could not open file " << filename << " for writing!" << std::endl;
      return;
  }

  int width = image.cols;
  int height = image.rows;
  int bitsPerSample = (image.depth() == CV_8U) ? 8 : 16; // Supports 8-bit or 16-bit images

  // Set TIFF fields
  TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, width);
  TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, height);
  TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, 4);  // 4 channels (RGBA or other)
  TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, bitsPerSample);
  TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB); // Can also use PHOTOMETRIC_SEPARATED
  TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG); // Store channels together
  TIFFSetField(tiff, TIFFTAG_COMPRESSION, COMPRESSION_LZW);  // No compression for speed

  // Allocate buffer for a single row (to optimize memory usage)
  std::vector<uint16_t> buffer(width * 4);

  for (int row = 0; row < height; row++) {
      const uint16_t* src = image.ptr<uint16_t>(row);
      std::memcpy(buffer.data(), src, width * 4 * sizeof(uint16_t));  // Copy row data
      TIFFWriteScanline(tiff, buffer.data(), row, 0);
  }

  TIFFClose(tiff);
}