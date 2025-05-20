#ifndef UTILS
#define UTILS

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <tiffio.h>


using namespace std;

struct ImageParameters {
  int height;
  int width;
  int x0;
  int y0;
};

struct CameraParameters {
  double exposure_time;
  int x0;
  int y0;
  int width;
  int height;
  bool external_trigger;
};

struct ArduinoParameters {
  bool alternating;
  int record_time;
  bool optogenetics;
  int off1;
  int off2;
  int on1;
  int stimnum;
  int fps;
};

// file
void prepare_output_dir(string directory);
bool mkdir_recursive(string const &dirName, std::error_code &err);
CameraParameters read_camera_parameters(string& yaml_path);
ArduinoParameters read_arduino_parameters(string& yaml_path);

//time
double get_current_time();
long long getHighResolutionTimestamp();

//miscellaneous
void draw_vertical_line(cv::Mat image, int position, int length, int img_width,
                        cv::Scalar color);
void draw_horizontal_line(cv::Mat image, int position, int length, int img_height,
                        cv::Scalar color);
int clipToMultipleOf32(int value);

vector<std::string> split(const std::string &s, char delim);

void save_tiff_lib(const std::string& filename, const cv::Mat& image);

#endif