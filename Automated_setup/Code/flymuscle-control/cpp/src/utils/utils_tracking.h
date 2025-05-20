#ifndef UTILS_TRACKING_H
#define UTILS_TRACKING_H

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include "utils.h"

struct CalibrationParameters {
  int img_width;
  int img_height;
  float home_pos;
  float home_disappeared;
  float far_pos;
  float far_disappeared;
  int right_limit;
  int left_limit;
  float img_scale;
};

struct ArenaLimitsPos {
  int home_edge;
  int far_edge;
};

struct TrackingParameters {
  const int low_speed_dist; //within 30 pixels the stage will use low velocity low accel for tracking
  const int max_speed_dist; //beyond 100pxls the stage will move at max speed to track
  const int stop_zone; //pxls
  const float max_tracking_speed; // / mm/s
  const float max_tracking_accel; // mm/s²
  const float min_tracking_speed; // / mm/s
  const float min_tracking_accel; // mm/s²
};

//Classes
class TruncatedLinearRegression{
public:
    // Constructor
    TruncatedLinearRegression(const float x0, const float y0_in,
                              const float x1, const float y1_in,
                              const float max_y_in, const float min_y_in);
    TruncatedLinearRegression(const float x0, const float y0_in,
                              const float x1, const float y1_in);
    // Function to predict y for a given x
    float predict(float x);
    void set_slope(float x0, float y0, float x1, float y1);
    void set_intercept(float x0, float y0, float x1, float y1);

private:
    float slope;
    float intercept;
    float max_y;
    float min_y;
};

class FlyDetector{
public:
    FlyDetector(CalibrationParameters calibration_params_in, int crop_margin_in, float stage_pos_in);
    int detect_fly_pos(cv::Mat& image);
    void update_edge_positions(float stage_pos);
    const int crop_margin;
    ArenaLimitsPos arena_limits;
    cv::Mat prepare_display_image(cv::Mat& image);

private:
    CalibrationParameters calibration_params;
    TruncatedLinearRegression home_edge_reg;
    TruncatedLinearRegression far_edge_reg;
};

//Functions
CalibrationParameters load_calibration_parameters(std::string calibration_path);

#endif // Include guard ends