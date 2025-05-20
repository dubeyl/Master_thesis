#include "utils_tracking.h"

using namespace std;

CalibrationParameters load_calibration_parameters(std::string calibration_path){
    // Load variables from YAML file
    YAML::Node config = YAML::LoadFile(calibration_path);
    const int calibration_height = config["Height"].as<int>();
    const int calibration_width = config["Width"].as<int>();
    const int ventral_view_left = config["LeftLimit"].as<int>();
    const int ventral_view_right = config["RightLimit"].as<int>();
    const float home_pos = config["HomePos"].as<float>(); //position of the translating stage when image center is aligned with the home end
    const float far_pos = config["FarPos"].as<float>();
    const float home_disappeared = config["HomeDisappeared"].as<float>();
    const float far_disappeared = config["FarDisappeared"].as<float>();
    const float img_scale = config["ImgScale"].as<float>();

    CalibrationParameters calibration_tracking_parameters = {
        .img_width = calibration_width,
        .img_height = calibration_height,
        .home_pos = home_pos,
        .home_disappeared = home_disappeared,
        .far_pos = far_pos,
        .far_disappeared = far_disappeared,
        .right_limit = ventral_view_right,
        .left_limit = ventral_view_left,
        .img_scale = img_scale
    };
    
    return calibration_tracking_parameters;
}

// Classes
TruncatedLinearRegression::TruncatedLinearRegression(const float x0, const float y0_in,
                                                      const float x1, const float y1_in,
                                                      const float max_y_in, const float min_y_in) {
    max_y = max_y_in;
    min_y = min_y_in;
    set_slope(x0, y0_in, x1, y1_in);
    set_intercept(x0, y0_in, x1, y1_in);
}

TruncatedLinearRegression::TruncatedLinearRegression(const float x0, const float y0_in,
                                                      const float x1, const float y1_in) {
    
    min_y = y0_in;
    max_y = y1_in;
    set_slope(x0, y0_in, x1, y1_in);
    set_intercept(x0, y0_in, x1, y1_in);
}

void TruncatedLinearRegression::set_slope(float x0, float y0, float x1, float y1){
    slope = (y1 - y0) / (x1 - x0);
}
void TruncatedLinearRegression::set_intercept(float x0, float y0, float x1, float y1){
    intercept = y0 - slope * x0;
}

// Function to predict y for a given x
float TruncatedLinearRegression::predict(float x) {
    float out =  slope * x + intercept;
    out = max(out, min_y);
    out = min(out, max_y);
    return out;
}

FlyDetector::FlyDetector(CalibrationParameters calibration_params_in, int crop_margin_in, float stage_pos_in)
    : calibration_params(calibration_params_in),
      home_edge_reg(calibration_params_in.home_pos, calibration_params_in.img_height*calibration_params.img_scale/2,
                    calibration_params_in.home_disappeared, 0,
                    calibration_params_in.img_height*calibration_params.img_scale, 0),
      far_edge_reg(calibration_params_in.far_pos, calibration_params_in.img_height*calibration_params.img_scale/2,
                   calibration_params_in.far_disappeared, calibration_params_in.img_height*calibration_params.img_scale,
                   calibration_params_in.img_height*calibration_params.img_scale, 0),
      crop_margin(crop_margin_in),
      arena_limits({0, 0})
{
    update_edge_positions(stage_pos_in);
}


void FlyDetector::update_edge_positions(float stage_pos) {
    arena_limits.home_edge = (int)home_edge_reg.predict(stage_pos);
    arena_limits.far_edge = (int)far_edge_reg.predict(stage_pos);
} 

int FlyDetector::detect_fly_pos(cv::Mat& image) {

    // Crop ventral view
    int y_max = min(int(calibration_params.img_height*calibration_params.img_scale), arena_limits.far_edge - crop_margin);
    int y_min = max(0, arena_limits.home_edge + crop_margin);
    if (y_max - y_min < 10) {
        // Something wrong happened; the view is too small. Fly not found.
        return -1;
    }
    int corridor_width = calibration_params.right_limit - calibration_params.left_limit;
    cv::Mat ventral_view = image(cv::Rect(
        calibration_params.left_limit, y_min,
        corridor_width, y_max - y_min));

    // Apply thresholding and morphological transform (for denoising)
    cv::Mat binary_image;
    cv::threshold(ventral_view, binary_image, 50, 255, cv::THRESH_BINARY);
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(40, 101), cv::Point(31, 31));
    cv::Mat opened_image;
    cv::morphologyEx(binary_image, opened_image, cv::MORPH_OPEN, kernel);

    // Find largest contour if one exists and is big enough
    vector<vector<cv::Point>> contours;
    cv::findContours(opened_image, contours, cv::RETR_EXTERNAL,
                    cv::CHAIN_APPROX_SIMPLE);
    float max_area = 0;
    vector<cv::Point> largest_contour;
    if (contours.size() > 0) {
        for (const auto &contour : contours) {
        float area = cv::contourArea(contour);
        if (area > max_area) {
            max_area = area;
            largest_contour = contour;
        }
        }
    }

    if (max_area >= 1000) {
        // Fly found
        cv::Moments moments = cv::moments(largest_contour);
        //float centroid_x = moments.m10 / moments.m00 + x_min;
        float centroid_y = moments.m01 / moments.m00 + y_min;
        return (int)centroid_y;
    } else {
        // Fly not found
        return -1;
    }
}

cv::Mat FlyDetector::prepare_display_image(cv::Mat& image) {
    cv::Mat display_image;
    cv::cvtColor(image, display_image, cv::COLOR_GRAY2BGR);
    draw_horizontal_line(display_image, arena_limits.home_edge, calibration_params.img_width*calibration_params.img_scale,
                        calibration_params.img_height*calibration_params.img_scale, cv::Scalar(255, 0, 0));
    draw_horizontal_line(display_image, arena_limits.far_edge, calibration_params.img_width*calibration_params.img_scale,
                        calibration_params.img_height*calibration_params.img_scale, cv::Scalar(0, 0, 255));
    draw_vertical_line(display_image, calibration_params.right_limit, calibration_params.img_height*calibration_params.img_scale,
                        calibration_params.img_width*calibration_params.img_scale, cv::Scalar(255, 255, 255));
    draw_vertical_line(display_image, calibration_params.left_limit, calibration_params.img_height*calibration_params.img_scale,
                        calibration_params.img_width*calibration_params.img_scale, cv::Scalar(0, 255, 0));
    return display_image;
}