#include "opencv2/imgcodecs.hpp"
#include "stdafx.h"
#include "defs.h"
#include "cameraexception.h"
#include "utils_camera.h"


using namespace std;
using namespace pco;

ImageDataVec::ImageDataVec(int num_channels)
    : frame_ids(num_channels),
      timestamps(num_channels),
      is_kinematics(num_channels, false),
      images(num_channels),
      stage_pos(num_channels),
      num_channels(num_channels),
      current_index(0),
      is_full(false) {}

// Accessor methods
int ImageDataVec::getNumChannels() const {
    return num_channels;
}

std::vector<int>& ImageDataVec::getFrameIds() {
    return frame_ids;
}

std::vector<long long>& ImageDataVec::getTimestamps() {
    return timestamps;
}

std::vector<bool>& ImageDataVec::getIsKinematics() {
    return is_kinematics;
}

std::vector<cv::Mat>& ImageDataVec::getImages() {
    return images;
}

std::vector<double>& ImageDataVec::getStagePos() {
    return stage_pos;
}

// Method to insert data in a circular manner
void ImageDataVec::addData(int frame_id, long long timestamp, cv::Mat image, bool is_kin , double stage_pos_value) {
    // Insert data at the current index
    frame_ids[current_index] = frame_id;
    timestamps[current_index] = timestamp;
    is_kinematics[current_index] = is_kin;  // Always set to true on update
    images[current_index] = image;  // Deep copy to prevent external modifications
    stage_pos[current_index] = stage_pos_value;

    // Move to the next index
    current_index++;

    // If we reached the end, wrap around and mark as full
    if (current_index >= num_channels) {
        current_index = 0;
        is_full = true;
    }else {
        is_full = false;  // Not full yet
    }
}

// Check if buffer is full
bool ImageDataVec::isFull() const {
    return is_full;
}

void setupCamera(Camera& cam, CameraParameters& camera_parameters){
    try{
    // Configure camera
    cout<<"Setting up camera configuration"<< endl;    
    cam.defaultConfiguration();
    Configuration config = cam.getConfiguration();
    
    config.roi.x0 = camera_parameters.x0 + 1;
    config.roi.y0 = camera_parameters.y0 + 1;
    config.roi.x1 = camera_parameters.x0 + camera_parameters.width;
    config.roi.y1 = camera_parameters.y0 + camera_parameters.height;

    if (camera_parameters.external_trigger){
        config.trigger_mode = TRIGGER_MODE_EXTERNALTRIGGER;
    }else{
        config.trigger_mode = TRIGGER_MODE_AUTOTRIGGER;
    }
    config.acquire_mode = ACQUIRE_MODE_AUTO;
    config.delay_time_s = 0;
    config.noise_filter_mode = NOISE_FILTER_MODE_ON;

    cam.setConfiguration(config);
    cout<<"Done."<<endl;

    cout<<"Set exposure time"<<endl;
    cam.setExposureTime(camera_parameters.exposure_time);


    cam.configureHWIO_4_statusExpos(
        true,
        HWIO_Polarity::high_level,
        HWIO_4_SignalType::status_expos,
        HWIO_StatusExpos_Timing::all_lines
        );
    
    cam.autoExposureOff();

    cout << "  Set resolution: H=" + to_string(camera_parameters.height) +
                ", W=" + to_string(camera_parameters.width)
        << endl;
    
  } catch (CameraException& err) {
      cout << "Error Code: " << (uint)err.error_code() << endl;
      cout << err.what() << endl;
      throw;
      return ;
  }
}
