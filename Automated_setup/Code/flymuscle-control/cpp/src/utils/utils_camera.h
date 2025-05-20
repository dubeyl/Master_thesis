#ifndef UTILS_CAMERA_H
#define UTILS_CAMERA_H

#include "utils.h"
#include <opencv2/opencv.hpp>
#include "camera.h"

struct ImageData {
  int frame_id;         // id of the frame
  cv::Mat image;        // 1-channel image
  bool is_kinematic;
  long long timestamp;
  double stage_pos;
};

class ImageDataVec {
  public:
      explicit ImageDataVec(int num_channels);  // Constructor
  
      // Accessor methods
      int getNumChannels() const;
      std::vector<int>& getFrameIds();
      std::vector<long long>& getTimestamps();
      std::vector<bool>& getIsKinematics();
      std::vector<cv::Mat>& getImages();
      std::vector<double>& getStagePos();
  
      // New method to insert data circularly
      void addData(int frame_id, long long timestamp, cv::Mat image, bool iskin, double stage_pos);
  
      // Additional helper methods
      bool isFull() const;  // Check if all slots are occupied
  
  private:
      std::vector<int> frame_ids;
      std::vector<long long> timestamps;
      std::vector<bool> is_kinematics;
      std::vector<cv::Mat> images;
      std::vector<double> stage_pos;
  
      int num_channels;
      int current_index;  // Points to the next slot to insert
      bool is_full;       // Becomes true when buffer wraps around
  };
void setupCamera(pco::Camera& cam, CameraParameters& camera_parameters);

#endif