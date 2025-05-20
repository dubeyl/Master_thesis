#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "stdafx.h"
#include "camera.h"
#include "cameraexception.h"
#include <chrono>
#include <fstream>
#include <stdio.h>

using namespace std;
using namespace pco;

int record_few()
{
  try
  {
    int err = PCO_InitializeLib();
    if (err)
    {
      throw pco::CameraException(err);
    }
    cout<<"Reoopening camera"<<endl;
    pco::Camera cam;
    pco::Image img;
    int width, height;
    int image_count = 10;
    cam.defaultConfiguration();

    cam.setExposureTime(0.01);
    std::string path = "./reboot_rec";

    cam.record(image_count, pco::RecordMode::sequence);

    for (int i = 0; i < image_count; i++)
    {
      cout<<"Recording image "<<i<<endl;
      cam.image(img, i, pco::DataFormat::Mono16);
      width = img.width();
      height = img.height();
      std::filesystem::path name = std::filesystem::absolute(path).append("img_" + std::to_string(i + 1) + ".tif");
      std::filesystem::path name_raw = std::filesystem::absolute(path).append("img_" + std::to_string(i + 1) + "_raw.tif");
      std::cout << "Image Count:" << i + 1 << " > " << name << std::endl;
      int err;
     
       err = PCO_RecorderSaveImage(img.raw_data().first, width, height, FILESAVE_IMAGE_BW_16, false, name_raw.string().c_str(), true, img.getMetaDataPtr());
       if (err)
       {
         throw pco::CameraException(err);
       }
       err = PCO_RecorderSaveImage(img.data().first, width, height, FILESAVE_IMAGE_BW_8, true, name.string().c_str(), true, img.getMetaDataPtr());
       if (err)
       {
         throw pco::CameraException(err);
       }
     }
  }
  catch (pco::CameraException& err)
  {
    std::cout << "Error Code: " << err.error_code() << std::endl;
    std::cout << err.what() << std::endl;
  }
  PCO_CleanupLib();
  cout<<"Done and cleaned up"<<endl;

  return 0;
}

int main()
{cout<<"Opening camera"<<endl;

  PCO_InitializeLib();
  Camera cam = Camera();
  
  cout<<"Camera open Rebooting"<<endl;

  PCO_RebootCamera(cam.sdk());
  std::this_thread::sleep_for(std::chrono::seconds(10));
  PCO_CloseCamera(cam.sdk());
  
  cout<<"Closed camera now acquiring a few images"<<endl;
  record_few();

  cout << "All done." << endl;
  return 0;

  }
