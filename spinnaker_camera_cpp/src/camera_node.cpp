#include "SpinGenApi/SpinnakerGenApi.h"
#include "Spinnaker.h"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <atomic>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <thread>

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace cv;
using namespace std;

class SpinnakerCameraNode : public rclcpp::Node {
public:
  SpinnakerCameraNode() : Node("spinnaker_camera_node") {
    // ROS 2 parameters
    this->declare_parameter("camera_index", 0);
    this->declare_parameter("acquisition_mode", "Continuous");
    this->declare_parameter("frame_id", "camera_optical_frame");

    camera_index_ = this->get_parameter("camera_index").as_int();
    frame_id_ = this->get_parameter("frame_id").as_string();

    publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
        "/camera/image_raw", 10);

    if (init_camera()) {
      start_capture_thread();
    }

    RCLCPP_INFO(this->get_logger(), "Spinnaker Camera Node Initialized");
  }

  ~SpinnakerCameraNode() {
    stop_capture_thread();
    cleanup_camera();
  }

private:
  bool init_camera() {
    int retries = 5;
    while (retries > 0) {
      try {
        system_ = System::GetInstance();
        cam_list_ = system_->GetCameras();

        if (cam_list_.GetSize() > 0) {
          break;
        }

        RCLCPP_WARN(this->get_logger(),
                    "No cameras found. Retrying... (%d left)", retries);
        cam_list_.Clear();
        system_->ReleaseInstance();

        std::this_thread::sleep_for(std::chrono::seconds(1));
        retries--;
      } catch (Spinnaker::Exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Error querying cameras: %s",
                     e.what());
        std::this_thread::sleep_for(std::chrono::seconds(1));
        retries--;
      }
    }

    if (cam_list_.GetSize() == 0) {
      RCLCPP_ERROR(this->get_logger(),
                   "No camera detected after retries. Node will standby.");
      RCLCPP_ERROR(
          this->get_logger(),
          "CHECK UDEV RULES: ensure /etc/udev/rules.d/40-flir-spinnaker.rules "
          "exists and reload rules.");
      return false;
    }

    try {
      if (static_cast<unsigned int>(camera_index_) >= cam_list_.GetSize()) {
        RCLCPP_ERROR(
            this->get_logger(),
            "Camera index %d out of bounds (found %d cameras). Using index 0.",
            camera_index_, cam_list_.GetSize());
        camera_index_ = 0;
      }

      camera_ = cam_list_.GetByIndex(camera_index_);

      camera_->Init();

      // Configure for continuous acquisition
      // It's good practice to set acquisition mode to Continuous
      INodeMap &nodeMap = camera_->GetNodeMap();
      CEnumerationPtr ptrAcquisitionMode = nodeMap.GetNode("AcquisitionMode");
      if (!IsAvailable(ptrAcquisitionMode) || !IsWritable(ptrAcquisitionMode)) {
        RCLCPP_WARN(this->get_logger(),
                    "Unable to set acquisition mode (node retrieval failed).");
      } else {
        CEnumEntryPtr ptrAcquisitionModeContinuous =
            ptrAcquisitionMode->GetEntryByName("Continuous");
        if (IsAvailable(ptrAcquisitionModeContinuous) &&
            IsReadable(ptrAcquisitionModeContinuous)) {
          ptrAcquisitionMode->SetIntValue(
              ptrAcquisitionModeContinuous->GetValue());
        }
      }

      // Enable and set frame rate to 60 FPS
      CBooleanPtr ptrFrameRateEnable =
          nodeMap.GetNode("AcquisitionFrameRateEnable");
      if (IsAvailable(ptrFrameRateEnable) && IsWritable(ptrFrameRateEnable)) {
        ptrFrameRateEnable->SetValue(true);
        RCLCPP_INFO(this->get_logger(), "Frame rate control enabled");
      }

      CFloatPtr ptrFrameRate = nodeMap.GetNode("AcquisitionFrameRate");
      if (IsAvailable(ptrFrameRate) && IsWritable(ptrFrameRate)) {
        ptrFrameRate->SetValue(30.0);
        RCLCPP_INFO(this->get_logger(), "Frame rate set to 30 FPS");
      } else {
        RCLCPP_WARN(this->get_logger(),
                    "Unable to set frame rate (node not available)");
      }

      camera_->BeginAcquisition();
      RCLCPP_INFO(this->get_logger(),
                  "Camera acquisition started for camera %s",
                  camera_->GetDeviceID().c_str());
      return true;

    } catch (Spinnaker::Exception &e) {
      RCLCPP_ERROR(this->get_logger(), "Error initializing camera: %s",
                   e.what());
      return false;
    }
  }

  void cleanup_camera() {
    if (camera_ && camera_->IsInitialized()) {
      try {
        camera_->EndAcquisition();
        camera_->DeInit();
      } catch (Spinnaker::Exception &e) {
        RCLCPP_WARN(this->get_logger(), "Error cleaning up camera: %s",
                    e.what());
      }
    }
    // Clear camera pointer before clearing list to release reference
    camera_ = nullptr;
    cam_list_.Clear();
    if (system_) {
      system_->ReleaseInstance();
      system_ = nullptr;
    }
  }

  void start_capture_thread() {
    running_ = true;
    capture_thread_ = std::thread(&SpinnakerCameraNode::capture_loop, this);
  }

  void stop_capture_thread() {
    running_ = false;
    if (capture_thread_.joinable()) {
      capture_thread_.join();
    }
  }

  void capture_loop() {
    if (!camera_ || !camera_->IsInitialized())
      return;

    RCLCPP_INFO(this->get_logger(), "Entering capture loop...");

    while (rclcpp::ok() && running_) {
      try {
        // GetNextImage(timeout) blocks. Default timeout is usually ample.
        // We check running_ flag to exit loop gracefully.
        ImagePtr img = camera_->GetNextImage(1000); // 1s timeout

        if (img->IsIncomplete()) {
          // Timeouts or incomplete frames are expected occasionally
          if (img->GetImageStatus() != SPINNAKER_IMAGE_STATUS_NO_ERROR) {
            RCLCPP_WARN(
                this->get_logger(), "Image incomplete: %s",
                Image::GetImageStatusDescription(img->GetImageStatus()));
          }
          img->Release();
          continue;
        }

        // Convert to BGR for OpenCV/ROS
        ImageProcessor processor;
        ImagePtr converted = processor.Convert(img, PixelFormat_BGR8);

        // Create OpenCV matrix with CORRECT STRIDE
        // converted->GetStride() returns bytes per row.
        // Important: We share the buffer of 'converted' which is managed by
        // Spinnaker We must ensure 'converted' outlives usage or we copy
        // safely. cv_bridge copy happens right after this.
        Mat frame(converted->GetHeight(), converted->GetWidth(), CV_8UC3,
                  converted->GetData(), converted->GetStride());

        // Convert to ROS message
        std_msgs::msg::Header header;
        header.stamp = this->now();
        header.frame_id = frame_id_;

        // CvImage copies the data into the ROS message
        sensor_msgs::msg::Image::SharedPtr msg =
            cv_bridge::CvImage(header, "bgr8", frame).toImageMsg();

        publisher_->publish(*msg);

        img->Release();

      } catch (Spinnaker::Exception &e) {
        // SPINNAKER_ERR_TIMEOUT is common if no trigger, but here in continuous
        // mode it might mean issue. We just log and continue.
        RCLCPP_DEBUG(this->get_logger(), "Spinnaker Exception in loop: %s",
                     e.what());
      }
    }
    RCLCPP_INFO(this->get_logger(), "Capture loop exited.");
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;

  SystemPtr system_;
  CameraList cam_list_;
  CameraPtr camera_;

  int camera_index_;
  std::string frame_id_;

  std::thread capture_thread_;
  std::atomic<bool> running_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SpinnakerCameraNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}