cmake_minimum_required(VERSION 3.8)
project(spinnaker_camera_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# internal dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(std_msgs REQUIRED)

# external dependencies
find_package(OpenCV REQUIRED)

# Spinnaker needs to be found manually if no cmake config is provided
set(SPINNAKER_INCLUDE_DIRS "/opt/spinnaker/include")
set(SPINNAKER_LIB_DIRS "/opt/spinnaker/lib")
link_directories(${SPINNAKER_LIB_DIRS})

add_executable(camera_node src/camera_node.cpp)
target_include_directories(camera_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${SPINNAKER_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Use ament_target_dependencies for ROS packages to handle includes/libraries automatically
ament_target_dependencies(camera_node
  rclcpp
  sensor_msgs
  cv_bridge
  image_transport
  std_msgs
)

# Link non-ament dependencies manually
target_link_libraries(camera_node
  ${OpenCV_LIBRARIES}
  Spinnaker
)

install(TARGETS camera_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
