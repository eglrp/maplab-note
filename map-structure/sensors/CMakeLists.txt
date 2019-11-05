cmake_minimum_required(VERSION 2.8.3)
project(sensors)

find_package(catkin_simple REQUIRED)
#catkin_simple(ALL_DEPS_REQUIRED)
catkin_simple()

############
# PROTOBUF #
############
set(PROTO_DEFNS proto/sensors/measurements.proto)
set(BASE_PATH "proto")
#PROTOBUF_CATKIN_GENERATE_CPP2(${BASE_PATH} PROTO_SRCS PROTO_HDRS ${PROTO_DEFNS})

#############
# LIBRARIES #
#############
cs_add_library(${PROJECT_NAME} ${PROTO_SRCS} ${PROTO_HDRS}
  src/gps-utm.cc
  src/gps-wgs.cc
  src/imu.cc
  src/measurement.cc
  src/relative-6dof-pose.cc
  src/sensor.cc
  src/sensor-extrinsics.cc
  src/sensor-factory.cc
  src/sensor-system.cc
  ${PROTO_SRCS}
  ${PROTO_HDRS}
)

find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR}
        /home/wya/maplab_ws/src/maplab/algorithms/landmark-triangulation/include
        /home/wya/maplab_ws/devel/include/
        /home/wya/maplab_ws/src/maplab/console-plugins/loop-closure-plugin/include/
        /home/wya/maplab_ws/src/maplab/algorithms/map-optimization/include
        /home/wya/maplab_ws/src/maplab/console-plugins/map-sparsification-plugin/include
        /home/wya/maplab_ws/src/maplab/algorithms/vi-map-helpers/include
        /home/wya/maplab_ws/src/maplab/map-structure/vi-map/include
        /home/wya/maplab_ws/src/maplab/aslam_cv2/aslam_cv_common/include/
        /home/wya/maplab_ws/src/maplab/backend/map-resources/include
        /home/wya/maplab_ws/src/maplab/common/maplab-common/include
        /home/wya/maplab_ws/src/maplab/map-structure/posegraph/include
        /home/wya/maplab_ws/src/maplab_dependencies/internal/minkindr/minkindr/include
        /home/wya/maplab_ws/src/maplab_dependencies/internal/voxblox/voxblox/include
        /home/wya/maplab_ws/src/maplab/aslam_cv2/aslam_cv_cameras/include
        /home/wya/maplab_ws/src/maplab/aslam_cv2/aslam_cv_frames/include
        /home/wya/maplab_ws/src/maplab/common/console-common/include
        /home/wya/maplab_ws/src/maplab/backend/map-manager/include
        /home/wya/maplab_ws/src/maplab/algorithms/map-sparsification/include
        /home/wya/maplab_ws/src/maplab/visualization/include/
        /home/wya/maplab_ws/src/maplab_dependencies/internal/eigen_checks/include
        /home/wya/maplab_ws/src/maplab/algorithms/map-optimization-legacy/include
        /home/wya/maplab_ws/src/maplab/algorithms/ceres-error-terms/include/
        /home/wya/maplab_ws/src/maplab/common/maplab-common/include/
        /home/wya/maplab_ws/src/maplab/common/vio-common/include/
        /home/wya/maplab_ws/src/maplab/map-structure/sensors/include
        )


##########
# GTESTS #
##########
catkin_add_gtest(test_serialization test/test-sensor-serialization.cc)
target_link_libraries(test_serialization ${PROJECT_NAME})

catkin_add_gtest(test_sensors test/test-sensors.cc)
target_link_libraries(test_sensors ${PROJECT_NAME})

catkin_add_gtest(test_sensor_system test/test-sensor-system.cc)
target_link_libraries(test_sensor_system ${PROJECT_NAME})

catkin_add_gtest(test_measurements test/test-measurements.cc)
target_link_libraries(test_measurements ${PROJECT_NAME})

##########
# EXPORT #
##########
cs_install()
cs_export()
