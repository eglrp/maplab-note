cmake_minimum_required(VERSION 2.8.3)
project(vio_common)

find_package(catkin_simple REQUIRED)
#catkin_simple(ALL_DEPS_REQUIRED)
catkin_simple()

############
# PROTOBUF #
############
SET(PROTO_DEFNS proto/vio-common/vio_update.proto
                proto/vio-common/vio_types.proto)
#PROTOBUF_CATKIN_GENERATE_CPP2("proto" PROTO_SRCS PROTO_HDRS ${PROTO_DEFNS})

#############
# LIBRARIES #
#############
add_definitions(-std=c++11)

#add_doxygen(NOT_AUTOMATIC)
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
        /home/wya/maplab_ws/src/maplab/algorithms/map-optimization/include
        /home/wya/maplab_ws/src/maplab/algorithms/loopclosure/loop-closure-handler/include
        /home/wya/maplab_ws/src/maplab/map-structure/vi-map/include
        /home/wya/maplab_ws/src/maplab/common/maplab-common/include
        /home/wya/maplab_ws/src/maplab/algorithms/loopclosure/descriptor-projection/include
        /home/wya/maplab_ws/src/maplab/map-structure/localization-summary-map/include
        /home/wya/maplab_ws/src/maplab/algorithms/loopclosure/loopclosure-common/include
        /home/wya/maplab_ws/src/maplab/common/message_flow/include
        /home/wya/maplab_ws/src/maplab_dependencies/internal/maplab_rovio/include
        /home/wya/maplab_ws/src/maplab/algorithms/online_map_builders/include
        /home/wya/maplab_ws/src/maplab/console-plugins/mapping-workflows-plugin/include
        /home/wya/maplab_ws/src/maplab_dependencies/internal/minkindr_ros/minkindr_conversions/include
        /home/wya/maplab_ws/src/maplab/algorithms/feature-tracking/include
        /home/wya/maplab_ws/src/maplab/aslam_cv2/aslam_cv_tracker/include
        /home/wya/test_maplab_ws/src/maplab_dependencies/internal/ethzasl_brisk/brisk/include)

if(APPLE)
  add_definitions(-DGTEST_HAS_TR1_TUPLE=0)
endif()
SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS} -lpthread")

cs_add_library(${PROJECT_NAME} ${PROTO_SRCS} ${PROTO_HDRS}
  src/imu-measurements-buffer.cc
  src/test/vio-update-simulation.cc
  src/rostopic-settings.cc
  src/vio-update-serialization.cc
)

##########
# GTESTS #
##########
catkin_add_gtest(test_imu_measurements_buffer test/test-imu-measurements-buffer.cc)
target_link_libraries(test_imu_measurements_buffer ${PROJECT_NAME})

catkin_add_gtest(test_vio_update_serialization_test test/test-vio-update-serialization.cc)
target_link_libraries(test_vio_update_serialization_test ${PROJECT_NAME})

##########
# EXPORT #
##########
cs_install()
cs_export()
