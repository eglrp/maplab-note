cmake_minimum_required(VERSION 2.8.3)
project(map_resources)

find_package(catkin_simple REQUIRED)
#catkin_simple(ALL_DEPS_REQUIRED)
catkin_simple()
############
# PROTOBUF #
############
set(PROTO_DEFNS proto/map-resources/resource_info_map.proto
                proto/map-resources/resource_metadata.proto)
#PROTOBUF_CATKIN_GENERATE_CPP2("proto" PROTO_SRCS PROTO_HDRS ${PROTO_DEFNS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

#############
# LIBRARIES #
#############
cs_add_library(${PROJECT_NAME} src/resource-cache.cc
                               src/resource-common.cc
                               src/resource-conversion.cc
                               src/resource-loader.cc
                               src/resource-map-serialization.cc
                               src/resource-map.cc
                               src/tinyply/tinyply.cc
                               ${PROTO_SRCS}
                               ${PROTO_HDRS})

#############
## TESTING ##
#############
SET(PROJECT_TEST_DATA "${PROJECT_NAME}_test_data")
add_custom_target(${PROJECT_TEST_DATA})
add_custom_command(TARGET ${PROJECT_TEST_DATA}
COMMAND rm -rf "./${PROJECT_TEST_DATA}/*" && tar -xvzf ${MAPLAB_TEST_DATA_DIR}/${PROJECT_TEST_DATA}/${PROJECT_TEST_DATA}.tar.gz)

catkin_add_gtest(test_resource_loader test/test_resource_loader.cc)
target_link_libraries(test_resource_loader ${PROJECT_NAME})
add_dependencies(test_resource_loader ${PROJECT_TEST_DATA})

catkin_add_gtest(test_resource_map test/test_resource_map.cc)
target_link_libraries(test_resource_map ${PROJECT_NAME})
add_dependencies(test_resource_map ${PROJECT_TEST_DATA})

catkin_add_gtest(test_resource_conversion test/test_resource_conversion.cc)
target_link_libraries(test_resource_conversion ${PROJECT_NAME})
add_dependencies(test_resource_conversion ${PROJECT_TEST_DATA})

catkin_add_gtest(test_optional_sensor_resources test/test_optional_sensor_resources.cc)
target_link_libraries(test_optional_sensor_resources ${PROJECT_NAME})
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
        /home/wya/maplab_ws/src/maplab/console-plugins/mapping-workflows-plugin/include)
############
## EXPORT ##
############
cs_install()
cs_export()
