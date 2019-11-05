cmake_minimum_required(VERSION 2.8)
project(matching_based_loopclosure)

find_package(catkin_simple REQUIRED)
catkin_simple()

set(PROTO_DEFNS proto/matching-based-loopclosure/matching_based_loop_detector.proto)
#PROTOBUF_CATKIN_GENERATE_CPP2("proto" PROTO_SRCS PROTO_HDRS ${PROTO_DEFNS})

set(LIBRARY_NAME ${PROJECT_NAME})
#cs_add_library(${LIBRARY_NAME} src/detector-settings.cc
#                               src/matching-based-engine.cc
#                               src/train-vocabulary.cc
#                               ${PROTO_SRCS})

# Install loopclosure files and export their location to the Catkin environment.
set(MAPLAB_LOOPCLOSURE_DESTINATION ${CATKIN_GLOBAL_SHARE_DESTINATION}/loopclosure)
set(FILES_TO_COPY share/inverted_multi_index_quantizer_freak.dat share/projection_matrix_freak.dat
                  share/inverted_multi_index_quantizer_brisk.dat share/projection_matrix_brisk.dat)
#file(COPY ${FILES_TO_COPY}
#     DESTINATION ${CATKIN_DEVEL_PREFIX}/${MAPLAB_LOOPCLOSURE_DESTINATION})
install(FILES ${FILES_TO_COPY}
        DESTINATION ${MAPLAB_LOOPCLOSURE_DESTINATION})
catkin_add_env_hooks(40.loopclosure
                     SHELLS sh
                     DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/env-hooks)

catkin_add_gtest(test_scoring test/test_scoring.cc)
target_link_libraries(test_scoring ${LIBRARY_NAME})

# CMake Indexing
FILE(GLOB_RECURSE LibFiles "include/*")
add_custom_target(headers SOURCES ${LibFiles})
FILE(GLOB_RECURSE ProtoFiles "proto/*")
add_custom_target(protos SOURCES ${ProtoFiles})

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
        /home/wya/maplab_ws/src/maplab/aslam_cv2/aslam_cv_geometric_vision/include
        /home/wya/maplab_ws/src/maplab/algorithms/loopclosure/matching-based-loopclosure/include
        /home/wya/maplab_ws/src/maplab/algorithms/loopclosure/inverted-multi-index/include)


cs_install()
cs_export()
