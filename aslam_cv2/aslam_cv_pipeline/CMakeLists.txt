cmake_minimum_required(VERSION 2.8.3)
project(aslam_cv_pipeline)

find_package(catkin_simple REQUIRED)
#catkin_simple(ALL_DEPS_REQUIRED)
catkin_simple()

#############
# LIBRARIES #
#############
set(HEADERS
  include/aslam/pipeline/test/convert-maps-legacy.h
  include/aslam/pipeline/undistorter.h
  include/aslam/pipeline/undistorter-mapped.h
  include/aslam/pipeline/undistorter-mapped-inl.h
  include/aslam/pipeline/visual-npipeline.h
  include/aslam/pipeline/visual-pipeline.h
  include/aslam/pipeline/visual-pipeline-brisk.h
  include/aslam/pipeline/visual-pipeline-freak.h
  include/aslam/pipeline/visual-pipeline-null.h
)

set(SOURCES
  src/test/convert-maps-legacy.cc
  src/undistorter.cc
  src/undistorter-mapped.cc
  src/visual-npipeline.cc
  src/visual-pipeline-brisk.cc
  src/visual-pipeline-freak.cc
  src/visual-pipeline-null.cc
  src/visual-pipeline.cc
)

cs_add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

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


add_definitions(-std=c++11)
SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS} -lpthread")

##########
# GTESTS #
##########
catkin_add_gtest(test_undistorters test/test-undistorters.cc)
target_link_libraries(test_undistorters ${PROJECT_NAME}) 

catkin_add_gtest(test_visual-npipeline test/test-visual-npipeline.cc)
target_link_libraries(test_visual-npipeline ${PROJECT_NAME}) 

##########
# EXPORT #
##########
cs_install()
cs_export()
