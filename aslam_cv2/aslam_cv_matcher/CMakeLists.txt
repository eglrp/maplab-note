cmake_minimum_required(VERSION 2.8.3)
project(aslam_cv_matcher)

find_package(catkin_simple REQUIRED)
#catkin_simple(ALL_DEPS_REQUIRED)
catkin_simple()

#############
# LIBRARIES #
#############
set(HEADERS
        include/aslam/matcher/gyro-two-frame-matcher.h
        include/aslam/matcher/match.h
        include/aslam/matcher/match-helpers.h
        include/aslam/matcher/match-helpers-inl.h
        include/aslam/matcher/match-visualization.h
        include/aslam/matcher/matching-engine.h
        include/aslam/matcher/matching-engine-exclusive.h
        include/aslam/matcher/matching-engine-greedy.h
        include/aslam/matcher/matching-engine-non-exclusive.h
        include/aslam/matcher/matching-problem.h
        include/aslam/matcher/matching-problem-frame-to-frame.h
        )

set(SOURCES
        src/gyro-two-frame-matcher.cc
        src/match-helpers.cc
        src/match-visualization.cc
        src/matching-problem.cc
        src/matching-problem-frame-to-frame.cc
        )

# for ./aslam_cv/aslam_cv_common/include/aslam/common/hamming-inl.h
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mssse3")
cs_add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

#add_doxygen(NOT_AUTOMATIC)

add_definitions(-std=c++11)
SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS} -lpthread")


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
        /home/wya/maplab_ws/src/maplab/aslam_cv2/aslam_cv_tracker/include
        /home/wya/maplab_ws/src/maplab_dependencies/internal/ethzasl_brisk/brisk/include
        /home/wya/maplab_ws/src/maplab/aslam_cv2/aslam_cv_visualization/include
        /home/wya/maplab_ws/src/maplab/aslam_cv2/aslam_cv_matcher/include
        /home/wya/maplab_ws/src/maplab/aslam_cv2/aslam_cv_triangulation/include)


##########
# GTESTS #
##########
catkin_add_gtest(test_matcher test/test-matcher.cc)
target_link_libraries(test_matcher ${PROJECT_NAME})

catkin_add_gtest(test_matcher_non_exclusive test/test-matcher-non-exclusive.cc)
target_link_libraries(test_matcher_non_exclusive ${PROJECT_NAME})

##########
# EXPORT #
##########
cs_install()
cs_export()

