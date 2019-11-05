cmake_minimum_required(VERSION 2.8.3)
project(rovioli)
set(CMAKE_CXX_STANDARD 14)

find_package(catkin_simple REQUIRED)
#catkin_simple(ALL_DEPS_REQUIRED)
catkin_simple()

#############
# LIBRARIES #
#############
cs_add_library(${PROJECT_NAME}_lib
  src/data-publisher-flow.cc
  src/datasource-factory.cc
  src/datasource-rosbag.cc
  src/datasource-rostopic.cc
  src/feature-tracking.cc
  src/imu-camera-synchronizer.cc
  src/localizer.cc
  src/map-builder-flow.cc
  src/rovio-factory.cc
  src/rovio-flow.cc
  src/rovioli-node.cc
  src/synced-nframe-throttler.cc
  src/vio-update-builder.cc
)

#######
# APP #
#######
cs_add_executable(rovioli
  app/rovioli-app.cc
)
target_link_libraries(rovioli ${PROJECT_NAME}_lib)

#########
# SHARE #
#########
catkin_add_env_hooks(10.rovio-config-template
  SHELLS sh
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/env-hooks
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
        /home/wya/maplab_ws/src/maplab/aslam_cv2/aslam_cv_tracker/include)


##########
# GTESTS #
##########
#catkin_add_gtest(test_localizer test/test-localizer.cc)
#target_link_libraries(test_localizer ${PROJECT_NAME}_lib)
#maplab_import_test_maps(test_localizer)

#catkin_add_gtest(test_feature_tracking test/test-feature-tracking.cc)
#target_link_libraries(test_feature_tracking ${PROJECT_NAME}_lib)
#maplab_import_test_maps(test_feature_tracking)

#catkin_add_gtest(test_vio_update_builder test/test-vio-update-builder.cc)
#target_link_libraries(test_vio_update_builder ${PROJECT_NAME}_lib)
#
#execute_process(COMMAND tar -xzf ${MAPLAB_TEST_DATA_DIR}/end_to_end_test/end_to_end_test.tar.gz)
#catkin_add_nosetests(test/end-to-end-test.py)

############
## EXPORT ##
############
cs_install()
cs_export()
