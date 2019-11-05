cmake_minimum_required(VERSION 2.8.3)
project(map_optimization)

find_package(catkin_simple REQUIRED)
#catkin_simple(ALL_DEPS_REQUIRED)
catkin_simple()

########
# LIB  #
########
cs_add_library(${PROJECT_NAME}
  src/augment-loopclosure.cc
  src/optimization-problem.cc
  src/optimization-state-buffer.cc
  src/optimization-terms-addition.cc
  src/outlier-rejection-solver.cc
  src/solver.cc
  src/solver-options.cc
  src/vi-map-optimizer.cc
  src/vi-map-relaxation.cc
  src/vi-optimization-builder.cc
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
        /home/wya/maplab_ws/src/maplab/algorithms/loopclosure/loopclosure-common/include)


#############
## TESTING ##
#############
catkin_add_gtest(test_optimization test/test-optimization.cc)
target_link_libraries(test_optimization ${PROJECT_NAME})
#maplab_import_test_maps(test_optimization)

##########
# EXPORT #
##########
cs_install()
cs_export()
