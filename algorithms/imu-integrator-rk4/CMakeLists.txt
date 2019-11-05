cmake_minimum_required(VERSION 2.8.3)
project(imu_integrator_rk4)

find_package(catkin_simple REQUIRED)
#catkin_simple(ALL_DEPS_REQUIRED)
catkin_simple()

cs_add_library(${PROJECT_NAME}
  src/imu-integrator.cc
  src/imu-integrator-eigen.cc
)
target_link_libraries(${PROJECT_NAME} pthread)

catkin_add_gtest(test_imu_integrator_basic_test
  test/test_imu_integrator_basic_test.cc)
target_link_libraries(test_imu_integrator_basic_test ${PROJECT_NAME})

catkin_add_gtest(test_imu_integrator_trajectory_test
  test/test_imu_integrator_trajectory_test.cc)
target_link_libraries(test_imu_integrator_trajectory_test ${PROJECT_NAME})

catkin_add_gtest(test_imu_integrator_cov_phi_test
  test/test_imu_integrator_cov_phi_test.cc)
target_link_libraries(test_imu_integrator_cov_phi_test ${PROJECT_NAME})

catkin_add_gtest(test_imu_integrator_eigen_basic_test
  test/test_imu_integrator_eigen_basic_test.cc)
target_link_libraries(test_imu_integrator_eigen_basic_test ${PROJECT_NAME})

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
        /home/wya/maplab_ws/src/maplab/common/maplab-common/include)


cs_install()
cs_export()
