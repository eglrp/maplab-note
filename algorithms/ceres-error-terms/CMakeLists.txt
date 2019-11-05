cmake_minimum_required(VERSION 2.8.3)
project(ceres_error_terms)

link_directories(${CATKIN_DEVEL_PREFIX}/lib)

find_package(catkin_simple REQUIRED)
#catkin_simple(ALL_DEPS_REQUIRED)
catkin_simple()

add_definitions(-std=c++11 -Wno-enum-compare)

cs_add_library(${PROJECT_NAME}
  src/block-pose-prior-error-term.cc
  src/ceres-signal-handler.cc
  src/inertial-error-term.cc
  src/inertial-error-term-eigen.cc
  src/parameterization/quaternion-param-eigen.cc
  src/parameterization/quaternion-param-hamilton.cc
  src/parameterization/quaternion-param-jpl.cc
  src/parameterization/unit3-param.cc
  src/pose-prior-error-term-eigen.cc
  src/pose-prior-error-term.cc
  src/position-error-term.cc
  src/problem-information.cc)

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
        /home/wya/maplab_ws/src/maplab/algorithms/imu-integrator-rk4/include
        )

target_link_libraries(${PROJECT_NAME} pthread)

catkin_add_gtest(test_quaternion_parameterization_test
  test/test_quaternion_parameterization_test.cc)
target_link_libraries(test_quaternion_parameterization_test ${PROJECT_NAME})

catkin_add_gtest(test_position_error_term
  test/test_position_error_term.cc)
target_link_libraries(test_position_error_term ${PROJECT_NAME})

catkin_add_gtest(test_pose_parameterization
  test/test_pose_parameterization_test.cc)
target_link_libraries(test_pose_parameterization ${PROJECT_NAME})

catkin_add_gtest(test_few_points_quaternion_test
  test/test_few_points_quaternion_test.cc)
target_link_libraries(test_few_points_quaternion_test ${PROJECT_NAME})

catkin_add_gtest(test_generic_prior_error_term
  test/test_generic_prior_error_term.cc)
target_link_libraries(test_generic_prior_error_term ${PROJECT_NAME})

catkin_add_gtest(test_visual_term_test test/test_visual_term_test.cc)
target_link_libraries(test_visual_term_test ${PROJECT_NAME})

catkin_add_gtest(test_switchable_constraints_block_pose_test
  test/test_switchable_constraints_block_pose_test.cc)
target_link_libraries(test_switchable_constraints_block_pose_test ${PROJECT_NAME})

catkin_add_gtest(test_switchable_constraints_block_pose_baseframe_test
  test/test_switchable_constraints_block_pose_baseframe_test.cc)
target_link_libraries(test_switchable_constraints_block_pose_baseframe_test ${PROJECT_NAME})

catkin_add_gtest(test_inertial_term_test
  test/test_inertial_term_test.cc)
target_link_libraries(test_inertial_term_test ${PROJECT_NAME})

catkin_add_gtest(test_inertial_term_test_eigen
  test/test_inertial_term_test_eigen.cc)
target_link_libraries(test_inertial_term_test_eigen ${PROJECT_NAME})

catkin_add_gtest(test_3keyframe_inertial_term_test
  test/test_3keyframe_inertial_term_test.cc)
target_link_libraries(test_3keyframe_inertial_term_test ${PROJECT_NAME})

catkin_add_gtest(test_mission_baseframe_visual_term_test
  test/test_mission_baseframe_visual_term_test.cc)
target_link_libraries(test_mission_baseframe_visual_term_test ${PROJECT_NAME})

catkin_add_gtest(test_block_pose_prior_error_term test/test_block_pose_prior_error_term.cc)
target_link_libraries(test_block_pose_prior_error_term ${PROJECT_NAME})

catkin_add_gtest(test_pose_prior_error_term test/test_pose_prior_error_term.cc)
target_link_libraries(test_pose_prior_error_term ${PROJECT_NAME})

catkin_add_gtest(test_six_dof_block_transformation_error_term
  test/test_six_dof_block_transformation_error_term.cc)
target_link_libraries(test_six_dof_block_transformation_error_term
  ${PROJECT_NAME})

catkin_add_gtest(test_six_dof_block_transformation_error_term_with_extrinsics
  test/test_six_dof_block_transformation_error_term_with_extrinsics.cc)
target_link_libraries(test_six_dof_block_transformation_error_term_with_extrinsics ${PROJECT_NAME})

catkin_add_gtest(test_unit3_parameterization
  test/test_unit3_parameterization.cc)
target_link_libraries(test_unit3_parameterization ${PROJECT_NAME})

catkin_add_gtest(test_quaternion_parameterization_eigen
  test/test_quaternion_eigen_parameterization.cc)
target_link_libraries(test_quaternion_parameterization_eigen ${PROJECT_NAME})

catkin_add_gtest(test_aid_error_term
  test/test_aid_error_term.cc)
target_link_libraries(test_aid_error_term ${PROJECT_NAME})

cs_install()
cs_export()
