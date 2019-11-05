cmake_minimum_required(VERSION 2.8.3)
project(vi_map_helpers)

find_package(catkin_simple REQUIRED)
#catkin_simple(ALL_DEPS_REQUIRED)

#only debug
catkin_simple()


#cs_add_library(${PROJECT_NAME}
#  src/mission-clustering-coobservation.cc
#  src/near-camera-pose-sampling.cc
#  src/spatial-database-vertex-id.cc
#  src/vi-map-descriptor-utils.cc
#  src/vi-map-geometry.cc
#  src/vi-map-landmark-quality-evaluation.cc
#  src/vi-map-manipulation.cc
#  src/vi-map-nearest-neighbor-lookup.cc
#  src/vi-map-partitioner.cc
#  src/vi-map-queries.cc
#  src/vi-map-stats.cc
#  src/vi-map-vertex-time-queries.cc
#)


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
        /home/wya/maplab_ws/src/maplab/algorithms/map-optimization-legacy/include)

catkin_add_gtest(test_map_geometry_test
  test/test_map_geometry_test.cc)
target_link_libraries(test_map_geometry_test ${PROJECT_NAME})

catkin_add_gtest(test_mission_clustering_coobservation
  test/test_mission_clustering_coobservation.cc)
target_link_libraries(test_mission_clustering_coobservation ${PROJECT_NAME})

catkin_add_gtest(test_mission_observer_clustering
  test/test_mission_observer_clustering.cc)
target_link_libraries(test_mission_observer_clustering ${PROJECT_NAME})

catkin_add_gtest(test_queries_vertex_getter
  test/test_queries_vertex_getter.cc)
target_link_libraries(test_queries_vertex_getter ${PROJECT_NAME})

catkin_add_gtest(test_spatial_database_test
  test/test_spatial_database_test.cc)
target_link_libraries(test_spatial_database_test ${PROJECT_NAME})

catkin_add_gtest(test_nearest_neighbor_lookup_test
  test/test_nearest_neighbor_lookup_test.cc
)
target_link_libraries(test_nearest_neighbor_lookup_test ${PROJECT_NAME})

catkin_add_gtest(test_vertex_time_queries_test
  test/test_vertex_time_queries_test.cc
)
target_link_libraries(test_vertex_time_queries_test ${PROJECT_NAME})

cs_install()
cs_export()
