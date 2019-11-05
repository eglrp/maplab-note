#include "mapping-workflows-plugin/localization-map-creation.h"

#include <algorithm>

#include <glog/logging.h>
#include <landmark-triangulation/landmark-triangulation.h>
#include <loop-closure-plugin/vi-map-merger.h>
#include <map-optimization/solver-options.h>
#include <map-optimization/vi-map-optimizer.h>
#include <map-optimization/vi-map-relaxation.h>
#include <map-sparsification-plugin/keyframe-pruning.h>
#include <vi-map-helpers/vi-map-landmark-quality-evaluation.h>
#include <vi-map-helpers/vi-map-manipulation.h>
#include <vi-map/vi-map.h>

DECLARE_uint64(vi_map_landmark_quality_min_observers);

namespace mapping_workflows_plugin {//将一个刚跑完的地图转成定位地图
// Processes a raw map containing landmarks into a localization summary map.
// It runs the following steps:
// 1) landmark intialization, 看完了
// 2) retriangulation, 看完了
// 3) keyframing, 看完了
// 4) landmark quality evaluation, 看完了
// 5) visual-inertial batch optimization.
// 6) relaxation,
// 7) loopclosure,
// 8) visual-inertial batch optimization.
int processVIMapToLocalizationMap(
    bool initialize_landmarks,//不会初始化
    const map_sparsification::KeyframingHeuristicsOptions& keyframe_options,
    vi_map::VIMap* map, visualization::ViwlsGraphRvizPlotter* plotter) {
  // plotter is optional.
  CHECK_NOTNULL(map);

  vi_map::MissionIdList mission_ids;
  map->getAllMissionIds(&mission_ids);//获取所有任务
  if (mission_ids.size() != 1u) {
    LOG(WARNING) << "Only single mission supported. Aborting.";
    return common::CommandStatus::kStupidUserError;
  }
  const vi_map::MissionId& mission_id = mission_ids.front();

  // Initialize landmarks by triangulation.
  if (initialize_landmarks) {
    vi_map_helpers::VIMapManipulation manipulation(map);
    manipulation.initializeLandmarksFromUnusedFeatureTracksOfMission(//对所有的地标点进行更新
        mission_id);
    //重新三角化
    landmark_triangulation::retriangulateLandmarksOfMission(mission_id, map);
  }
  CHECK_GT(map->numLandmarks(), 0u);

  // Select keyframes along the mission. Unconditionally add the last vertex as
  // a keyframe if it isn't a keyframe already.
  //启发式的选择关键帧,并且会把非关键帧的节点进行删除
  int retval = map_sparsification_plugin::keyframeMapBasedOnHeuristics(
      keyframe_options, mission_id, plotter, map);
  if (retval != common::CommandStatus::kSuccess) {
    LOG(ERROR) << "Keyframing failed! Aborting.";
    return retval;
  }

  // Evaluate the quality of landmarks after keyframing the map.
  //在选择关键帧后还会重新评估一下地图点的质量
  FLAGS_vi_map_landmark_quality_min_observers = 2;
  vi_map_helpers::evaluateLandmarkQuality(map);

  // Common options for the subsequent optimizations.
  ceres::Solver::Options solver_options =
      map_optimization::initSolverOptionsFromFlags();//设置一些ceres的优化参数
  constexpr bool kEnableSignalHandler = true;
  map_optimization::VIMapOptimizer optimizer(plotter, kEnableSignalHandler);
  map_optimization::ViProblemOptions vi_problem_options =//
      map_optimization::ViProblemOptions::initFromGFlags();

  constexpr int kNumInitialOptviIterations = 3;
  solver_options.max_num_iterations = kNumInitialOptviIterations;

  // Initial visual-inertial optimization.
  //开始优化vio，算法在maplab_ws/src/maplab/algorithms/map-optimization/src/vi-map-optimizer.cc
  //
  bool success = optimizer.optimizeVisualInertial(//第一个option是优化时固定哪些参数的选择，第二个option是ceres自己的配置选项
      vi_problem_options, solver_options, {mission_id}, nullptr, map);
  if (!success) {
    LOG(ERROR) << "Optimization failed! Aborting.";
    return common::CommandStatus::kUnknownError;
  }

  // Overwrite the number of iterations to a reasonable value.
  // TODO(dymczykm) A temporary solution for the optimization not to take too
  // long. Better termination conditions are necessary.
  constexpr int kMaxNumIterations = 10;
  solver_options.max_num_iterations = kMaxNumIterations;

  // Relax the map.
  map_optimization::VIMapRelaxation relaxation(plotter, kEnableSignalHandler);//初始化ralax，输入可视化器
  success = relaxation.relax(solver_options, {mission_id}, map);//输入ceres优化选项，{mission_id}，地图
  if (!success) {
    LOG(WARNING) << "Pose-graph relaxation failed, but this might be fine if "
                 << "no loopclosures are present in the dataset.";
  }

  // Loop-close the map.
  loop_closure_plugin::VIMapMerger merger(map, plotter);
  retval = merger.findLoopClosuresBetweenAllMissions();
  if (retval != common::CommandStatus::kSuccess) {
    LOG(ERROR) << "Loop-closure failed! Aborting.";
    return retval;
  }

  // Optimize the map.
  success = optimizer.optimizeVisualInertial(
      vi_problem_options, solver_options, {mission_id}, nullptr, map);
  if (!success) {
    LOG(ERROR) << "Optimization failed! Aborting.";
    return common::CommandStatus::kUnknownError;
  }

  return common::CommandStatus::kSuccess;
}

}  // namespace mapping_workflows_plugin
