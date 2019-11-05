#include "map-optimization/vi-map-optimizer.h"

#include <functional>
#include <string>
#include <unordered_map>

#include <map-optimization/callbacks.h>
#include <map-optimization/outlier-rejection-solver.h>
#include <map-optimization/solver-options.h>
#include <map-optimization/solver.h>
#include <map-optimization/vi-optimization-builder.h>
#include <maplab-common/file-logger.h>
#include <maplab-common/progress-bar.h>
#include <visualization/viwls-graph-plotter.h>

DEFINE_int32(//每多少次优化更新一次可视化
    ba_visualize_every_n_iterations, 3,
    "Update the visualization every n optimization iterations.");

namespace map_optimization {

VIMapOptimizer::VIMapOptimizer(
    visualization::ViwlsGraphRvizPlotter* plotter, bool signal_handler_enabled)
    : plotter_(plotter), signal_handler_enabled_(signal_handler_enabled) {}

bool VIMapOptimizer::optimizeVisualInertial(
    const map_optimization::ViProblemOptions& options,
    const vi_map::MissionIdSet& missions_to_optimize,
    const map_optimization::OutlierRejectionSolverOptions* const
        outlier_rejection_options,
    vi_map::VIMap* map) {
  // outlier_rejection_options is optional.
  CHECK_NOTNULL(map);

  ceres::Solver::Options solver_options =
      map_optimization::initSolverOptionsFromFlags();
  return optimizeVisualInertial(
      options, solver_options, missions_to_optimize, outlier_rejection_options,
      map);
}

//vio优化
bool VIMapOptimizer::optimizeVisualInertial(
    const map_optimization::ViProblemOptions& options,//vio优化选项
    const ceres::Solver::Options& solver_options,//ceres优化选项
    const vi_map::MissionIdSet& missions_to_optimize,//优化任务队列
    const map_optimization::OutlierRejectionSolverOptions* const
        outlier_rejection_options,//这里是nullptr
    vi_map::VIMap* map)
    {
        // outlier_rejection_options is optional.
        CHECK_NOTNULL(map);

        if (missions_to_optimize.empty()) {
            LOG(WARNING) << "Nothing to optimize.";
            return false;
        }

        //构造vio优化问题，fix优化变量以及添加残差块
        map_optimization::OptimizationProblem::UniquePtr optimization_problem(
                map_optimization::constructViProblem(missions_to_optimize, options, map));
        CHECK(optimization_problem != nullptr);

        std::vector<std::shared_ptr<ceres::IterationCallback>> callbacks;
        if (plotter_) {//可视化部分
            map_optimization::appendVisualizationCallbacks//初始化可视化的回调函数
                    (
                            FLAGS_ba_visualize_every_n_iterations,//每几次迭代可视化一次
                            *(optimization_problem->getOptimizationStateBufferMutable()),
                            *plotter_, map, &callbacks
                    );
        }
        map_optimization::appendSignalHandlerCallback(&callbacks);
        ceres::Solver::Options solver_options_with_callbacks = solver_options;
        //这里只要是为了可视化操作，将option里的update_state_every_iteration设成true，设成true以后，
        // 每一次迭代都会去更新待优化变量，而不会等到找到极小值了以后再去更新优化量
        map_optimization::addCallbacksToSolverOptions(
                callbacks, &solver_options_with_callbacks);

        if (outlier_rejection_options != nullptr)
        {//outlier_rejection_options maplab给的是个空指针，所以就不去关注有的这种情况了
            map_optimization::solveWithOutlierRejection(
                    solver_options_with_callbacks, *outlier_rejection_options,
                    optimization_problem.get());
        } else
        {
            map_optimization::solve(
                    solver_options_with_callbacks, optimization_problem.get());
        }

        if (plotter_ != nullptr) {
            plotter_->visualizeMap(*map);
        }
        return true;
    }

}  // namespace map_optimization
