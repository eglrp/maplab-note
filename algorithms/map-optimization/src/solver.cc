#include "map-optimization/solver.h"

#include <ceres-error-terms/problem-information.h>
#include <ceres/ceres.h>

namespace map_optimization {

ceres::TerminationType solve(
    const ceres::Solver::Options& solver_options,//优化选项
    map_optimization::OptimizationProblem* optimization_problem)//待优化问题
{
    CHECK_NOTNULL(optimization_problem);

    ceres::Problem problem(ceres_error_terms::getDefaultProblemOptions());//对问题进行初始化
    ceres_error_terms::buildCeresProblemFromProblemInformation(
            optimization_problem->getProblemInformationMutable(), &problem);

    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);//优化

    optimization_problem->getOptimizationStateBufferMutable()//得到优化后的所有状态量
            ->copyAllStatesBackToMap(optimization_problem->getMapMutable());

    LOG(INFO) << summary.FullReport();
    return summary.termination_type;
}

}  // namespace map_optimization
