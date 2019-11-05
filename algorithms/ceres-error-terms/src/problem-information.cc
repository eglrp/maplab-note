#include "ceres-error-terms/problem-information.h"

#include <map>
#include <vector>

#include <ceres/problem.h>
#include <glog/logging.h>
#include <maplab-common/accessors.h>

#include "ceres-error-terms/common.h"

namespace ceres_error_terms {

constexpr int ProblemInformation::kDefaultParameterBlockId;

ceres::Problem::Options getDefaultProblemOptions()
{
  ceres::Problem::Options options;
  options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  options.local_parameterization_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  return options;
}

void buildOrderedCeresProblemFromProblemInformation(
    ProblemInformation* problem_information,
    const std::vector<int>& groupid_ordering, int* ordered_group_start_index,
    ceres::Problem* problem)
{
    CHECK_NOTNULL(problem_information);
    CHECK_NOTNULL(problem);
    CHECK_NOTNULL(ordered_group_start_index);

    std::vector<double*> parameter_blocks;
    problem->GetParameterBlocks(&parameter_blocks);//得到所有待优化参数
    CHECK(parameter_blocks.empty())
    << "Problem passed to buildProblem must be empty.";

    VLOG(3) << "Build problem from "
            << problem_information->residual_blocks.size()
            << " active residual blocks.";

    // Add all parameter blocks to the problem. Usually this step is not
    // required as all blocks are added along with the residual blocks.
    // Here this is done manually to have control over the implicit parameter
    // ordering determined by the underlying parameter vector.
    //这步不需要,跳过，groupid_ordering基本都是空的情况
    if (!groupid_ordering.empty())
    {
        VLOG(3) << "Enforcing a special groupid ordering using "
                << groupid_ordering.size() << " groups.";

        struct ParamBlockAndSize {
            ParamBlockAndSize(double* _param_block, size_t _block_size)
                    : param_block(_param_block), block_size(_block_size) {
                CHECK_NOTNULL(_param_block);
                CHECK_GT(_block_size, 0u);
            }
            double* param_block;
            size_t block_size;

            inline bool operator<(const ParamBlockAndSize& other) const {
                return (param_block < other.param_block);
            }
        };
        typedef std::map<int, std::set<ParamBlockAndSize>> GroupIdParamBlockSetMap;
        GroupIdParamBlockSetMap parameter_blocks_with_ordering;
        std::unordered_set<double*> inserted_param_blocks;

        // Go over all residual blocks and add the associated parameter blocks
        // if it is not assigned to a groupid with associated ordering. Keep
        // all residual blocks with associated ordering for later insertion.
        *ordered_group_start_index = 0;
        for (ProblemInformation::ResidualInformationMap::value_type&
                    residual_information_item : problem_information->residual_blocks) {
            ResidualInformation& residual_information =
                    residual_information_item.second;

            if (!residual_information.active_) {
                continue;
            }

            // Go over all parameter blocks in this residual block.
            const size_t num_param_blocks =
                    residual_information.parameter_blocks.size();

            for (size_t block_idx = 0; block_idx < num_param_blocks; ++block_idx) {
                double* parameter_block =
                        CHECK_NOTNULL(residual_information.parameter_blocks[block_idx]);
                const int block_size = residual_information.cost_function
                        ->parameter_block_sizes()[block_idx];

                int parameter_block_groupid;
                const bool parameter_block_has_groupid =
                        problem_information->getParameterBlockGroupId(
                                parameter_block, &parameter_block_groupid);

                // Add block if no special ordering is requested, otherwise queue them
                // for later insertion.
                if (!parameter_block_has_groupid ||
                    !common::containsValue(groupid_ordering, parameter_block_groupid)) {
                    // Make sure that we only add each parameter block once.
                    if (inserted_param_blocks.emplace(parameter_block).second) {
                        problem->AddParameterBlock(parameter_block, block_size);
                        *ordered_group_start_index += block_size;
                    }
                } else {
                    // Keep the parameter blocks with special ordering requirements.
                    parameter_blocks_with_ordering[parameter_block_groupid].emplace(
                            ParamBlockAndSize(parameter_block, block_size));
                }
            }
        }

        // Now add all parameter blocks with special ordering in ascending groupid
        // ordering.
        for (const GroupIdParamBlockSetMap::value_type& groupid_paramblockset :
                parameter_blocks_with_ordering) {
            for (const ParamBlockAndSize& paramblock_size :
                    groupid_paramblockset.second) {
                problem->AddParameterBlock(
                        paramblock_size.param_block, paramblock_size.block_size);
                inserted_param_blocks.emplace(paramblock_size.param_block);
            }
        }
    } else
    {
        // No parameters are in the calibration group.
        *ordered_group_start_index = -1;
    }

    // Add all residual blocks to the ceres Problem.
    for (ProblemInformation::ResidualInformationMap::value_type&
                residual_information_item : problem_information->residual_blocks) //遍历costfun映射
    {
        ResidualInformation& residual_information =
                residual_information_item.second;
        if (!residual_information.active_) {//如果这个代价函数被禁用了才会跳过，不过一般不会被禁用
            continue;
        }
        //之前只是用ResidualInformation暂时保存了这些信息，这里正式加到ceres的残差块里
        ceres::ResidualBlockId residual_block_id = problem->AddResidualBlock(
                residual_information.cost_function.get(),
                residual_information.loss_function.get(),
                residual_information.parameter_blocks);//同时要优化的参数块也会被添加进来
        residual_information.latest_residual_block_id = residual_block_id;//保存这个残差块的索引
    }

    // Set specified parameter block constant.
    for (double* value : problem_information->constant_parameter_blocks)
    {
        if (problem->HasParameterBlock(value))
        {//不去优化这个参数块
            problem->SetParameterBlockConstant(value);
        } else {
            LOG(WARNING)
                    << "Parameter block " << value << " is in the constant "
                    << "blocks of the problem information, but it is not present "
                    << "in the ceres problem, which means it was not present in the "
                    << "active residual blocks of the problem information.";
        }
    }

    for (const ProblemInformation::ParameterBoundMap::value_type& bound_info :
            problem_information->parameter_bounds)// 这个在闭环前的优化是用不到的，只有闭环以后才会去添加这个
    {
        if (problem->HasParameterBlock(bound_info.first))
        {
            problem->SetParameterLowerBound(
                    bound_info.first, bound_info.second.index_in_param_block,
                    bound_info.second.lower_bound);
            problem->SetParameterUpperBound(
                    bound_info.first, bound_info.second.index_in_param_block,
                    bound_info.second.upper_bound);
        } else {
            LOG(WARNING)
                    << "Parameter block " << bound_info.first << " has upper/lower bound "
                    << "information associated, but it is not present "
                    << "in the ceres problem, which means it was not present in the "
                    << "active residual blocks of the problem information.";
        }
    }

    // Set local parameterizations for all parameter blocks.
    for (const std::pair<double*, std::shared_ptr<ceres::LocalParameterization>>
                parameterization : problem_information->parameterizations) {
        if (problem->HasParameterBlock(parameterization.first)) {
            problem->SetParameterization(//每一个子参数块对应的局部参数化
                    parameterization.first, parameterization.second.get());
        } else {
            LOG(WARNING)
                    << "Parameter block " << parameterization.first
                    << " has a parametrization, but the block is not present "
                    << "in the ceres problem, which means it was not present in the "
                    << "active residual blocks of the problem information.";
        }
    }
}

void buildCeresProblemFromProblemInformation(//构建ceres优化问题
    ProblemInformation* problem_information, ceres::Problem* problem)
    {
  CHECK_NOTNULL(problem_information);
  CHECK_NOTNULL(problem);

  int ordered_group_start_index;
  buildOrderedCeresProblemFromProblemInformation(
      problem_information, std::vector<int>(), &ordered_group_start_index,
      problem);
}
}  // namespace ceres_error_terms
