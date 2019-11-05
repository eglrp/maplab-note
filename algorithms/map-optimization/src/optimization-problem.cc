#include "map-optimization/optimization-problem.h"

#include <memory>
#include <sstream>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ceres-error-terms/parameterization/pose-param-jpl.h>
#include <ceres-error-terms/problem-information.h>
#include <vi-map-helpers/mission-clustering-coobservation.h>
#include <vi-map/vi-map.h>

#include "map-optimization/mission-cluster-gauge-fixes.h"
#include "map-optimization/optimization-state-buffer.h"

namespace map_optimization {
namespace {
void fixOpenDoFOfInitialVertex(//对优化节点中的第一个节点进行fix，并且根据要fix替换对应的局部参数化
    double* first_pose_q_IM__M_p_MI_JPL_state,
    double* baseframe_q_GM__G_p_GM_JPL_state,
    FixedRotationDoF rotation_dof_fixed, bool position_dof_fixed,
    ceres_error_terms::ProblemInformation* problem_info)
{
    CHECK_NOTNULL(first_pose_q_IM__M_p_MI_JPL_state);
    CHECK_NOTNULL(baseframe_q_GM__G_p_GM_JPL_state);
    CHECK_NOTNULL(problem_info);

    // Assemble a parameterization for the first vertex.
    ceres::LocalParameterization* position_parametrization = nullptr;
    if (position_dof_fixed) //如果要固定住pose
    {
        position_parametrization =
                new ceres::SubsetParameterization(3, std::vector<int>{0, 1, 2});
        //将这三维设置成const（SubsetParameterization是用来设置不去优化参数中某一个变量的）
    } else {
        position_parametrization = new ceres::IdentityParameterization(3);//全部优化
    }

    ceres::LocalParameterization* rotation_parametrization = nullptr;
    switch (rotation_dof_fixed) {
        case FixedRotationDoF::kAll: {//如果4维都要fix,因为JPL形式的四元数，优化时是左乘扰动，所以这里定义了对应左乘PLUS的JPL形四元数的局部参数化
            rotation_parametrization =
                    new ceres_error_terms::JplQuaternionParameterization;
            problem_info->setParameterBlockConstant(//固定
                    first_pose_q_IM__M_p_MI_JPL_state);
            break;
        }
        case FixedRotationDoF::kYaw: {
            Eigen::Map<Eigen::Matrix<double, 7, 1>> q_GM__G_p_GM_JPL(
                    baseframe_q_GM__G_p_GM_JPL_state);
            rotation_parametrization =
                    new ceres_error_terms::JplRollPitchQuaternionParameterization(//只对roll，pitch有扰动
                            q_GM__G_p_GM_JPL.head<4>());
            problem_info->setParameterBlockVariable(
                    first_pose_q_IM__M_p_MI_JPL_state);//first_pose_q_IM__M_p_MI_JPL_state是可以变动的

            // This parameterization requires the baseframe to be constant.
            problem_info->setParameterBlockConstantIfPartOfTheProblem(//基帧看成常数去优化
                    baseframe_q_GM__G_p_GM_JPL_state);
            break;
        }
        case FixedRotationDoF::kNone: {//什么都不fix的化，first_pose_q_IM__M_p_MI_JPL_state是可以变动的
            rotation_parametrization =
                    new ceres_error_terms::JplQuaternionParameterization;
            problem_info->setParameterBlockVariable(
                    first_pose_q_IM__M_p_MI_JPL_state);
            break;
        }
        default:
            LOG(FATAL);
    }
    std::shared_ptr<ceres::LocalParameterization> first_pose_param(//如果我要对激光位姿去用这种pose，也是没问题的
            new ceres::ProductParameterization(
                    rotation_parametrization, position_parametrization));
    problem_info->replaceParameterization(
            first_pose_q_IM__M_p_MI_JPL_state, first_pose_param);
    //将first_pose_q_IM__M_p_MI_JPL_state的局部参数化替换成first_pose_param
}
}  // namespace

//优化问题的构造
    OptimizationProblem::OptimizationProblem(
            vi_map::VIMap* map, const vi_map::MissionIdSet& mission_ids)
            : map_(CHECK_NOTNULL(map)),
              missions_ids_(mission_ids),//存储所有任务
              mission_coobservation_clusters_(
             //mission_coobservation_clusters_里就是一个个簇，每个簇里的任务之间是有共视的，但是簇和簇之间的任务是没有共视的
                      vi_map_helpers::clusterMissionByLandmarkCoobservations(
                              *map, mission_ids))
    {
    //将节点位姿，基准帧位姿，相机外参等保存到state_buffer_中，q都是JPL格式，因为后面的误差推导是用JPL形式来推导的
        state_buffer_.importStatesOfMissions(*map, mission_ids);

        // Initialize the parameterizations.
        //局部参数化
        local_parameterizations_.pose_parameterization.reset(
                new ceres_error_terms::JplPoseParameterization);
        local_parameterizations_.baseframe_parameterization.reset(
                new ceres_error_terms::JplYawOnlyPoseParameterization);
        local_parameterizations_.quaternion_parameterization.reset(
                new ceres_error_terms::JplQuaternionParameterization);
    }

    //对首帧进行fix
void OptimizationProblem::applyGaugeFixesForInitialVertices(
    const std::vector<MissionClusterGaugeFixes>& new_cluster_fixes)//所有簇任务的控制第一个节点fix的策略
{
    CHECK_EQ(new_cluster_fixes.size(), mission_coobservation_clusters_.size());

    std::stringstream message;
    message << "VI gauge fixes: \n"
            << "  num mission co-observability clusters: "
            << new_cluster_fixes.size() << "\n";

    for (size_t cluster_idx = 0u; cluster_idx < new_cluster_fixes.size();//遍历所有的簇任务
         ++cluster_idx)
    {
        const MissionClusterGaugeFixes& new_cluster_fix =//当前簇任务集合的fix策略
                new_cluster_fixes[cluster_idx];
        const vi_map::MissionIdSet& missionids_of_cluster =//当前簇任务集合
                mission_coobservation_clusters_[cluster_idx];

        // Get the first vertex of the first mission of the cluster (which is also
        // part of the problem) and fix the open degrees of freedom.
        //因为fix是针对第一个任务的第一个节点（当然这个节点是要在优化中存在的）
        //所以先提取簇任务集合的第一个任务
        const vi_map::MissionId& first_mission_id = *missionids_of_cluster.begin();

        //得到根节点id
        pose_graph::VertexId current_vertex_id =
                map_->getMission(first_mission_id).getRootVertexId();
        pose_graph::VertexId first_vertex_id_in_problem;
        do {
            CHECK(current_vertex_id.isValid());
            //如果这个节点在优化中存在，就认为这是这个簇任务的第一个优化节点，保存id
            //如果在优化中不存在，那么就查看下一个节点
            if (problem_books_.keyframes_in_problem.count(current_vertex_id) > 0u)
            {
                first_vertex_id_in_problem = current_vertex_id;
                break;
            }
        } while (map_->getNextVertex(//直到遍历不到下一个节点
                current_vertex_id, map_->getGraphTraversalEdgeType(first_mission_id),
                &current_vertex_id));
        CHECK(first_vertex_id_in_problem.isValid());
        fixOpenDoFOfInitialVertex(//对优化节点中的第一个节点进行fix，并且根据要fix替换对应的局部参数化
                state_buffer_.get_vertex_q_IM__M_p_MI_JPL(first_vertex_id_in_problem),//优化中的第一个节点的位姿
                state_buffer_.get_baseframe_q_GM__G_p_GM_JPL(//第一个任务（也就是第一个节点所在任务）基准帧的位姿
                        map_->getMissionBaseFrameForMission(first_mission_id).id()),
                new_cluster_fix.rotation_dof_fixed, new_cluster_fix.position_dof_fixed,//fix选项
                &problem_information_);

        message << "  cluster " << cluster_idx << ":\n"
                << "    missions of cluster: "
                << printIdContainer(missionids_of_cluster) << "\n"
                << "    1st mission: " << first_mission_id << "\n"
                << "    1st vertex: " << first_vertex_id_in_problem << "\n"
                << "    position-fixed: " << new_cluster_fix.position_dof_fixed
                << "\n"
                << "    rotation-fixed: " << new_cluster_fix.rotation_dof_fixed
                << "\n"
                << "    scale-fixed: " << new_cluster_fix.scale_fixed << "\n";

        // Fix scale of mission cluster by fixing a landmark expressed in the
        // vertex of the first vertex of the first mission in the cluster.
        //通过固定在集群中第一个任务的第一个顶点的顶点上表示的地标来确定任务集群的规模。
        if (new_cluster_fix.scale_fixed)
        {
            const vi_map::LandmarkStore& landmark_store_first_vertex =//得到所有被第一个节点观测到的地图点
                    map_->getVertex(first_vertex_id_in_problem).getLandmarks();

            vi_map::LandmarkId first_landmark_of_first_mission;
            for (const vi_map::Landmark& landmark : landmark_store_first_vertex)
            {//得到第一个在优化里的地图点
                if (problem_books_.landmarks_in_problem.count(landmark.id()) > 0u) {
                    first_landmark_of_first_mission = landmark.id();
                    break;
                }
            }
            // TODO(schneith): Loop over the vertices if the first one does not
            // see any landmarks.
            CHECK(first_landmark_of_first_mission.isValid())
            << "The first vertex has no landmarks. This case is not supported "
            << "right now. Consider extending this function.";

            problem_information_.setParameterBlockConstantIfPartOfTheProblem(
                    map_->getLandmark(first_landmark_of_first_mission).get_p_B_Mutable());
        }
    }
    if (problem_books_.cluster_gauge_fixes_initial_vertex == nullptr) {
        problem_books_.cluster_gauge_fixes_initial_vertex.reset(
                new std::vector<MissionClusterGaugeFixes>(
                        mission_coobservation_clusters_.size()));
    }
    LOG(INFO) << message.str();

    *problem_books_.cluster_gauge_fixes_initial_vertex = new_cluster_fixes;//将这次的fix策略赋值给cluster_gauge_fixes_initial_vertex
}

}  // namespace map_optimization
