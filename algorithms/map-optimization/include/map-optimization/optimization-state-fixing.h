#ifndef MAP_OPTIMIZATION_OPTIMIZATION_STATE_FIXING_H_
#define MAP_OPTIMIZATION_OPTIMIZATION_STATE_FIXING_H_

#include <vi-map/unique-id.h>

#include "map-optimization/optimization-problem.h"

namespace map_optimization {
inline void fixAllBaseframesInProblem(OptimizationProblem* problem)
{//这里把baseframe的位姿都认为是常数，不然没法进行对于roll，pitch的扰动
  CHECK_NOTNULL(problem);
  for (const vi_map::MissionId& mission_id : problem->getMissionIds()) //遍历所有的任务
  {
    double* baseframe_state =
        problem->getOptimizationStateBufferMutable()
            ->get_baseframe_q_GM__G_p_GM_JPL(//找到当前任务对应的基帧的位姿
                problem->getMapMutable()
                    ->getMissionBaseFrameForMission(mission_id)
                    .id());
    CHECK_NOTNULL(baseframe_state);
    problem->getProblemInformationMutable()
        ->setParameterBlockConstantIfPartOfTheProblem(baseframe_state);//固定急诊位姿
  }
}


    inline void fixAllPoseInProblem(OptimizationProblem* problem)
    {
        CHECK_NOTNULL(problem);

        for (const vi_map::MissionId& mission_id : problem->getMissionIds()) //遍历所有的任务
        {
            pose_graph::VertexIdList relevant_vertex_ids;
            problem->getMapMutable()->getAllVertexIdsInMissionAlongGraph(mission_id,&relevant_vertex_ids);//通过图，得到这个任务的所有的节点

            for (int i = 0; i <relevant_vertex_ids.size() ; ++i)
            {
                const pose_graph::VertexId &observer_id = relevant_vertex_ids[i];//得到当前观测点对应的节点id
                CHECK(problem->getMapMutable()->hasVertex(observer_id))
                << "Observer " << observer_id << " of store landmark ";

                double* Vertex_state =problem->getOptimizationStateBufferMutable()->get_vertex_q_IM__M_p_MI_JPL(observer_id);

                CHECK_NOTNULL(Vertex_state);
                problem->getProblemInformationMutable()
                        ->setParameterBlockConstantIfPartOfTheProblem(Vertex_state);//固定急诊位姿

            }

        }
    }


}  // namespace map_optimization

#endif  // MAP_OPTIMIZATION_OPTIMIZATION_STATE_FIXING_H_
