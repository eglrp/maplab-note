#include "map-sparsification-plugin/keyframe-pruning.h"

#include <string>
#include <vector>

#include <console-common/console.h>
#include <map-manager/map-manager.h>
#include <map-sparsification/keyframe-pruning.h>
#include <vi-map/vi-map.h>
#include <visualization/viwls-graph-plotter.h>

namespace map_sparsification_plugin {

    int keyframeMapBasedOnHeuristics(//基于启发式的关键帧选择
            const map_sparsification::KeyframingHeuristicsOptions& options,
            const vi_map::MissionId& mission_id,
            visualization::ViwlsGraphRvizPlotter* plotter, vi_map::VIMap* map)
    {
        // plotter is optional.
        CHECK_NOTNULL(map);
        CHECK(mission_id.isValid());
        //这个任务下所有节点的数目
        const size_t num_initial_vertices = map->numVerticesInMission(mission_id);
        //找到这个任务的根节点
        pose_graph::VertexId root_vertex_id =
                map->getMission(mission_id).getRootVertexId();
        CHECK(root_vertex_id.isValid());
        //找到这个任务的最后一个节点
        pose_graph::VertexId last_vertex_id =
                map->getLastVertexIdOfMission(mission_id);

        // Select keyframes along the mission. Unconditionally add the last vertex as
        // a keyframe if it isn't a keyframe already.
        //沿着任务选择关键帧，无条件地添加最后一个顶点作为关键帧
        pose_graph::VertexIdList keyframe_ids;
        //输入的是地图，根节点id，最后一个节点id，选择关键帧的选项，引用形式传入关键帧id
        map_sparsification::selectKeyframesBasedOnHeuristics(
                *map, root_vertex_id, last_vertex_id, options, &keyframe_ids);
        if (keyframe_ids.empty()) {
            LOG(ERROR) << "No keyframes found.";
            return common::CommandStatus::kUnknownError;
        }

        //如果最后一帧不是关键帧，那么就要把最后一帧也当成关键帧插入
        if (keyframe_ids.back() != last_vertex_id) {
            keyframe_ids.emplace_back(last_vertex_id);
        }

        // Optionally, visualize the selected keyframes.
        //可视化操作，将选择的关键帧标出来
        if (plotter != nullptr) {
            std::vector<pose_graph::VertexIdList> partitions;
            partitions.emplace_back(keyframe_ids);
            plotter->plotPartitioning(*map, partitions);
            LOG(INFO) << "Selected " << keyframe_ids.size() << " keyframes of "
                      << num_initial_vertices << " vertices.";
        }

        // Remove non-keyframe vertices.
        //将两个关键帧之间的普通帧节点进行删除：遍历前一个关键帧的下一个节点，都和前关键帧进行合并，直到合并到后一个关键帧
        const size_t num_removed_keyframes =
                map_sparsification::removeVerticesBetweenKeyframes(keyframe_ids, map);
        LOG(INFO) << "Removed " << num_removed_keyframes << " vertices of "
                  << num_initial_vertices << " vertices.";
        return common::CommandStatus::kSuccess;
    }

}  // namespace map_sparsification_plugin
