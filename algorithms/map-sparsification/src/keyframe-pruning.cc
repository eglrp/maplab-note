#include "map-sparsification/keyframe-pruning.h"

#include <memory>
#include <vector>

#include <Eigen/Core>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <posegraph/unique-id.h>
#include <vi-map-helpers/vi-map-queries.h>
#include <vi-map/vi-map.h>

DEFINE_double(kf_distance_threshold_m, 0.75,
              "Distance threshold to add a new keyframe [m].");
DEFINE_double(kf_rotation_threshold_deg, 20,
              "Rotation threshold to add a new keyframe [deg].");
DEFINE_uint64(kf_every_nth_vertex, 10, "Force a keyframe every n-th vertex.");
DEFINE_uint64(kf_min_shared_landmarks_obs, 20,
              "Coobserved landmark number to add a new keyframe.");

namespace map_sparsification {
    namespace {
        size_t removeAllVerticesBetweenTwoVertices(
                pose_graph::Edge::EdgeType traversal_edge,
                const pose_graph::VertexId& start_kf_id,
                const pose_graph::VertexId& end_kf_id,
                vi_map::VIMap* map)
        {
            CHECK_NOTNULL(map);
            CHECK(traversal_edge != pose_graph::Edge::EdgeType::kUndefined);
            CHECK(start_kf_id.isValid());
            CHECK(end_kf_id.isValid());

            size_t num_merged_vertices = 0u;
            pose_graph::VertexId current_vertex_id;
            while (map->getNextVertex(start_kf_id, traversal_edge, &current_vertex_id) &&
                   current_vertex_id != end_kf_id) {//从图里找到和开始关键帧相连的下一帧，只要这帧不是结束关键帧
                CHECK(current_vertex_id.isValid());
                CHECK(start_kf_id != current_vertex_id);
                //去掉这个关键帧节点之后的那个非关键帧节点
                //1.将这个节点首次观测的地图点汇合到前一个关键帧节点。
                //2.把这个节点连接的出入边都删掉
                //3.把地图点关于这个节点的观测信息都删除后
                //4.在位姿图里去掉这个节点
                map->mergeNeighboringVertices(start_kf_id, current_vertex_id);
                ++num_merged_vertices;
            }
            return num_merged_vertices;
        }
    }  // namespace

    KeyframingHeuristicsOptions//在生成重定位地图的工作流程中所用的关键帧选择
    KeyframingHeuristicsOptions::initializeFromGFlags() {
        KeyframingHeuristicsOptions options;
        options.kf_distance_threshold_m = FLAGS_kf_distance_threshold_m;
        options.kf_rotation_threshold_deg = FLAGS_kf_rotation_threshold_deg;
        options.kf_every_nth_vertex = FLAGS_kf_every_nth_vertex;
        options.kf_min_shared_landmarks_obs = FLAGS_kf_min_shared_landmarks_obs;
        return options;
    }

    void KeyframingHeuristicsOptions::serialize(
            proto::KeyframingHeuristicsOptions* proto) const {
        CHECK_NOTNULL(proto);
        proto->set_kf_distance_threshold_m(kf_distance_threshold_m);
        proto->set_kf_rotation_threshold_deg(kf_rotation_threshold_deg);
        proto->set_kf_every_nth_vertex(kf_every_nth_vertex);
        proto->set_kf_min_shared_landmarks_obs(kf_min_shared_landmarks_obs);
    }

    void KeyframingHeuristicsOptions::deserialize(
            const proto::KeyframingHeuristicsOptions& proto) {
        CHECK(proto.has_kf_distance_threshold_m());
        kf_distance_threshold_m = proto.kf_distance_threshold_m();
        CHECK(proto.has_kf_rotation_threshold_deg());
        kf_rotation_threshold_deg = proto.kf_rotation_threshold_deg();
        CHECK(proto.has_kf_every_nth_vertex());
        kf_every_nth_vertex = proto.kf_every_nth_vertex();
        CHECK(proto.has_kf_min_shared_landmarks_obs());
        kf_min_shared_landmarks_obs = proto.kf_min_shared_landmarks_obs();
    }

    //启发式搜索关键帧，在起始帧和终止帧之间挑选需要的关键帧
    size_t selectKeyframesBasedOnHeuristics(
            const vi_map::VIMap& map, const pose_graph::VertexId& start_keyframe_id,
            const pose_graph::VertexId& end_vertex_id,
            const KeyframingHeuristicsOptions& options,
            std::vector<pose_graph::VertexId>* selected_keyframes)
            {
        CHECK_NOTNULL(selected_keyframes)->clear();
        CHECK(start_keyframe_id.isValid());
        CHECK(end_vertex_id.isValid());

        const vi_map::MissionId& mission_id =//当前任务的id
                map.getMissionIdForVertex(start_keyframe_id);
        CHECK_EQ(mission_id, map.getMissionIdForVertex(end_vertex_id));
        //在Edge.h中定义边的类型，比如这是一个vi里程计信息的边，或者这是一个闭环信息的边等等
        pose_graph::Edge::EdgeType backbone_type =
                map.getGraphTraversalEdgeType(mission_id);

        pose_graph::VertexId last_keyframe_id = start_keyframe_id;
        pose_graph::VertexId current_vertex_id = start_keyframe_id;//当前节点id
        size_t num_frames_since_last_keyframe = 0u;//插入上一个关键帧到这帧之间的帧数间隔

        //lambda定义插入关键帧函数
        auto insert_keyframe = [&](const pose_graph::VertexId& keyframe_id) {
            CHECK(keyframe_id.isValid());
            num_frames_since_last_keyframe = 0u;//帧数间隔归0
            last_keyframe_id = keyframe_id;//将当前插入的关键帧id设成最近关键帧id
            selected_keyframes->emplace_back(keyframe_id);//插入选择的关键帧序列
        };

        // The first vertex in the range is always a keyframe.
        //第一个节点无条件的认为是关键帧
        insert_keyframe(start_keyframe_id);

        // Traverse the posegraph and select keyframes to keep based on the following
        // conditions.
        // The ordering of the condition evaluation is important.
        //   - max. temporal spacing (force a keyframe every n-th frame)
        //   - common landmark/track count
        //   - distance and rotation threshold
        // TODO(schneith): Online-viwls should add special vio constraints (rot. only,
        // stationary) and avoid keyframes during these states.
        //根据以下条件遍历posegraph并选择关键帧。
        //maplab认为条件评估的顺序很重要，也就是说要先保证时间间隔，再是共视数目，然后再是距离。
        // - 时间间隔(每n帧强制一个关键帧)
        // -共视的地图点
        // -距离和旋转阈值
        //TODO(schneith):在线viwls应该添加特殊的vio约束(只固定旋转)，并在这些状态下避免关键帧。
        //得到相同边类型的下一个节点
        while (map.getNextVertex(current_vertex_id, backbone_type,
                                 &current_vertex_id))
        {
            // Add a keyframe every n-th frame.
            //如果和上一帧的关键帧之间间隔的帧数太远（kf_every_nth_vertex）就需要直接添加关键帧
            if (num_frames_since_last_keyframe >= options.kf_every_nth_vertex) {
                insert_keyframe(current_vertex_id);
                continue;
            }

            // Insert a keyframe if the common landmark observations between the last
            // keyframe and this current vertex frame drop below a certain threshold.
            vi_map_helpers::VIMapQueries queries(map);
            //检测最近关键帧节点和当前帧共视到的地图点的数量，小于一个阈值，也要作为关键帧
            const size_t common_observations =
                    queries.getNumberOfCommonLandmarks(current_vertex_id, last_keyframe_id);
            if (common_observations < options.kf_min_shared_landmarks_obs) {
                insert_keyframe(current_vertex_id);
                continue;
            }

            // Select keyframe if the distance or rotation to the last keyframe exceeds
            // a threshold.
            //得到最近关键帧和当前帧在世界坐标系下的本体位姿
            aslam::Transformation T_M_Bkf = map.getVertex_T_G_I(last_keyframe_id);
            aslam::Transformation T_M_Bi = map.getVertex_T_G_I(current_vertex_id);

            aslam::Transformation T_Bkf_Bi = T_M_Bkf.inverse() * T_M_Bi;//最近关键帧本地坐标系下观测到的当前帧本体坐标的位姿
            //计算相对距离和旋转角
            const double distance_to_last_keyframe_m = T_Bkf_Bi.getPosition().norm();
            const double rotation_to_last_keyframe_rad =
                    aslam::AngleAxis(T_Bkf_Bi.getRotation()).angle();
            //如果运动过大（距离和角度有一个过大），就需要加入关键帧
            if (distance_to_last_keyframe_m >= options.kf_distance_threshold_m ||
                rotation_to_last_keyframe_rad >=
                options.kf_rotation_threshold_deg * kDegToRad) {
                insert_keyframe(current_vertex_id);
                continue;
            }

            ++num_frames_since_last_keyframe;
        }

        return selected_keyframes->size();
    }

    size_t removeVerticesBetweenKeyframes(//将没有关键帧的节点进行剔除
            const pose_graph::VertexIdList& keyframe_ids, vi_map::VIMap* map) {
        // Nothing to remove if there are less than two keyframes.
        if (keyframe_ids.size() < 2) {
            return 0u;
        }
        //提取当前的任务，和这个图的边的类型
        const vi_map::MissionId& mission_id =
                map->getMissionIdForVertex(keyframe_ids.front());
        pose_graph::Edge::EdgeType traversal_edge =
                map->getGraphTraversalEdgeType(mission_id);

        size_t num_removed_vertices = 0u;
        for (size_t idx = 0u; idx < keyframe_ids.size() - 1; ++idx)
        {
            CHECK_EQ(mission_id, map->getMissionIdForVertex(keyframe_ids[idx + 1]))
                    << "All keyframes must be of the same mission.";
            //将这两个关键帧之间的节点都进行剔除
            num_removed_vertices += removeAllVerticesBetweenTwoVertices(
                    traversal_edge, keyframe_ids[idx], keyframe_ids[idx + 1], map);
        }
        return num_removed_vertices;
    }

}  // namespace map_sparsification
