#ifndef VI_MAP_LOOP_CONSTRAINT_H_
#define VI_MAP_LOOP_CONSTRAINT_H_
#include <vector>

#include <aslam/frames/visual-nframe.h>

#include "vi-map/landmark.h"
#include "vi-map/unique-id.h"
#include "vi-map/vertex.h"

namespace vi_map {
struct FrameKeyPointToStructureMatch {
  vi_map::KeypointIdentifier keypoint_id_query;//存储的是当前正在匹配的特征点的信息
  vi_map::VisualFrameIdentifier keyframe_id_result;//这个是它找到的可能有匹配的这一帧
  vi_map::LandmarkId landmark_result;//地图点的id
  bool operator==(const FrameKeyPointToStructureMatch& other) const
  {
    bool result = true;
    result &= keypoint_id_query == other.keypoint_id_query;
    result &= keyframe_id_result == other.keyframe_id_result;
    result &= landmark_result == other.landmark_result;
    return result;
  }
  bool isValid() const {
    return keypoint_id_query.isValid() && keyframe_id_result.isValid() &&
           landmark_result.isValid();
  }
};
//类似Match的数据结构
struct VertexKeyPointToStructureMatch
{
    VertexKeyPointToStructureMatch()
            : keypoint_index_query(-1),
              frame_index_query(-1),
              appearance(vi_map::Landmark::kInvalidAppearance) {}//构造函数默认不出现
    VertexKeyPointToStructureMatch(
            unsigned int keypoint_index_query_, unsigned int frame_index_query,
            const vi_map::LandmarkId& landmark_result_)
            : keypoint_index_query(keypoint_index_query_),
              frame_index_query(frame_index_query),
              landmark_result(landmark_result_),
              appearance(vi_map::Landmark::kInvalidAppearance) {}
    VertexKeyPointToStructureMatch(
            unsigned int keypoint_index_query_, unsigned int frame_index_query,
            const vi_map::LandmarkId& landmark_result_, int appearance_)
            : keypoint_index_query(keypoint_index_query_),
              frame_index_query(frame_index_query),
              landmark_result(landmark_result_),
              appearance(appearance_) {}
    VertexKeyPointToStructureMatch(
            unsigned int keypoint_index_query_, unsigned int frame_index_query,
            const vi_map::LandmarkId& landmark_result_, int appearance_,
            const vi_map::VisualFrameIdentifier& frame_identifier_result_)
            : keypoint_index_query(keypoint_index_query_),
              frame_index_query(frame_index_query),
              landmark_result(landmark_result_),
              appearance(appearance_),
              frame_identifier_result(frame_identifier_result_) {}
    unsigned int keypoint_index_query;//这个地图点对应的特征点在这帧中的id
    unsigned int frame_index_query;//正在检测闭环的那个相机
    vi_map::LandmarkId landmark_result;//重定位检测到的闭环的地图点id
    int appearance;
    vi_map::VisualFrameIdentifier frame_identifier_result;//可能有匹配的这一帧的索引，这一帧也能看到这个地图点
    bool operator==(const VertexKeyPointToStructureMatch& other) const {
        bool result = true;
        result &= keypoint_index_query == other.keypoint_index_query;
        result &= frame_index_query == other.frame_index_query;
        result &= landmark_result == other.landmark_result;
        result &= appearance == other.appearance;
        result &= frame_identifier_result == other.frame_identifier_result;
        return result;
    }
};
typedef std::vector<VertexKeyPointToStructureMatch>
    VertexKeyPointToStructureMatchList;

// Returns true, if there are two structure matches present in the given
// list of structure matches for the same frame_idx + keypoint_idx pair.
inline bool doStructureMatchesContainNonExclusiveKeypoints(
    const VertexKeyPointToStructureMatchList& structure_matches) {
  pose_graph::VertexId invalid_vertex_id;
  std::unordered_set<vi_map::KeypointIdentifier> keypoint_identifiers;
  for (const vi_map::VertexKeyPointToStructureMatch& structure_match :
       structure_matches) {
    if (!keypoint_identifiers
             .emplace(
                 vi_map::VisualFrameIdentifier(
                     invalid_vertex_id, structure_match.frame_index_query),
                 structure_match.keypoint_index_query)
             .second) {
      LOG(ERROR) << "Duplicate structure-match for frame "
                 << structure_match.frame_index_query << " and keypoint "
                 << structure_match.keypoint_index_query;
      return true;
    }
  }
  return false;
}

struct LoopClosureConstraint {
  pose_graph::VertexId query_vertex_id;
  VertexKeyPointToStructureMatchList structure_matches;
};
typedef std::vector<LoopClosureConstraint> LoopClosureConstraintVector;

struct LocalizationConstraint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  LocalizationConstraint() = default;
  LocalizationConstraint(
      const aslam::NFramesId& nframe_id_,
      const VertexKeyPointToStructureMatchList& structure_matches_)
      : nframe_id(nframe_id_), structure_matches(structure_matches_) {}
  ~LocalizationConstraint() = default;
  aslam::NFramesId nframe_id;
  VertexKeyPointToStructureMatchList structure_matches;
};
typedef std::vector<LocalizationConstraint> LocalizationConstraints;

}  // namespace vi_map
#endif  // VI_MAP_LOOP_CONSTRAINT_H_
