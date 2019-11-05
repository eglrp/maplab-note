#ifndef MATCHING_BASED_LOOPCLOSURE_MATCHING_BASED_ENGINE_INL_H_
#define MATCHING_BASED_LOOPCLOSURE_MATCHING_BASED_ENGINE_INL_H_

#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <aslam/common/reader-writer-lock.h>
#include <vi-map/unique-id.h>

#include "matching-based-loopclosure/matching-based-engine.h"

namespace std {
template <>
struct hash<vi_map::FrameKeyPointToStructureMatch> {
  std::size_t operator()(
      const vi_map::FrameKeyPointToStructureMatch& value) const {
    const std::size_t h0(
        std::hash<vi_map::KeypointIdentifier>()(value.keypoint_id_query));
    const std::size_t h1(
        std::hash<vi_map::VisualFrameIdentifier>()(value.keyframe_id_result));
    const std::size_t h2(
        std::hash<vi_map::LandmarkId>()(value.landmark_result));
    return h0 ^ h1 ^ h2;
  }
};

template <>
struct hash<std::pair<vi_map::KeypointIdentifier, vi_map::LandmarkId>> {
  std::size_t operator()(
      const std::pair<vi_map::KeypointIdentifier, vi_map::LandmarkId>& value)
      const {
    const std::size_t h1(std::hash<vi_map::KeypointIdentifier>()(value.first));
    const std::size_t h2(std::hash<vi_map::LandmarkId>()(value.second));
    return h1 ^ h2;
  }
};
}  // namespace std

namespace matching_based_loopclosure {


    //IdType是loop_closure::KeyframeId
template <typename IdType>
void MatchingBasedLoopDetector::doCovisibilityFiltering(
    const loop_closure::IdToMatches<IdType>& id_to_matches_map,
    const bool make_matches_unique,//如果是单目的话，这里就应该是true;
    loop_closure::FrameToMatches* frame_matches_ptr,//loop_closure::IdToMatches<IdType>和loop_closure::FrameToMatches是一种数据结构
    std::mutex* frame_matches_mutex) const
{
    // WARNING: Do not clear frame matches. It is intended that new matches can
    // be added to already existing matches. The mutex passed to the function
    // can be nullptr, in which case locking is disabled.
    //警告:不清除帧匹配。新的匹配可以添加到已经存在的匹配中。传递给函数的互斥对象可以是nullptr，在这种情况下锁被禁用。
    CHECK_NOTNULL(frame_matches_ptr);
    loop_closure::FrameToMatches& frame_matches = *frame_matches_ptr;//这就是一个map，是一个帧的id对应着多个匹配

    typedef int ComponentId;
    constexpr ComponentId kInvalidComponentId = -1;


    //定义了一些需要的数据结构

    typedef std::unordered_map<loop_closure::Match, ComponentId>
            MatchesToComponents;//先将所有的match都设成不合格
    typedef std::unordered_map<ComponentId,std::unordered_set<loop_closure::Match>> Components;
    typedef std::unordered_map<vi_map::LandmarkId,std::vector<loop_closure::Match>> LandmarkMatches;

    const size_t num_matches_to_filter =//得到所有可能匹配的描述子的总数量，匹配用Match来表示。
            loop_closure::getNumberOfMatches(id_to_matches_map);
    if (num_matches_to_filter == 0u)
    {
        return;
    }

    MatchesToComponents matches_to_components;//key是所有的候选匹配,value是是否合法
    LandmarkMatches landmark_matches;//key是地图点的id，value是所有的候选描述子
    // To avoid rehashing, we reserve at least twice the number of elements.
    matches_to_components.reserve(num_matches_to_filter * 2u);
    // Reserving the number of matches is still conservative because the number
    // of matched landmarks is smaller than the number of matches.
    //保留匹配的数量仍然是保守的，因为匹配的地标的数量小于匹配的数量。
    landmark_matches.reserve(num_matches_to_filter);

    //遍历所有的匹配
    for (const typename loop_closure::IdToMatches<IdType>::value_type&
                id_matches_pair : id_to_matches_map)//遍历所有可能含有匹配的帧
    {
        for (const loop_closure::Match& match : id_matches_pair.second) //这些帧里面的匹配>1,遍历这些匹配
        {
            landmark_matches[match.landmark_result].emplace_back(match);
            //之所以对应的是一个vec，是因为这里记录的是可能的所有匹配，
            //比如我现在一个描述子对应了10个候选匹配描述子，但是它们都对应了一个地标点
            matches_to_components.emplace(match, kInvalidComponentId);
        }
    }



    //typedef float ScoreType;
//    template <typename IdType>
//    using IdToScoreMap = std::unordered_map<IdType, scoring::ScoreType>;

        //IdType是loop_closure::KeyframeId

    IdToScoreMap<IdType> id_to_score_map;
    //返回了得分在前n高的候选帧集合
    computeRelevantIdsForFiltering(id_to_matches_map, &id_to_score_map);

    ComponentId count_component_index = 0;
    size_t max_component_size = 0u;
    ComponentId max_component_id = kInvalidComponentId;//max_component_id初始化给的-1
    Components components;//这里的key是组号，value是组里所有的匹配，就是所有具有共视关系的都在一组。
    // 影响共视关系的就是3d点下的所有观测2d点组，以及前n个得分，n的选取

//    typedef std::unordered_map<ComponentId,std::unordered_set<loop_closure::Match>> Components;

        for (const MatchesToComponents::value_type& match_to_component ://遍历所有的匹配
            matches_to_components)
    {
        if (match_to_component.second != kInvalidComponentId)//应该就是这个匹配已经被修正过了
            continue;
        ComponentId component_id = count_component_index++;//匹配的id

        // Find the largest set of keyframes connected by landmark covisibility.
        std::queue<loop_closure::Match> exploration_queue;
        exploration_queue.push(match_to_component.first);//exploration_queue存储了所有匹配的集合
        while (!exploration_queue.empty())//直到把这个匹配的共视都给轮完了
        {
            const loop_closure::Match& exploration_match = exploration_queue.front();//存储的最头上的匹配

            if (skipMatch(id_to_score_map, exploration_match))//因为之前只取了前n个得分最高的帧，所以匹配中候选帧如果不在这些帧中的要剔除
            {
                exploration_queue.pop();
                continue;
            }

            const MatchesToComponents::iterator exploration_match_and_component =
                    matches_to_components.find(exploration_match);
            CHECK(exploration_match_and_component != matches_to_components.end());

            if (exploration_match_and_component->second == kInvalidComponentId)//刚开始都是-1
            {
                // Not part of a connected component.
                exploration_match_and_component->second = component_id;//重新赋值它的id，因为传到这里的肯定都是某一个帧中的第一个候选描述子，所以要初始化组的编号
                components[component_id].insert(exploration_match);//
                // Mark all observations (which are matches) from this ID (keyframe or
                // vertex) as visited.
                const typename loop_closure::IdToMatches<IdType>::const_iterator
                        id_and_matches =//这里存了这个匹配所在的帧的所有匹配到的描述子
                        getIteratorForMatch(id_to_matches_map, exploration_match);
                CHECK(id_and_matches != id_to_matches_map.cend());
                const std::vector<loop_closure::Match>& id_matches =
                        id_and_matches->second;
                for (const loop_closure::Match& id_match : id_matches)//遍历这帧所有的候选描述子
                {
                    matches_to_components[id_match] = component_id;//就是所有在同一帧里的描述子，都是同一个组的
                    components[component_id].insert(id_match);

                    // Put all observations of this landmark on the stack.
                    const std::vector<loop_closure::Match>& lm_matches =//这个地图点对应的所有的2d共视
                            landmark_matches[id_match.landmark_result];
                    for (const loop_closure::Match& lm_match : lm_matches)//遍里这里的匹配
                    {
                        if (matches_to_components[lm_match] == kInvalidComponentId)
                        {
                            exploration_queue.push(lm_match);
                        }
                    }
                }

                if (components[component_id].size() > max_component_size)
                {
                    max_component_size = components[component_id].size();//最大的共视组的个数
                    max_component_id = component_id;//最大的共视组的id
                }
            }
            exploration_queue.pop();
        }
    }


        //如果大过了需要最小的共视关系的数量
    // Only store the structure matches if there is a relevant amount of them.
    if (max_component_size > settings_.min_verify_matches_num)
    {
        const std::unordered_set<loop_closure::Match>& matches_max_component =
                components[max_component_id];// 共视关系最强的这一组
        typedef std::pair<loop_closure::KeypointId, vi_map::LandmarkId>
                KeypointLandmarkPair;
        std::unordered_set<KeypointLandmarkPair> used_matches;
        if (make_matches_unique) //单目会用到这个
        {
            // Conservative reserve to avoid rehashing.
            used_matches.reserve(2u * matches_max_component.size());
        }
        auto lock = (frame_matches_mutex == nullptr)
                    ? std::unique_lock<std::mutex>()
                    : std::unique_lock<std::mutex>(*frame_matches_mutex);
        for (const loop_closure::Match& structure_match : matches_max_component) //遍历这一组每一个匹配
        {
            if (make_matches_unique)
            {
                // clang-format off
                const bool is_match_unique = used_matches.emplace(//利用unordered_set数据结构来检查插入是否成功，也就是里面有没有已经重复的
                        structure_match.keypoint_id_query,//当前正要匹配的这个特征点的id，它匹配到的地图点的id
                        structure_match.landmark_result).second;
                // clang-format on
                if (!is_match_unique)
                {
                    // Skip duplicate (keypoint to landmark) structure matches.
                    continue;
                }
            }
            //typedef IdToMatches<KeyframeId> FrameToMatches;
            frame_matches[structure_match.keypoint_id_query.frame_id].push_back(//这个key是当前正要匹配的点所在的这帧的信息
                    structure_match);//匹配信息,这里的key为啥是这个
        }
    }
}

template <>
typename loop_closure::FrameToMatches::const_iterator
MatchingBasedLoopDetector::getIteratorForMatch(
    const loop_closure::FrameToMatches& frame_to_matches,
    const loop_closure::Match& match) const {
  return frame_to_matches.find(match.keyframe_id_result);
}

template <>
typename loop_closure::VertexToMatches::const_iterator
MatchingBasedLoopDetector::getIteratorForMatch(
    const loop_closure::VertexToMatches& vertex_to_matches,
    const loop_closure::Match& match) const {
  return vertex_to_matches.find(match.keyframe_id_result.vertex_id);
}

template <>
bool MatchingBasedLoopDetector::skipMatch(
    const IdToScoreMap<loop_closure::KeyframeId>& frame_to_score_map,
    const loop_closure::Match& match) const {
  const typename IdToScoreMap<loop_closure::KeyframeId>::const_iterator iter =
      frame_to_score_map.find(match.keyframe_id_result);
  return iter == frame_to_score_map.cend();
}

template <>
bool MatchingBasedLoopDetector::skipMatch(
    const IdToScoreMap<loop_closure::VertexId>& /* vertex_to_score_map */,
    const loop_closure::Match& /* match */) const {
  // We do not skip vertices because we want to consider all keyframes that
  // passed the keyframe covisibility filtering step.
  return false;
}

template <>
void MatchingBasedLoopDetector::computeRelevantIdsForFiltering(
    const loop_closure::FrameToMatches& frame_to_matches,
    IdToScoreMap<loop_closure::KeyframeId>* frame_to_score_map) const
{
    CHECK_NOTNULL(frame_to_score_map)->clear();
    // Score each keyframe, then take the part which is in the
    // top fraction and allow only matches to landmarks which are associated with
    // these keyframes.对每个关键帧进行评分，然后选择最上面的部分，只允许匹配与这些关键帧相关的地标。

    scoring::ScoreList<loop_closure::KeyframeId> score_list;
    timing::Timer timer_scoring("Loop Closure: scoring for covisibility filter");
    CHECK(compute_keyframe_scores_);
    compute_keyframe_scores_(//会有两种计算方式，一种是基于累加的，就是得分就是这帧中候选描述子的个数，默认是用这个去计算的。一种是概率
            frame_to_matches, keyframe_id_to_num_descriptors_,
            static_cast<size_t>(NumDescriptors()), &score_list);
    timer_scoring.Stop();

    // We want to take matches from the best n score keyframes, but make sure
    // that we evaluate at minimum a given number.
    constexpr size_t kNumMinimumScoreIdsToEvaluate = 4u;
    //fraction_best_scores的意思是考虑前百分之多少，然后kNumMinimumScoreIdsToEvaluate的意思是最小要去评估前几帧
    size_t num_score_ids_to_evaluate = std::max<size_t>(
            static_cast<size_t>(score_list.size() * settings_.fraction_best_scores),
            kNumMinimumScoreIdsToEvaluate);
    // Ensure staying in bounds.
    num_score_ids_to_evaluate =//还不能超过了score_list的size()
            std::min<size_t>(num_score_ids_to_evaluate, score_list.size());
    std::nth_element(//把得分最大的排在前面
            score_list.begin(), score_list.begin() + num_score_ids_to_evaluate,
            score_list.end(),
            [](const scoring::Score<loop_closure::KeyframeId>& lhs,
               const scoring::Score<loop_closure::KeyframeId>& rhs) -> bool {
                return lhs.second > rhs.second;
            });
    frame_to_score_map->insert(//只取前num_score_ids_to_evaluate个
            score_list.begin(), score_list.begin() + num_score_ids_to_evaluate);
}

template <>
void MatchingBasedLoopDetector::computeRelevantIdsForFiltering(
    const loop_closure::VertexToMatches& /* vertex_to_matches */,
    IdToScoreMap<loop_closure::VertexId>* /* vertex_to_score_map */) const {
  // We do not have to score vertices to filter unlikely matches because this
  // is done already at keyframe level.
}
}  // namespace matching_based_loopclosure

#endif  // MATCHING_BASED_LOOPCLOSURE_MATCHING_BASED_ENGINE_INL_H_
