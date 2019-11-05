#include "vi-map-helpers/mission-clustering-coobservation.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <maplab-common/accessors.h>
#include <vi-map/vi-map.h>

#include "vi-map-helpers/vi-map-queries.h"

namespace vi_map_helpers {
namespace {
bool haveCommonElements(
    const vi_map::LandmarkIdSet& set_a, const vi_map::LandmarkIdSet& set_b) {
  // Make sure we loop over the smaller set.
  if (set_b.size() < set_a.size()) {
    return haveCommonElements(set_b, set_a);
  }

  for (const vi_map::LandmarkId& landmark_id : set_a) {
    if (set_b.count(landmark_id) > 0u) {
      return true;
    }
  }
  return false;
}
}  // namespace

//MissionCoobservationCachedQuery的构造函数
MissionCoobservationCachedQuery::MissionCoobservationCachedQuery(
    const vi_map::VIMap& vi_map, const vi_map::MissionIdSet& mission_ids) {
  for (const vi_map::MissionId& mission_id : mission_ids)//遍历每一个任务
  {
    vi_map_helpers::VIMapQueries queries(vi_map);//构造vimap查询器
    queries.getLandmarksObservedByMission(//mission_landmark_map_里存了每个任务的所有地图点的set集合
        mission_id, &mission_landmark_map_[mission_id]);
  }
}
//查询这个任务和其他的簇任务之间也有没有共视
bool MissionCoobservationCachedQuery::hasCommonObservations(
    const vi_map::MissionId& mission_id,
    const vi_map::MissionIdSet& other_mission_ids) const
    {
        CHECK(mission_id.isValid());
        const vi_map::LandmarkIdSet& landmarks_of_mission =//mission_id这个任务的所有地图点set
                common::getChecked(mission_landmark_map_, mission_id);

        for (const vi_map::MissionId& other_mission_id : other_mission_ids)//遍历这个簇内的所有任务
        {
            const vi_map::LandmarkIdSet& landmarks_of_other_mission =//簇中某一个地图点的set
                    common::getChecked(mission_landmark_map_, other_mission_id);

            //其实就是遍历查询有没有重复的id
            if (haveCommonElements(landmarks_of_mission, landmarks_of_other_mission)) {
                return true;
            }
        }
        return false;
    }

//根据共视关系对多个任务分簇
    std::vector<vi_map::MissionIdSet> clusterMissionByLandmarkCoobservations(
            const vi_map::VIMap& vi_map, const vi_map::MissionIdSet& mission_ids)
    {
        std::vector<vi_map::MissionIdSet> components;
//构造任务共视查询器，就是把每个任务的地图点id都存到mission_landmark_map_
        MissionCoobservationCachedQuery coobservations(vi_map, mission_ids);

        for (const vi_map::MissionId& mission_id : mission_ids) //遍历每个任务
        {
            vi_map::MissionIdSet new_component;
            new_component.emplace(mission_id);
            //当前任务去和components查询有没有共视地图点
            //比如说1,2任务有共视，3,4,5任务有共视，components里就是vector<<1,2>,<3,4,5>>.
            // 当第6个任务来的时候，就会去和<1,2>和<3,4,5>检查共视，如果和1,2有共视就是<1,2,6>，如果都没有
            // components里就是vector<<1,2>,<3,4,5>,<6>>.
            for (std::vector<vi_map::MissionIdSet>::iterator it = components.begin();
                 it != components.end();)
            {
                //遍历components中的每一个簇
                if (coobservations.hasCommonObservations(mission_id, *it))
                {//如果有共视，那么就要更新这个簇
                    // Add the existing component items to the new component.
                    new_component.insert(it->begin(), it->end());
                    // Remove the old component from the components list.
                    it = components.erase(it);
                } else {
                    ++it;
                }
            }
            components.emplace_back(new_component);//将当前簇加进components
        }
        return components;//components里就是一个个簇，每个簇里的任务之间是有共视的，但是簇和簇之间的任务是没有共视的
    }

}  // namespace vi_map_helpers
