#include "vi-map/landmark-quality-metrics.h"

#include <limits>
#include <vector>

#include <Eigen/Dense>
#include <aslam/common/statistics/statistics.h>
#include <gflags/gflags.h>
#include <vi-map/landmark.h>
#include <vi-map/vi-map.h>

DEFINE_double(
    vi_map_landmark_quality_min_observation_angle_deg, 5,
    "Minimum angle disparity of observers for a landmark to be "
    "well constrained.");

DEFINE_uint64(
    vi_map_landmark_quality_min_observers, 4,
    "Minimum number of observers for a landmark to be "
    "well constrained.");

DEFINE_double(
    vi_map_landmark_quality_max_distance_from_closest_observer, 40,
    "Maximum distance from closest observer for a landmark to be "
    "well constrained [m].");

DEFINE_double(
    vi_map_landmark_quality_min_distance_from_closest_observer, 0.05,
    "Minimum distance from closest observer for a landmark to be "
    "well constrained [m].");

namespace vi_map {
LandmarkWellConstrainedSettings::LandmarkWellConstrainedSettings()
    : max_distance_to_closest_observer(
          FLAGS_vi_map_landmark_quality_max_distance_from_closest_observer),
      min_distance_to_closest_observer(
          FLAGS_vi_map_landmark_quality_min_distance_from_closest_observer),
      min_observation_angle_deg(
          FLAGS_vi_map_landmark_quality_min_observation_angle_deg),
      min_observers(FLAGS_vi_map_landmark_quality_min_observers) {}

bool isLandmarkWellConstrained(//在优化的时候调用
    const vi_map::VIMap& map, const vi_map::Landmark& landmark)
    {
  constexpr bool kReEvaluateQuality = false;//因为只是查看这个地图点的质量，所以不用再去重新评估这个地图点的质量
  return isLandmarkWellConstrained(map, landmark, kReEvaluateQuality);
}

    bool isLandmarkWellConstrained(//进行地图点质量的判定
            const vi_map::VIMap& map, const vi_map::Landmark& landmark,
            bool re_evaluate_quality)
    {
        //在重新三角化时，这里会默认设成kbad，而在关键帧选择以后重新评估点质量时，这里应该是good和bad都有
        vi_map::Landmark::Quality quality = landmark.getQuality();
        // Localization landmarks are always good.
        if (quality == vi_map::Landmark::Quality::kLocalizationSummaryLandmark) {
            return true;
        }
        if (re_evaluate_quality) {//在重新三角化时，kReEvaluateQuality会默认设成true,也就是重新评估质量
            quality = vi_map::Landmark::Quality::kUnknown;
        }

        if (quality != vi_map::Landmark::Quality::kUnknown) {
            return quality == vi_map::Landmark::Quality::kGood;
        }

        // Use default settings.
        LandmarkWellConstrainedSettings settings;

        const vi_map::KeypointIdentifierList& backlinks = landmark.getObservations();//得到这个地图点的所有观测信息
        if (backlinks.size() < settings.min_observers) {//FLAGS_vi_map_landmark_quality_min_observers去赋值这个最小观测
            statistics::StatsCollector stats("Landmark has too few backlinks");
            stats.IncrementOne();
            return false;
        }

        const Eigen::Vector3d& p_G_fi = map.getLandmark_G_p_fi(landmark.id());//应该是vio三角化的点吧
        Aligned<std::vector, Eigen::Vector3d> G_normalized_incidence_rays;
        G_normalized_incidence_rays.reserve(backlinks.size());
        double signed_distance_to_closest_observer =
                std::numeric_limits<double>::max();
        for (const vi_map::KeypointIdentifier& backlink : backlinks)
        {
            const vi_map::Vertex& vertex = map.getVertex(backlink.frame_id.vertex_id);

            const pose::Transformation T_G_C =//得到这帧在世界坐标系下的位姿
                    map.getVertex_T_G_I(backlink.frame_id.vertex_id) *
                    map.getSensorManager()
                            .getNCameraForMission(vertex.getMissionId())
                            .get_T_C_B(backlink.frame_id.frame_index)
                            .inverse();

            const Eigen::Vector3d p_C_fi = map.getLandmark_p_C_fi(//当前这个地图点在这帧中的相机坐标系中的位置
                    landmark.id(), vertex, backlink.frame_id.frame_index);
            const Eigen::Vector3d G_incidence_ray = T_G_C.getPosition() - p_G_fi;//这个只是全局坐标系下相机的位置和这个地图点位置的差值

            const double distance = G_incidence_ray.norm();//求相机到这个点的绝对距离
            const double signed_distance = distance * (p_C_fi(2) < 0.0 ? -1.0 : 1.0);//带符号的距离
            signed_distance_to_closest_observer =//求最小的带符号的距离
                    std::min(signed_distance_to_closest_observer, signed_distance);

            if (distance > 0) {
                G_normalized_incidence_rays.emplace_back(G_incidence_ray / distance);//这是为了之后算cos，也就是两帧和他们观测到的路标点之间算夹角时用到
            }
        }

        return isLandmarkWellConstrained(
                G_normalized_incidence_rays, signed_distance_to_closest_observer,
                settings);
    }
}  // namespace vi_map
