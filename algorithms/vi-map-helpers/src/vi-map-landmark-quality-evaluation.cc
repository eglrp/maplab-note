#include "vi-map-helpers/vi-map-landmark-quality-evaluation.h"

#include <glog/logging.h>
#include <maplab-common/multi-threaded-progress-bar.h>
#include <maplab-common/parallel-process.h>
#include <maplab-common/threading-helpers.h>
#include <vi-map/landmark-quality-metrics.h>
#include <vi-map/landmark.h>
#include <vi-map/vi-map.h>

namespace vi_map_helpers {

    void evaluateLandmarkQuality(vi_map::VIMap* map)
    {
        //评估所有地图点的质量
        CHECK_NOTNULL(map);
        constexpr bool kEvaluateLandmarkQuality = true;//要重新评估点的质量

        vi_map::LandmarkIdList landmark_ids;
        //得到当前地图内的所有地图点
        map->getAllLandmarkIds(&landmark_ids);
        //所有地图点的个数
        const size_t num_landmarks = landmark_ids.size();

        VLOG(1) << "Evaluating quality of landmarks of " << num_landmarks
                << " landmarks.";

        common::MultiThreadedProgressBar progress_bar;

        //定义了一个lambda函数
        std::function<void(const std::vector<size_t>&)> evaluator =
                [&landmark_ids, map, &progress_bar](const std::vector<size_t>& batch) {
                    progress_bar.setNumElements(batch.size());
                    size_t num_processed = 0u;
                    for (size_t idx : batch)
                    {
                        CHECK_LT(idx, landmark_ids.size());
                        const vi_map::LandmarkId& landmark_id = landmark_ids[idx];
                        CHECK(landmark_id.isValid());
                        vi_map::Landmark& landmark = map->getLandmark(landmark_id);
                        landmark.setQuality(
                                vi_map::isLandmarkWellConstrained(
                                        *map, landmark, kEvaluateLandmarkQuality)
                                ? vi_map::Landmark::Quality::kGood//重新评估地图点的质量，之前重新三角化时也会评估点的质量
                                : vi_map::Landmark::Quality::kBad);
                        progress_bar.update(++num_processed);
                    }
                };

        static constexpr bool kAlwaysParallelize = false;
        const size_t num_threads = common::getNumHardwareThreads();
        common::ParallelProcess(
                num_landmarks, evaluator, kAlwaysParallelize, num_threads);
    }

void resetLandmarkQualityToUnknown(vi_map::VIMap* map) {
  CHECK_NOTNULL(map);
  vi_map::LandmarkIdList landmark_ids;
  map->getAllLandmarkIds(&landmark_ids);
  const size_t num_landmarks = landmark_ids.size();

  common::MultiThreadedProgressBar progress_bar;
  std::function<void(const std::vector<size_t>&)> evaluator =
      [&landmark_ids, map, &progress_bar](const std::vector<size_t>& batch) {
        progress_bar.setNumElements(batch.size());
        size_t num_processed = 0u;
        for (size_t idx : batch) {
          CHECK_LT(idx, landmark_ids.size());
          const vi_map::LandmarkId& landmark_id = landmark_ids[idx];
          CHECK(landmark_id.isValid());
          vi_map::Landmark& landmark = map->getLandmark(landmark_id);
          landmark.setQuality(vi_map::Landmark::Quality::kUnknown);
          progress_bar.update(++num_processed);
        }
      };

  static constexpr bool kAlwaysParallelize = false;
  const size_t num_threads = common::getNumHardwareThreads();
  common::ParallelProcess(
      num_landmarks, evaluator, kAlwaysParallelize, num_threads);
}

}  // namespace vi_map_helpers
