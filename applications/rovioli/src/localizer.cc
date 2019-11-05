#include "rovioli/localizer.h"

#include <localization-summary-map/localization-summary-map.h>
#include <loop-closure-handler/loop-detector-node.h>
#include <vio-common/vio-types.h>

namespace rovioli {

Localizer::Localizer(
    const summary_map::LocalizationSummaryMap& localization_summary_map,
    const bool visualize_localization)
    : localization_summary_map_(localization_summary_map) //初始化地图
    {
  current_localization_mode_ = Localizer::LocalizationMode::kGlobal;

  global_loop_detector_.reset(new loop_detector_node::LoopDetectorNode);

  CHECK(global_loop_detector_ != nullptr);
  if (visualize_localization) {
    global_loop_detector_->instantiateVisualizer();//初始化闭环的相关话题
  }

  LOG(INFO) << "Creating localization database...";//将map里地图点的id,描述子信息等信息加入
  global_loop_detector_->addLocalizationSummaryMapToDatabase(
      localization_summary_map_);
  LOG(INFO) << "Done.";
}

Localizer::LocalizationMode Localizer::getCurrentLocalizationMode() const {
  return current_localization_mode_;
}

bool Localizer::localizeNFrame(//重定位,输入多相机图片
    const aslam::VisualNFrame::ConstPtr& nframe,
    vio::LocalizationResult* localization_result) const
{
    CHECK(nframe);
    CHECK_NOTNULL(localization_result);

    bool result = false;
    switch (current_localization_mode_)
    {
        case Localizer::LocalizationMode::kGlobal:
            result = localizeNFrameGlobal(nframe, &localization_result->T_G_I_lc_pnp);
            break;
        case Localizer::LocalizationMode::kMapTracking:
            result =
                    localizeNFrameMapTracking(nframe, &localization_result->T_G_I_lc_pnp);
            break;
        default:
            LOG(FATAL) << "Unknown localization mode.";
            break;
    }

    localization_result->timestamp = nframe->getMinTimestampNanoseconds();
    localization_result->nframe_id = nframe->getId();
    localization_result->localization_type = current_localization_mode_;
    return result;
}

bool Localizer::localizeNFrameGlobal(//全局重定位
    const aslam::VisualNFrame::ConstPtr& nframe,
    aslam::Transformation* T_G_I_lc_pnp) const
    {
  constexpr bool kSkipUntrackedKeypoints = false;//默认不跳过未追踪的点
  unsigned int num_lc_matches;//匹配到的点对个数
  vi_map::VertexKeyPointToStructureMatchList inlier_structure_matches;


  //输入多相机图片。是否跳过未追踪到的点，离线地图
  //输出pnp结果，匹配的个数，还有一个是啥？
  return global_loop_detector_->findNFrameInSummaryMapDatabase(
      *nframe, kSkipUntrackedKeypoints, localization_summary_map_, T_G_I_lc_pnp,
      &num_lc_matches, &inlier_structure_matches);
}

bool Localizer::localizeNFrameMapTracking(
    const aslam::VisualNFrame::ConstPtr& /*nframe*/,
    aslam::Transformation* /*T_G_I_lc_pnp*/) const {
  LOG(FATAL) << "Not implemented yet.";
  return false;
}

}  // namespace rovioli
