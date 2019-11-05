#include "rovioli/vio-update-builder.h"

#include <maplab-common/interpolation-helpers.h>
extern std::unordered_map<double,std::vector<double>> msckf_datas_map;


namespace rovioli {

VioUpdateBuilder::VioUpdateBuilder()
    : last_received_timestamp_synced_nframe_queue_(
          aslam::time::getInvalidTime()),
      last_received_timestamp_rovio_estimate_queue(
          aslam::time::nanoSecondsToSeconds(aslam::time::getInvalidTime())) {}

void VioUpdateBuilder::processSynchronizedNFrameImu(//处理多相机的数据
    const vio::SynchronizedNFrameImu::ConstPtr& synced_nframe_imu)
{
    CHECK(synced_nframe_imu != nullptr);
    const int64_t timestamp_nframe_ns =
            synced_nframe_imu->nframe->getMaxTimestampNanoseconds();//多相机系统中最大的时间戳
    CHECK_GT(timestamp_nframe_ns, last_received_timestamp_synced_nframe_queue_);
    last_received_timestamp_synced_nframe_queue_ = timestamp_nframe_ns;

    std::lock_guard<std::recursive_mutex> lock(queue_mutex_);
    synced_nframe_imu_queue_.push(synced_nframe_imu);
    findMatchAndPublish();
}

void VioUpdateBuilder::processRovioEstimate(//处理rovio的测量
    const RovioEstimate::ConstPtr& rovio_estimate) {
  CHECK(rovio_estimate != nullptr);
  const double timestamp_rovio_estimate_s = rovio_estimate->timestamp_s;//
  CHECK_GT(
      timestamp_rovio_estimate_s, last_received_timestamp_rovio_estimate_queue);
  last_received_timestamp_rovio_estimate_queue = timestamp_rovio_estimate_s;



  std::lock_guard<std::recursive_mutex> lock(queue_mutex_);
  rovio_estimate_queue_.push_back(rovio_estimate);
  findMatchAndPublish();
}

void VioUpdateBuilder::processLocalizationResult(
    const vio::LocalizationResult::ConstPtr& localization_result) {
  CHECK(localization_result);
  std::lock_guard<std::mutex> lock(mutex_last_localization_state_);
  switch (localization_result->localization_type) {
    case vio::LocalizationResult::LocalizationMode::kGlobal:
      last_localization_state_ = vio::LocalizationState::kLocalized;
      break;
    case vio::LocalizationResult::LocalizationMode::kMapTracking:
      last_localization_state_ = vio::LocalizationState::kMapTracking;
      break;
  }
}
void VioUpdateBuilder::findMatchAndPublish() {//同步图像和rovio的信息
  std::lock_guard<std::recursive_mutex> lock(queue_mutex_);

  if (synced_nframe_imu_queue_.empty() || rovio_estimate_queue_.empty()) //这两个队列都要有数据才可以
  {
    // Nothing to do.
    return;
  }
  const vio::SynchronizedNFrameImu::ConstPtr& oldest_unmatched_synced_nframe =
      synced_nframe_imu_queue_.front();//最老的没有同步的视觉信息
  const int64_t timestamp_nframe_ns =
      oldest_unmatched_synced_nframe->nframe->getMinTimestampNanoseconds();//得到多相机系统中所有相机中时间戳最小的时间

  // We need to use iterator instead of const_iterator because erase isn't
  // defined for const_iterators in g++ 4.8.
  RovioEstimateQueue::iterator it_rovio_estimate_before_nframe =
      rovio_estimate_queue_.end();
  RovioEstimateQueue::iterator it_rovio_estimate_after_nframe =
      rovio_estimate_queue_.end();

  double exact_time = 0;
  bool found_exact_match = false;
  bool found_matches_to_interpolate = false;
  // Need at least two values for interpolation.
  for (it_rovio_estimate_before_nframe = rovio_estimate_queue_.begin();
       it_rovio_estimate_before_nframe != rovio_estimate_queue_.end();
       ++it_rovio_estimate_before_nframe)
  {
    it_rovio_estimate_after_nframe = it_rovio_estimate_before_nframe + 1;
    // Check if exact match.
    if (aslam::time::secondsToNanoSeconds(
            (*it_rovio_estimate_before_nframe)->timestamp_s) ==
        timestamp_nframe_ns) {//视觉信息和imu信息是刚好对应上的
      found_exact_match = true;
        exact_time = (*it_rovio_estimate_before_nframe)->timestamp_s;
      break;
    }
    if (it_rovio_estimate_after_nframe != rovio_estimate_queue_.end() &&
        aslam::time::secondsToNanoSeconds(
            (*it_rovio_estimate_before_nframe)->timestamp_s) <=
            timestamp_nframe_ns &&
        aslam::time::secondsToNanoSeconds(
            (*it_rovio_estimate_after_nframe)->timestamp_s) >
            timestamp_nframe_ns) {
      // Found matching vi nodes.
      found_matches_to_interpolate = true;//就是这两个rovio的状态信息是在两个图像信息之间
      break;
    }
  }

  if (!found_exact_match && !found_matches_to_interpolate) {//都没有发现的话，就返回
    return;
  }

  CHECK(it_rovio_estimate_before_nframe != rovio_estimate_queue_.end());
  CHECK(
      found_exact_match ||
      it_rovio_estimate_after_nframe != rovio_estimate_queue_.end());
  CHECK(it_rovio_estimate_before_nframe != it_rovio_estimate_after_nframe);
  const RovioEstimate::ConstPtr& rovio_estimate_before_nframe =//it_rovio_estimate_before_nframe<= 视觉时间
      *it_rovio_estimate_before_nframe;
  const RovioEstimate::ConstPtr& rovio_estimate_after_nframe =
      *it_rovio_estimate_after_nframe;

  // Build VioUpdate.
  vio::VioUpdate::Ptr vio_update = aligned_shared<vio::VioUpdate>();
  vio_update->timestamp_ns = timestamp_nframe_ns;//使用视觉的时间
  vio_update->keyframe_and_imudata = oldest_unmatched_synced_nframe;//最老的这个状态
  if (found_exact_match) //如果时间戳是刚好对上的，就不用插值
  {
////msckf数据
      vio::ViNodeState msckf_vi_node;
      std::vector<double> msckfdata  = msckf_datas_map[exact_time];

      aslam::Position3D msckf_position(msckfdata[0],msckfdata[1],msckfdata[2]);


      aslam::Quaternion msckf_q_A_B(msckfdata[6],msckfdata[3],msckfdata[4],msckfdata[5]);

      aslam::Transformation msckf_T_M_I(msckf_q_A_B, msckf_position);

      msckf_vi_node.set_T_M_I(msckf_T_M_I);

      Eigen::Vector3d msckf_v_M_I(msckfdata[7],msckfdata[8],msckfdata[9]);

      msckf_vi_node.set_v_M_I(msckf_v_M_I);

      // Interpolate biases.
      Eigen::Vector3d msckf_acc_bias(msckfdata[10],msckfdata[11],msckfdata[12]);
      Eigen::Vector3d msckf_gyro_bias(msckfdata[13],msckfdata[14],msckfdata[15]);

      msckf_vi_node.setAccBias(msckf_acc_bias);
      msckf_vi_node.setGyroBias(msckf_gyro_bias);
      vio_update->vinode = msckf_vi_node;

      ////替换结束


    if (rovio_estimate_before_nframe->has_T_G_M) {
      vio_update->T_G_M = rovio_estimate_before_nframe->T_G_M;
    }
  } else
      {
    // Need to interpolate ViNode.
    const int64_t t_before = aslam::time::secondsToNanoSeconds(
        rovio_estimate_before_nframe->timestamp_s);
    const int64_t t_after = aslam::time::secondsToNanoSeconds(
        rovio_estimate_after_nframe->timestamp_s);

    vio::ViNodeState interpolated_vi_node;
    interpolateViNodeState(//如果不是精准时间戳，需要将前后imu的时间戳和视觉的时间去进行插值
        t_before, rovio_estimate_before_nframe->vinode, t_after,
        rovio_estimate_after_nframe->vinode, timestamp_nframe_ns,
        &interpolated_vi_node);
    vio_update->vinode = interpolated_vi_node;

    if (rovio_estimate_before_nframe->has_T_G_M &&
        rovio_estimate_after_nframe->has_T_G_M) {
      common::interpolateTransformation(
          t_before, rovio_estimate_before_nframe->T_G_M, t_after,
          rovio_estimate_after_nframe->T_G_M, timestamp_nframe_ns,
          &vio_update->T_G_M);
    }
  }
  vio_update->vio_state = vio::EstimatorState::kRunning;//kUninitialized, kStartup, kRunning
  vio_update->vio_update_type = vio::UpdateType::kNormalUpdate;// kInvalid, kNormalUpdate, kZeroVelocityUpdate
  {
    std::lock_guard<std::mutex> lock(mutex_last_localization_state_);
    vio_update->localization_state = last_localization_state_;
    last_localization_state_ = vio::LocalizationState::kUninitialized;
  }

  // Publish VIO update.
  CHECK(vio_update_publish_function_);
  vio_update_publish_function_(vio_update);

  // Clean up queues.
  if (it_rovio_estimate_before_nframe != rovio_estimate_queue_.begin()) {
    if (found_exact_match) {
      rovio_estimate_queue_.erase(
          rovio_estimate_queue_.begin(), it_rovio_estimate_before_nframe);
    } else {
      // Keep the two ViNodeStates that were used for interpolation as a
      // subsequent SynchronizedNFrameImu may need to be interpolated between
      // those two points again.
      rovio_estimate_queue_.erase(
          rovio_estimate_queue_.begin(), it_rovio_estimate_before_nframe - 1);
    }
  }
  synced_nframe_imu_queue_.pop();
}

void VioUpdateBuilder::interpolateViNodeState(
    const int64_t timestamp_ns_a, const vio::ViNodeState& vi_node_a,
    const int64_t timestamp_ns_b, const vio::ViNodeState& vi_node_b,
    const int64_t timestamp_ns_interpolated,
    vio::ViNodeState* vi_node_interpolated) {
  CHECK_NOTNULL(vi_node_interpolated);
  CHECK_LT(timestamp_ns_a, timestamp_ns_b);
  CHECK_LE(timestamp_ns_a, timestamp_ns_interpolated);
  CHECK_LE(timestamp_ns_interpolated, timestamp_ns_b);

  // Interpolate pose.
  aslam::Transformation interpolated_T_M_I;
  common::interpolateTransformation(
      timestamp_ns_a, vi_node_a.get_T_M_I(), timestamp_ns_b,
      vi_node_b.get_T_M_I(), timestamp_ns_interpolated, &interpolated_T_M_I);
  vi_node_interpolated->set_T_M_I(interpolated_T_M_I);

  // Interpolate velocity.
  Eigen::Vector3d interpolated_v_M_I;
  common::linerarInterpolation(
      timestamp_ns_a, vi_node_a.get_v_M_I(), timestamp_ns_b,
      vi_node_b.get_v_M_I(), timestamp_ns_interpolated, &interpolated_v_M_I);
  vi_node_interpolated->set_v_M_I(interpolated_v_M_I);

  // Interpolate biases.
  Eigen::Vector3d interpolated_acc_bias, interpolated_gyro_bias;
  common::linerarInterpolation(
      timestamp_ns_a, vi_node_a.getAccBias(), timestamp_ns_b,
      vi_node_b.getAccBias(), timestamp_ns_interpolated,
      &interpolated_acc_bias);
  common::linerarInterpolation(
      timestamp_ns_a, vi_node_a.getGyroBias(), timestamp_ns_b,
      vi_node_b.getGyroBias(), timestamp_ns_interpolated,
      &interpolated_gyro_bias);
  vi_node_interpolated->setAccBias(interpolated_acc_bias);
  vi_node_interpolated->setGyroBias(interpolated_gyro_bias);
}

}  // namespace rovioli
