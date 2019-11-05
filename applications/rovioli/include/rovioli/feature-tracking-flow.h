#ifndef ROVIOLI_FEATURE_TRACKING_FLOW_H_
#define ROVIOLI_FEATURE_TRACKING_FLOW_H_

#include <aslam/cameras/ncamera.h>
#include <message-flow/message-flow.h>
#include <sensors/imu.h>
#include <vio-common/vio-types.h>

#include "rovioli/feature-tracking.h"
#include "rovioli/flow-topics.h"

namespace rovioli {

class FeatureTrackingFlow {
 public:
  FeatureTrackingFlow(
      const aslam::NCamera::Ptr& camera_system, const vi_map::Imu& imu_sensor)
      : tracking_pipeline_(camera_system, imu_sensor) //将多相机系统和imu传感器赋值给追踪线程tracking_pipeline_
      {
    CHECK(camera_system);
  }

  //追踪线程
  void attachToMessageFlow(message_flow::MessageFlow* flow)
  {
      CHECK_NOTNULL(flow);
      static constexpr char kSubscriberNodeName[] = "FeatureTrackingFlow";

      //SynchronizedNFrameImu的发布
      std::function<void(vio::SynchronizedNFrameImu::ConstPtr)> publish_result =
              flow->registerPublisher<message_flow_topics::TRACKED_NFRAMES_AND_IMU>();

      // NOTE: the publisher function pointer is copied intentionally; otherwise
      // we would capture a reference to a temporary.
      flow->registerSubscriber<message_flow_topics::SYNCED_NFRAMES_AND_IMU>(
              kSubscriberNodeName, message_flow::DeliveryOptions(),
              [publish_result,
                      this](const vio::SynchronizedNFrameImu::Ptr& nframe_imu) {
                  CHECK(nframe_imu);
                  const bool success =//跟踪线程，在这里会对所有的keyframe进行追踪，返回的是追踪的多相机系统，TRACKED_NFRAMES_AND_IMU是消息名称
                          this->tracking_pipeline_.trackSynchronizedNFrameImuCallback(
                                  nframe_imu);
                  if (success) {
                      // This will only fail for the first frame.
                      publish_result(nframe_imu);
                  }
              });

      flow->registerSubscriber<message_flow_topics::ROVIO_ESTIMATES>(
              kSubscriberNodeName, message_flow::DeliveryOptions(),
              [this](const RovioEstimate::ConstPtr& estimate) {
                  CHECK(estimate);
                  this->tracking_pipeline_.setCurrentImuBias(estimate);
              });
  }

 private:
  FeatureTracking tracking_pipeline_;
};

}  // namespace rovioli

#endif  // ROVIOLI_FEATURE_TRACKING_FLOW_H_
