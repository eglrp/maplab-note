#include "rovioli/rovioli-node.h"

#include <string>

#include <aslam/cameras/ncamera.h>
#include <localization-summary-map/localization-summary-map.h>
#include <message-flow/message-flow.h>
#include <message-flow/message-topic-registration.h>
#include <vio-common/vio-types.h>


#include "rovioli/datasource-flow.h"
#include "rovioli/feature-tracking-flow.h"
#include "rovioli/imu-camera-synchronizer-flow.h"
#include "rovioli/localizer-flow.h"
#include "rovioli/rovio-flow.h"
#include "rovioli/synced-nframe-throttler-flow.h"
#include "rovioli/replace-pose-flow.h"

DEFINE_bool(
    rovioli_run_map_builder, true,
    "When set to false, the map builder will be deactivated and no map will be "
    "built. Rovio+Localization will still run as usual.");

namespace rovioli {

//输入：多相机系统，maplab的imu传感器（里面有imu_sigmas）,rovio用的imu_sigmas，保存地图在哪个文件夹，定位地图，消息流
    RovioliNode::RovioliNode(
    const aslam::NCamera::Ptr& camera_system,
    vi_map::Imu::UniquePtr maplab_imu_sensor,
    const vi_map::ImuSigmas& rovio_imu_sigmas,
    const std::string& save_map_folder,
    const summary_map::LocalizationSummaryMap* const localization_map,
    message_flow::MessageFlow* flow)
    : is_datasource_exhausted_(false)
    {
        // localization_summary_map is optional and can be a nullptr.
        CHECK(camera_system);
        CHECK(maplab_imu_sensor);
        CHECK_NOTNULL(flow);

        // TODO(schneith): At the moment we need to provide two noise sigmas; one for
        // maplab and one for ROVIO. Unify this.
        datasource_flow_.reset(
                new DataSourceFlow(*camera_system, *maplab_imu_sensor));//创建相机和imu的回调函数
        datasource_flow_->attachToMessageFlow(flow);//向datasource_flow_中添加相机消息，imu消息

        //输入的是rovio的sigmas，多相机系统,初始化rovio
        rovio_flow_.reset(new RovioFlow(*camera_system, rovio_imu_sigmas));
        rovio_flow_->attachToMessageFlow(flow);

        const bool localization_enabled = localization_map != nullptr;
        if (FLAGS_rovioli_run_map_builder || localization_enabled)
        {
            // If there's no localization and no map should be built, no maplab feature
            // tracking is needed.
            //
            if (localization_enabled)//如果需要重定位
            {
                constexpr bool kVisualizeLocalization = true;
                localizer_flow_.reset(//将地图导入，以及初始化
                        new LocalizerFlow(*localization_map, kVisualizeLocalization));
                localizer_flow_->attachToMessageFlow(flow);
            }

            // Launch the synchronizer after the localizer because creating the
            // localization database can take some time. This can cause the
            // synchronizer's detection of missing image or IMU measurements to fire
            // early.

            //同步所有相机图像，还有imu数据,然后发出
            synchronizer_flow_.reset(new ImuCameraSynchronizerFlow(camera_system));
            synchronizer_flow_->attachToMessageFlow(flow);//接收IMAGE_MEASUREMENTS，IMU_MEASUREMENTS.同步以后发出SYNCED_NFRAMES_AND_IMU

            tracker_flow_.reset(
                    new FeatureTrackingFlow(camera_system, *maplab_imu_sensor));//订阅SYNCED_NFRAMES_AND_IMU，发出TRACKED_NFRAMES_AND_IMU
            tracker_flow_->attachToMessageFlow(flow);

            throttler_flow_.reset(new SyncedNFrameThrottlerFlow);
            throttler_flow_->attachToMessageFlow(flow);//订阅TRACKED_NFRAMES_AND_IMU，发出THROTTLED_TRACKED_NFRAMES_AND_IMU
        }

        data_publisher_flow_.reset(new DataPublisherFlow);
        data_publisher_flow_->attachToMessageFlow(flow);

        if (FLAGS_rovioli_run_map_builder)
        {
            map_builder_flow_.reset(
                    new MapBuilderFlow(//初始化地图构建器
                            camera_system, std::move(maplab_imu_sensor), save_map_folder));
            map_builder_flow_->attachToMessageFlow(flow);
        }

        // Subscribe to end of days signal from the datasource.
        datasource_flow_->registerEndOfDataCallback(
                [&]() { is_datasource_exhausted_.store(true); });
    }

RovioliNode::~RovioliNode() {
  shutdown();
}

void RovioliNode::saveMapAndOptionallyOptimize(//map_builder_flow_保存地图罢了
    const std::string& path, const bool overwrite_existing_map,
    bool process_to_localization_map) {
  if (map_builder_flow_) {
    map_builder_flow_->saveMapAndOptionallyOptimize(
        path, overwrite_existing_map, process_to_localization_map);
  }
}

void RovioliNode::start() {
  CHECK(!is_datasource_exhausted_.load())
      << "Cannot start localization node after the "
      << "end-of-days signal was received!";
  datasource_flow_->startStreaming();
  VLOG(1) << "Starting data source...";
}

void RovioliNode::shutdown() {
  datasource_flow_->shutdown();
  VLOG(1) << "Closing data source...";
}

std::atomic<bool>& RovioliNode::isDataSourceExhausted() {
  return is_datasource_exhausted_;
}

}  // namespace rovioli
