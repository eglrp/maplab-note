#include "rovioli/rovio-flow.h"

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <aslam/cameras/ncamera.h>
#include <aslam/common/pose-types.h>
#include <aslam/common/time.h>
#include <gflags/gflags.h>
#include <maplab-common/string-tools.h>
#include <maplab-common/unique-id.h>
#include <message-flow/message-flow.h>
#include <vio-common/vio-types.h>

#include "rovioli/flow-topics.h"
#include "rovioli/rovio-factory.h"

DEFINE_bool(
    rovio_update_filter_on_imu, true,
    "Update the filter state for IMU measurement; if false the IMU measurements"
    " are queued and the state is only forward propagated before the next "
    "update.");
DEFINE_string(
    rovio_active_camera_indices, "0",
    "Comma separated indices of cameras to use for motion tracking.");

namespace rovioli {
namespace {
void ensurePositiveQuaternion(aslam::Quaternion* quat) {
  CHECK_NOTNULL(quat);
  if (quat->toImplementation().w() < 0.0) {
    quat->toImplementation().coeffs() *= -1.0;
  }
}
}  // namespace

//输入多相机系统，给imu_sigms和多相机系统
RovioFlow::RovioFlow(
    const aslam::NCamera& camera_calibration,
    const vi_map::ImuSigmas& imu_sigmas)
{
    // Multi-camera support in ROVIO is still experimental. Therefore, only a
    // single camera will be used for motion tracking per default.
    const size_t num_cameras = camera_calibration.getNumCameras();
    LOG_IF(WARNING, num_cameras > 1u)
    << "Multi-camera support of ROVIO is still experimental. Per default "
    << "only the first camera will be used for motion tracking. However, all "
    << "cameras will be used for mapping and localization. You can override  "
    << "the active used for motion tracking with the flag: "
    << "--rovio_active_camera_indices";

    CHECK(!FLAGS_rovio_active_camera_indices.empty());//这个是用来指定vio模式用哪一个相机来做track
    constexpr char kDelimiter = ',';
    constexpr bool kRemoveEmpty = true;
    std::vector<std::string> tokens;
    common::tokenizeString(//将FLAGS_rovio_active_camera_indices里的相机id放到tokens中
            FLAGS_rovio_active_camera_indices, kDelimiter, kRemoveEmpty, &tokens);
    LOG_IF(WARNING, tokens.size() > 1u)
    << "Selected more than one camera for motion tracking. Consider only "
    << "using a single camera if latency issues develop.";

    is_camera_idx_active_in_motion_tracking_.resize(num_cameras, false);//初始化一个相机id的集合，用来指定哪个相机要被追踪
    for (const std::string& camera_id_str : tokens)
    {
        const int camera_idx = std::stoi(camera_id_str);//如果这个相机是被指定要用来追踪的，那么就将这个相机id对应的追踪状态设成true
        CHECK_GE(camera_idx, 0);
        CHECK_LT(camera_idx, static_cast<int>(num_cameras));
        is_camera_idx_active_in_motion_tracking_[camera_idx] = true;
    }

    // Build NCamera of active cameras.
    std::vector<aslam::Camera::Ptr> active_cameras;
    aslam::TransformationVector active_T_C_Bs;
    for (size_t idx = 0u; idx < is_camera_idx_active_in_motion_tracking_.size();//遍历
         ++idx)
    {//
        if (is_camera_idx_active_in_motion_tracking_[idx] == false)
        {
            continue;
        }
        active_cameras.emplace_back(//就是push这个相机的参数
                camera_calibration.getCameraShared(idx)->clone());
        active_T_C_Bs.emplace_back(camera_calibration.get_T_C_B(idx));//这个相机相对于imu的相对位姿
    }
    aslam::NCameraId id;
    common::generateId<aslam::NCameraId>(&id);//产生一个Ncamera的id
    aslam::NCamera motion_tracking_ncamera(//新建了一个多相机系统
            id, active_T_C_Bs, active_cameras, "Cameras active for motion tracking.");

    // Construct ROVIO interface using only the active cameras.
    rovio_interface_.reset(//输入多相机系统，
            constructAndConfigureRovio(motion_tracking_ncamera, imu_sigmas));
    rovio_interface_->setEnablePatchUpdateOutput(false);//不进行块和特征点的输出
    rovio_interface_->setEnableFeatureUpdateOutput(false);
}

void RovioFlow::attachToMessageFlow(message_flow::MessageFlow* flow)
{
  CHECK_NOTNULL(flow);
  static constexpr char kSubscriberNodeName[] = "RovioFlow";

  //所有数据输入订阅者都被放在一个排他性组中，这样所有消息(cam、imu、本地化)的交付顺序都对应于发布顺序，并且不能留下任何传感器。
  // All data input subscribers are put in an exclusivity group such that the
  // delivery ordering for all messages (cam, imu, localization) are
  // corresponding to the publishing order and no sensor can be left behind.
  message_flow::DeliveryOptions rovio_subscriber_options;//初始化rovio订阅选项
  rovio_subscriber_options.exclusivity_group_id =
      kExclusivityGroupIdRovioSensorSubscribers;
kExclusivityGroupIdRovioSensorSubscribers
  // Input IMU.
  // // The node name is just used to print human-readable queue statistics. It
    //  // has no meaning as an identifier internally.
    //  template <typename MessageTopicDefinition>
    //  void registerSubscriber(
    //      const std::string& subscriber_node_name,
    //      const DeliveryOptions& delivery_options,
    //      const SubscriberCallback<MessageTopicDefinition>& callback);
  flow->registerSubscriber<message_flow_topics::IMU_MEASUREMENTS>(//接收imu消息
      kSubscriberNodeName, rovio_subscriber_options,
      [this](const vio::ImuMeasurement::ConstPtr& imu)
      {
        // Do not apply the predictions but only queue them. They will be
        // applied before the next update.
        const bool measurement_accepted =
            this->rovio_interface_->processImuUpdate(//输入的是加速度，陀螺仪数据，时间戳
                imu->imu_data.head<3>(), imu->imu_data.tail<3>(),
                aslam::time::to_seconds(imu->timestamp),
                FLAGS_rovio_update_filter_on_imu);

        LOG_IF(
            WARNING, !measurement_accepted && rovio_interface_->isInitialized())
            << "ROVIO rejected IMU measurement. Latency is too large.";
      });
  // Input camera.
//    struct ImageMeasurement
//    {
//        MAPLAB_POINTER_TYPEDEFS(ImageMeasurement);
//        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//        int64_t timestamp;
//        int camera_index;
//        cv::Mat image;
//
//        ImageMeasurement()
//                : timestamp(aslam::time::getInvalidTime()), camera_index(-1) {}
//    };
  flow->registerSubscriber<message_flow_topics::IMAGE_MEASUREMENTS>(
      kSubscriberNodeName, rovio_subscriber_options,
      [this](const vio::ImageMeasurement::ConstPtr& image)
      {
        const size_t cam_idx = image->camera_index;
        CHECK_LT(cam_idx, is_camera_idx_active_in_motion_tracking_.size());
        if (is_camera_idx_active_in_motion_tracking_[cam_idx] == false) {
          // Skip this image, as the camera was marked as inactive.
          return;
        }

        const bool measurement_accepted =
            this->rovio_interface_->processImageUpdate(
                image->camera_index, image->image,
                aslam::time::to_seconds(image->timestamp));
        LOG_IF(
            WARNING, !measurement_accepted && rovio_interface_->isInitialized())
            << "ROVIO rejected image measurement. Latency is too large.";
      });
  // Input localization updates.
  flow->registerSubscriber<message_flow_topics::LOCALIZATION_RESULT>(
      kSubscriberNodeName, rovio_subscriber_options,
      [this](const vio::LocalizationResult::ConstPtr& localization_result) {
        CHECK(localization_result);
        // ROVIO coordinate frames:
        //  - J: Inertial frame of pose update
        //  - V: Body frame of pose update sensor
        const Eigen::Vector3d JrJV =
            localization_result->T_G_I_lc_pnp.getPosition();
        const kindr::RotationQuaternionPD qJV(
            localization_result->T_G_I_lc_pnp.getRotation().toImplementation());
        const bool measurement_accepted =
            this->rovio_interface_->processGroundTruthUpdate(
                JrJV, qJV,
                aslam::time::to_seconds(localization_result->timestamp));
        LOG_IF(
            WARNING, !measurement_accepted && rovio_interface_->isInitialized())
            << "ROVIO rejected localization update at time="
            << localization_result->timestamp << ". Latency is too large; "
            << "consider reducing the localization rate.";
      });

  // Output ROVIO estimates.
  publish_rovio_estimates_ =
      flow->registerPublisher<message_flow_topics::ROVIO_ESTIMATES>();
  CHECK(rovio_interface_);
  rovio_interface_->registerStateUpdateCallback(
      std::bind(&RovioFlow::processRovioUpdate, this, std::placeholders::_1));
}

void RovioFlow::processRovioUpdate(const rovio::RovioState& state) {
  if (!state.getIsInitialized()) {
    LOG(WARNING) << "ROVIO not yet initialized. Discarding state update.";
    return;
  }

  // ROVIO coordinate frames:
  //  - I: Inertial frame of pose update
  //  - V: Body frame of pose update sensor
  //  - W: Inertial frame of odometry
  //  - B: IMU-coordinate frame
  // ROVIO and maplab both use passive Hamilton quaternion convention; no
  // conversion is necessary.
  aslam::Transformation T_M_I = aslam::Transformation(
      state.get_qBW().inverted().toImplementation(), state.get_WrWB());
  ensurePositiveQuaternion(&T_M_I.getRotation());
  const Eigen::Vector3d v_M = T_M_I.getRotation().rotate(state.get_BvB());

  RovioEstimate::Ptr rovio_estimate(new RovioEstimate);
  // VIO states.
  rovio_estimate->timestamp_s = state.getTimestamp();
  rovio_estimate->vinode.set_T_M_I(T_M_I);
  rovio_estimate->vinode.set_v_M_I(v_M);
  rovio_estimate->vinode.setAccBias(state.getAcb());
  rovio_estimate->vinode.setGyroBias(state.getGyb());

  // Optional localization state.
  rovio_estimate->has_T_G_M = state.getHasInertialPose();
  if (state.getHasInertialPose()) {
    aslam::Transformation T_G_M = aslam::Transformation(
        state.get_qWI().inverted().toImplementation(), state.get_IrIW());
    ensurePositiveQuaternion(&T_G_M.getRotation());
    rovio_estimate->T_G_M = T_G_M;
  }
  publish_rovio_estimates_(rovio_estimate);
}
}  // namespace rovioli
