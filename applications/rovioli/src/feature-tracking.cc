#include "rovioli/feature-tracking.h"

#include <maplab-common/conversions.h>

namespace rovioli {

    //输入多相机系统，imu传感器信息
FeatureTracking::FeatureTracking(
    const aslam::NCamera::Ptr& camera_system, const vi_map::Imu& imu_sensor)
    : camera_system_(camera_system),//多相机系统
      imu_sensor_(imu_sensor),//imu传感器
      current_imu_bias_(Eigen::Matrix<double, 6, 1>::Zero()),//当前imu的两个bias信息
      current_imu_bias_timestamp_nanoseconds_(aslam::time::getInvalidTime()),//当前imu的时间戳
      previous_nframe_timestamp_ns_(-1),//
      tracker_(camera_system)//feature_tracking::VOFeatureTrackingPipeline,用多相机系统来初始化
      {
  CHECK(camera_system_ != nullptr);
}

//跟踪线程,返回已经同步的多相机
bool FeatureTracking::trackSynchronizedNFrameImuCallback(
    const vio::SynchronizedNFrameImu::Ptr& synced_nframe_imu)
{
    CHECK(synced_nframe_imu != nullptr);
    std::lock_guard<std::mutex> lock(m_previous_synced_nframe_imu_);

    //第一帧不包含第一次回调的任何跟踪信息，但会在第二次回调中添加。
    // The first frame will not contain any tracking information on the first
    // call, but it will be added in the second call.
    if (previous_synced_nframe_imu_ == nullptr) //只有第一帧才会调用，就是给之前数据去初始化，用多相机中最小的时间戳，多相机系统
    {
        previous_synced_nframe_imu_ = synced_nframe_imu;//将存储当前imu和相机的数据结构赋值给之前的
        previous_nframe_timestamp_ns_ =//n相机中最小的那个时间戳
                synced_nframe_imu->nframe->getMinTimestampNanoseconds();
        return false;
    }

    // Check if the IMU bias is up to date, if not - use zero.
    //看看imu的bais有没有更新,没有更新就设成0
    if (!hasUpToDateImuBias(//这里输入的是多相机系统的最小时间戳
            synced_nframe_imu->nframe->getMinTimestampNanoseconds()))
    {
        LOG(WARNING) << "No bias from the estimator available. Assuming zero bias.";
        std::unique_lock<std::mutex> bias_lock(m_current_imu_bias_);
        current_imu_bias_.setZero();//如果没有更新，就设成0
    }

    // Preintegrate the IMU measurements.
    aslam::Quaternion q_Ikp1_Ik;//用imu测量量来进行预积分，q_Ikp1_Ik是这段时间的旋转,qbkb1
    integrateInterframeImuRotation(
            synced_nframe_imu->imu_timestamps, synced_nframe_imu->imu_measurements,
            &q_Ikp1_Ik);

    CHECK(previous_synced_nframe_imu_ != nullptr);
    aslam::FrameToFrameMatchesList inlier_matches_kp1_k;
    aslam::FrameToFrameMatchesList outlier_matches_kp1_k;
    CHECK_GT(//时间顺序要满足
            synced_nframe_imu->nframe->getMinTimestampNanoseconds(),
            previous_synced_nframe_imu_->nframe->getMinTimestampNanoseconds());
    //输入这段时间内的旋转，当前的多相机状态，之前的多相机状态
    //输出多相机前后匹配的结果
    tracker_.trackFeaturesNFrame(
            q_Ikp1_Ik, synced_nframe_imu->nframe.get(),
            previous_synced_nframe_imu_->nframe.get(), &inlier_matches_kp1_k,
            &outlier_matches_kp1_k);

    previous_synced_nframe_imu_ = synced_nframe_imu;//将这个同步帧就变成了之前的同步帧
    return true;
}

void FeatureTracking::setCurrentImuBias(
    const RovioEstimate::ConstPtr& rovio_estimate)
{
    CHECK(rovio_estimate != nullptr);

    const int64_t bias_timestamp_ns =
            rovio_estimate->timestamp_s * kSecondsToNanoSeconds;

    // Only update the bias if we have a newer measurement.
    std::unique_lock<std::mutex> lock(m_current_imu_bias_);
    if (bias_timestamp_ns > current_imu_bias_timestamp_nanoseconds_) {
        current_imu_bias_timestamp_nanoseconds_ =
                rovio_estimate->timestamp_s * kSecondsToNanoSeconds;
        current_imu_bias_ = rovio_estimate->vinode.getImuBias();
        VLOG(5) << "Updated IMU bias in Pipeline node.";
    } else {
        LOG(WARNING) << "Received an IMU bias estimate that has an earlier "
                     << "timestamp than the previous one. Previous timestamp: "
                     << current_imu_bias_timestamp_nanoseconds_
                     << "ns, received timestamp: " << bias_timestamp_ns << "ns.";
    }
}

bool FeatureTracking::hasUpToDateImuBias(
    const int64_t current_timestamp_ns) const
{
    std::unique_lock<std::mutex> lock(m_current_imu_bias_);
    if (current_imu_bias_timestamp_nanoseconds_ == -1) {
        // No bias was ever set.
        return false;
    }
    //10s没有更新bias就认为是有问题的
    constexpr int64_t kImuBiasAgeThresholdNs = 10 * kSecondsToNanoSeconds;
    if (current_timestamp_ns - current_imu_bias_timestamp_nanoseconds_ >
        kImuBiasAgeThresholdNs) {
        // The bias estimate is not up to date.
        return false;
    }
    return true;
}

void FeatureTracking::integrateInterframeImuRotation(
    const Eigen::Matrix<int64_t, 1, Eigen::Dynamic>& imu_timestamps,
    const Eigen::Matrix<double, 6, Eigen::Dynamic>& imu_measurements,
    aslam::Quaternion* q_Ikp1_Ik) const
{
    CHECK_NOTNULL(q_Ikp1_Ik);
    CHECK_GT(imu_timestamps.cols(), 2);
    CHECK_EQ(imu_measurements.cols(), imu_timestamps.cols());
//这段时间内的旋转
    q_Ikp1_Ik->setIdentity();
    for (int i = 1; i < imu_measurements.cols(); ++i)
    {//遍历每一个imu
        const double delta_s =//dt
                (imu_timestamps(i) - imu_timestamps(i - 1)) * kNanosecondsToSeconds;//转成秒
        CHECK_GT(delta_s, 0);
        std::unique_lock<std::mutex> bias_lock(m_current_imu_bias_);
        const Eigen::Vector3d gyro_measurement =//这里这个是真值，减了bias的
                imu_measurements.col(i).tail<3>() - current_imu_bias_.tail<3>();
        bias_lock.unlock();

        *q_Ikp1_Ik =//所以这里的q就应该是qb1bk,它这里直接认为是角轴
                *q_Ikp1_Ik * aslam::Quaternion::exp(gyro_measurement * delta_s);//将微小的角度扰动转成四元数表达形式,hamilton形式
    }
    // We actually need to inverse the rotation so that transform from Ikp1 to Ik.
    *q_Ikp1_Ik = q_Ikp1_Ik->inverse();//但是实际上是qbkb1，也就是JPL形式，这个还需要再议，一会儿看看
}

}  // namespace rovioli
