#include "rovioli/imu-camera-synchronizer.h"

#include <aslam/pipeline/visual-pipeline-null.h>
#include <maplab-common/conversions.h>

DEFINE_int64(//
    vio_nframe_sync_tolerance_ns, 500000,//0.0005,这也太小了
    "Tolerance of the timestamps of two images to consider them as "
    "part of a single n-frame [ns].");
DEFINE_double(
    vio_nframe_sync_max_output_frequency_hz, 10.0,
    "Maximum output frequency of the synchronized IMU-NFrame structures "
    "from the synchronizer.");

namespace rovioli {

ImuCameraSynchronizer::ImuCameraSynchronizer(
    const aslam::NCamera::Ptr& camera_system)
    : camera_system_(camera_system),
      kImuBufferLengthNanoseconds(aslam::time::seconds(30u)),
      frame_skip_counter_(0u),
      previous_nframe_timestamp_ns_(-1),
      min_nframe_timestamp_diff_ns_(
          kSecondsToNanoSeconds /
          FLAGS_vio_nframe_sync_max_output_frequency_hz),
      initial_sync_succeeded_(false),
      shutdown_(false),
      time_last_imu_message_received_or_checked_ns_(
          aslam::time::nanoSecondsSinceEpoch()),
      time_last_camera_message_received_or_checked_ns_(
          aslam::time::nanoSecondsSinceEpoch()) {
  CHECK(camera_system_ != nullptr);
  CHECK_GT(FLAGS_vio_nframe_sync_max_output_frequency_hz, 0.);

  // Initialize the pipeline.
  static constexpr bool kCopyImages = false;
  std::vector<aslam::VisualPipeline::Ptr> mono_pipelines;//初始化pipelines
  for (size_t camera_idx = 0; camera_idx < camera_system_->getNumCameras();
       ++camera_idx)
  {
    mono_pipelines.emplace_back(
        new aslam::NullVisualPipeline(//默认不保存地图
            camera_system_->getCameraShared(camera_idx), kCopyImages));
  }

  const int kNFrameToleranceNs = FLAGS_vio_nframe_sync_tolerance_ns;
  constexpr size_t kNumThreads = 1u;
  visual_pipeline_.reset(//生成inputframe
      new aslam::VisualNPipeline(
          kNumThreads, mono_pipelines, camera_system_, camera_system_,
          kNFrameToleranceNs));//多少纳秒就认为是同一帧

  imu_buffer_.reset(
      new vio_common::ImuMeasurementBuffer(kImuBufferLengthNanoseconds));

  check_if_messages_are_incomfing_thread_ = std::thread(
      &ImuCameraSynchronizer::checkIfMessagesAreIncomingWorker, this);//检查是否要消息进入
  process_thread_ =
      std::thread(&ImuCameraSynchronizer::processDataThreadWorker, this);
}

ImuCameraSynchronizer::~ImuCameraSynchronizer() {
  shutdown();
}

//将这些N相机整合成Nframe
void ImuCameraSynchronizer::addCameraImage(
    size_t camera_index, const cv::Mat& image, int64_t timestamp) {
  constexpr int kMaxNFrameQueueSize = 50;
  CHECK(visual_pipeline_ != nullptr);
  time_last_camera_message_received_or_checked_ns_ =
      aslam::time::nanoSecondsSinceEpoch();//?,这里为啥记录的是当前的系统时间
  if (!visual_pipeline_->processImageBlockingIfFull(//输入:哪一个相机，对应的图片，这个图片对应的时间戳,最大相机队列
          camera_index, image, timestamp, kMaxNFrameQueueSize)) {
    shutdown();
  }
}


//添加imu的测量,在imu_buffer中的buffer_添加imu数据
void ImuCameraSynchronizer::addImuMeasurements(
    const Eigen::Matrix<int64_t, 1, Eigen::Dynamic>& timestamps_nanoseconds,
    const Eigen::Matrix<double, 6, Eigen::Dynamic>& imu_measurements) {
  CHECK(imu_buffer_ != nullptr);
  time_last_imu_message_received_or_checked_ns_ =
      aslam::time::nanoSecondsSinceEpoch();
  imu_buffer_->addMeasurements(timestamps_nanoseconds, imu_measurements);
}

void ImuCameraSynchronizer::checkIfMessagesAreIncomingWorker()
{
  constexpr int kMaxTimeBeforeWarningS = 5;//最大等待时间是5秒
  const int64_t kMaxTimeBeforeWarningNs =
      aslam::time::secondsToNanoSeconds(kMaxTimeBeforeWarningS);
  while (true)
  {
    std::unique_lock<std::mutex> lock(mutex_check_if_messages_are_incoming_);
    const bool shutdown_requested = cv_shutdown_.wait_for(
        lock, std::chrono::seconds(kMaxTimeBeforeWarningS),
        [this]() { return shutdown_.load(); });
    if (shutdown_requested) {
      return;
    }

    const int64_t current_time_ns = aslam::time::nanoSecondsSinceEpoch();
    LOG_IF(
        WARNING,
        current_time_ns - time_last_imu_message_received_or_checked_ns_ >
            kMaxTimeBeforeWarningNs)
        << "No IMU messages have been received in the last "
        << kMaxTimeBeforeWarningS
        << " seconds. Check for measurement drops or if the topic is properly "
        << "set in the maplab IMU configuration file";
    LOG_IF(
        WARNING,
        current_time_ns - time_last_camera_message_received_or_checked_ns_ >
            kMaxTimeBeforeWarningNs)
        << "No camera messages have been received in the last "
        << kMaxTimeBeforeWarningS
        << " seconds. Check for measurement drops or if the topic is properly "
        << "set in the camera configuration file";
    time_last_imu_message_received_or_checked_ns_ = current_time_ns;
    time_last_camera_message_received_or_checked_ns_ = current_time_ns;
  }
}

//在这里发出同步的SynchronizedNFrameImu
void ImuCameraSynchronizer::processDataThreadWorker() {
  while (!shutdown_) {
    aslam::VisualNFrame::Ptr new_nframe;
    if (!visual_pipeline_->getNextBlocking(&new_nframe)) {
      // Shutdown.
      return;
    }

    // Block the previous nframe timestamp so that no other thread can use it.
    // It should wait till this iteration is done.
    std::unique_lock<std::mutex> lock(m_previous_nframe_timestamp_ns_);//阻塞前一个nframe时间戳，这样其他线程就不能使用它。它应该等到这个迭代完成。
//删除一些第一个nframe，因为可能有不完整的IMU数据。
    // Drop few first nframes as there might have incomplete IMU data.
    const int64_t current_frame_timestamp_ns =
        new_nframe->getMinTimestampNanoseconds();//angina
    if (frame_skip_counter_ < kFramesToSkipAtInit) {
      ++frame_skip_counter_;
      previous_nframe_timestamp_ns_ = current_frame_timestamp_ns;
      continue;
    }

    // Throttle the output rate of VisualNFrames to reduce the rate of which
    // the following nodes are running (e.g. tracker).
    CHECK_GE(previous_nframe_timestamp_ns_, 0);
    if (new_nframe->getMinTimestampNanoseconds() -//太近的就不去发出了
            previous_nframe_timestamp_ns_ <
        min_nframe_timestamp_diff_ns_)
    {//这么说这里的阈值也会影响重定位输出频率
      continue;
    }

    vio::SynchronizedNFrameImu::Ptr new_imu_nframe_measurement(//新建一个多相机状态
        new vio::SynchronizedNFrameImu);
    new_imu_nframe_measurement->nframe = new_nframe;//添加新的多相机

    // Wait for the required IMU data.
    CHECK(aslam::time::isValidTime(previous_nframe_timestamp_ns_));
    const int64_t kWaitTimeoutNanoseconds = aslam::time::milliseconds(50);
    vio_common::ImuMeasurementBuffer::QueryResult result;
    bool skip_frame = false;
    //得到这两帧之间的所有的imu数值，然后两帧的时刻上imu也会进行插值
    while ((result = imu_buffer_->getImuDataInterpolatedBordersBlocking(
                previous_nframe_timestamp_ns_, current_frame_timestamp_ns,
                kWaitTimeoutNanoseconds,
                &new_imu_nframe_measurement->imu_timestamps,
                &new_imu_nframe_measurement->imu_measurements)) !=
           vio_common::ImuMeasurementBuffer::QueryResult::kDataAvailable) {
      if (result ==//这是由系统来决定的
          vio_common::ImuMeasurementBuffer::QueryResult::kQueueShutdown) {
        // Shutdown.
        return;
      }
      if (result ==
          vio_common::ImuMeasurementBuffer::QueryResult::kDataNeverAvailable) {
        LOG(ERROR) << "Camera/IMU data out-of-order. This might be okay during "
                      "initialization.";
        CHECK(!initial_sync_succeeded_)
            << "Some synced IMU-camera frames were"
            << "already published. This will lead to map inconsistency.";

        // Skip this frame, but also advanced the previous frame timestamp.//跳过这个时间戳
        previous_nframe_timestamp_ns_ = current_frame_timestamp_ns;
        skip_frame = true;
        break;
      }

      if (result ==
          vio_common::ImuMeasurementBuffer::QueryResult::kDataNotYetAvailable) {
        LOG(WARNING) << "NFrame-IMU synchronization timeout. IMU measurements "
                     << "lag behind. Dropping this nframe.";
        // Skip this frame.
        skip_frame = true;
        break;
      }

      if (result == vio_common::ImuMeasurementBuffer::QueryResult::
                        kTooFewMeasurementsAvailable) {//两个图片之间的imu数据小于3个就会认为是不行的
        LOG(WARNING) << "NFrame-IMU synchronization: Too few IMU measurements "
                     << "available between the previous and current nframe. "
                     << "Dropping this nframe.";
        // Skip this frame.
        skip_frame = true;
        break;
      }
    }

    if (skip_frame)
    {//如果imu的数据不行，一样会跳过这帧
      continue;
    }

    previous_nframe_timestamp_ns_ = current_frame_timestamp_ns;
    // Manually unlock the mutex as the previous nframe timestamp can be
    // consumed by the next iteration.
    lock.unlock();

    // All the synchronization succeeded so let's mark we will publish
    // the frames now. Any IMU data drops after this point mean that the map
    // is inconsistent.
    initial_sync_succeeded_ = true;

    std::lock_guard<std::mutex> callback_lock(m_nframe_callbacks_);
    for (const std::function<void(const vio::SynchronizedNFrameImu::Ptr&)>&
             callback : nframe_callbacks_) {
      callback(new_imu_nframe_measurement);
    }
  }
}

void ImuCameraSynchronizer::registerSynchronizedNFrameImuCallback(
    const std::function<void(const vio::SynchronizedNFrameImu::Ptr&)>&
        callback) {
  std::lock_guard<std::mutex> lock(m_nframe_callbacks_);
  CHECK(callback);
  nframe_callbacks_.push_back(callback);
}

void ImuCameraSynchronizer::shutdown() {
  shutdown_ = true;
  visual_pipeline_->shutdown();
  imu_buffer_->shutdown();
  if (process_thread_.joinable()) {
    process_thread_.join();
  }
  cv_shutdown_.notify_all();
  if (check_if_messages_are_incomfing_thread_.joinable()) {
    check_if_messages_are_incomfing_thread_.join();
  }
}

}  // namespace rovioli
