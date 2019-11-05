#ifndef VIO_COMMON_IMU_MEASUREMENTS_BUFFER_H_
#define VIO_COMMON_IMU_MEASUREMENTS_BUFFER_H_

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <utility>

#include <Eigen/Dense>
#include <glog/logging.h>
#include <maplab-common/macros.h>
#include <maplab-common/temporal-buffer.h>

#include "vio-common/vio-types.h"

namespace vio_common {
/// \class ImuMeasurementBuffer
/// This buffering class can be used to store a history of IMU measurements. It
/// allows to
/// retrieve a list  of measurements up to a given timestamp. The data is stored
/// in the order
/// it is added. So make sure to add it in correct time-wise order.
class ImuMeasurementBuffer {
 public:
  MAPLAB_POINTER_TYPEDEFS(ImuMeasurementBuffer);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW


  enum class QueryResult {
    /// Query was successful and the data is available.
    ///查询成功，数据可用。
    kDataAvailable,
    /// The required data is not (yet) available. The query timestamp is above
    /// the last IMU sample's time.
//所需的数据还没有。查询时间戳位于最后一个IMU的时间之后。
    kDataNotYetAvailable,
    /// The queried timestamp lies before the first IMU sample in the buffer.
    /// This request will never succeed considering chronological ordering
    /// of the buffer input data.
    //被查询的时间戳位于缓冲区中的第一个IMU样本之前。考虑到缓冲区输入数据的时间顺序，这个请求永远不会成功。
    kDataNeverAvailable,
    /// Queue shutdown.
    kQueueShutdown,
    kTooFewMeasurementsAvailable
  };

  explicit ImuMeasurementBuffer(int64_t buffer_length_ns)
      : buffer_(buffer_length_ns), shutdown_(false) {}
  ~ImuMeasurementBuffer() {
    shutdown();
  }

  /// Shutdown the queue and release all blocked waiters.
  inline void shutdown();
  inline size_t size() const;
  inline void clear();

  /// Add IMU measurement in IMU frame.
  /// (Ordering: accelerations [m/s^2], angular velocities [rad/s])
  inline void addMeasurement(
      int64_t timestamp_nanoseconds, const vio::ImuData& imu_measurement);
  inline void addMeasurements(
      const Eigen::Matrix<int64_t, 1, Eigen::Dynamic>& timestamps_nanoseconds,
      const Eigen::Matrix<double, 6, Eigen::Dynamic>& imu_measurements);

  /// \brief Return a list of the IMU measurements between the specified
  /// timestamps. The
  /// IMU values get interpolated if the queried timestamp does not match a
  /// measurement.
  /// The return code signals if the buffer does not contain data up to the
  /// requested timestamp.
  /// In this case the output matrices will be of size 0.
  /// @param[in] timestamp_from Try to get the IMU measurements from this time
  /// [ns].
  /// @param[in] timestamp_to Try to get the IMU measurements up this time [ns].
  /// @param[out] imu_timestamps List of timestamps. [ns]
  /// @param[out] imu_measurements List of IMU measurements. (Order: [acc,
  /// gyro])
  /// @return Was data removed from the buffer?
  QueryResult getImuDataInterpolatedBorders(
      int64_t timestamp_from, int64_t timestamp_to,
      Eigen::Matrix<int64_t, 1, Eigen::Dynamic>* imu_timestamps,
      Eigen::Matrix<double, 6, Eigen::Dynamic>* imu_measurements);

  /// Try to pop the requested IMU measurements for the duration of
  /// wait_timeout_nanoseconds.
  /// If the requested data is still not available when timeout has been
  /// reached, the method
  /// will return false and no data will be removed from the buffer.
//尝试在wait_timeout_nanosecond期间弹出所请求的IMU测量值。
// 如果超时时请求的数据仍然不可用，则该方法将返回false，并且不会从缓冲区中删除任何数据。获取两个时间图片时间戳之间的所有imu数据，包含了图片两头的
  QueryResult getImuDataInterpolatedBordersBlocking(
      int64_t timestamp_ns_from, int64_t timestamp_ns_to,
      int64_t wait_timeout_nanoseconds,
      Eigen::Matrix<int64_t, 1, Eigen::Dynamic>* imu_timestamps,
      Eigen::Matrix<double, 6, Eigen::Dynamic>* imu_measurements);

  /// Linear interpolation between two imu measurements.
  static void linearInterpolate(
      int64_t x0, const vio::ImuData& y0, int64_t x1, const vio::ImuData& y1,
      int64_t x, vio::ImuData* y);

 private:
  /// Is data available up to this timestamp? Note this function does not lock
  /// the buffers, the
  /// caller must hold the lock.
  QueryResult isDataAvailableUpToImpl(
      int64_t timestamp_ns_from, int64_t timestamp_ns_to) const;

  typedef std::pair<int64_t, vio::ImuMeasurement> BufferElement;
  typedef Eigen::aligned_allocator<BufferElement> BufferAllocator;
  typedef common::TemporalBuffer<vio::ImuMeasurement, BufferAllocator> Buffer;

  Buffer buffer_;
  mutable std::mutex m_buffer_;
  std::condition_variable cv_new_measurement_;
  std::atomic<bool> shutdown_;
};
}  // namespace vio_common
#include <vio-common/internal/imu-measurements-buffer-inl.h>
#endif  // VIO_COMMON_IMU_MEASUREMENTS_BUFFER_H_
