#include "vio-common/imu-measurements-buffer.h"

#include <algorithm>
#include <condition_variable>
#include <memory>
#include <mutex>

#include <Eigen/Dense>
#include <aslam/common/memory.h>
#include <aslam/common/statistics/statistics.h>
#include <aslam/common/time.h>
#include <glog/logging.h>

#include "vio-common/vio-types.h"

namespace vio_common {

    //查询imu数据是否可以获取
ImuMeasurementBuffer::QueryResult ImuMeasurementBuffer::isDataAvailableUpToImpl(
    int64_t timestamp_ns_from, int64_t timestamp_ns_to) const
    {
  CHECK_LT(timestamp_ns_from, timestamp_ns_to);

  if (buffer_.empty()) {//数据无法得到
    return QueryResult::kDataNotYetAvailable;
  }

  vio::ImuMeasurement value;
  const bool get_latest_value_success = buffer_.getNewestValue(&value);//得到最近的一个imu
  if (get_latest_value_success && value.timestamp < timestamp_ns_to) {//如果imu时间戳要比后一帧图片的时间戳延后，、
      // 那么说明imu需要的数据还没有来
    return QueryResult::kDataNotYetAvailable;
  }

  //imu的前后时间戳应该要包含我两个图片时间戳的时间序列的
  if (buffer_.getOldestValue(&value) && (value.timestamp >= timestamp_ns_to ||timestamp_ns_from < value.timestamp)) {
    return QueryResult::kDataNeverAvailable;
  }
  return QueryResult::kDataAvailable;
}

void ImuMeasurementBuffer::linearInterpolate(
    const int64_t t0, const vio::ImuData& y0, const int64_t t1,
    const vio::ImuData& y1, const int64_t t, vio::ImuData* y) {
  CHECK_NOTNULL(y);
  CHECK(aslam::time::isValidTime(t));
  CHECK(aslam::time::isValidTime(t0));
  CHECK(aslam::time::isValidTime(t1));
  CHECK_LE(t0, t);
  CHECK_LE(t, t1);
  *y = t0 == t1 ? y0 : y0 + (y1 - y0) * static_cast<double>(t - t0) /
                                static_cast<double>(t1 - t0);
}

ImuMeasurementBuffer::QueryResult
ImuMeasurementBuffer::getImuDataInterpolatedBorders(
    int64_t timestamp_ns_from, int64_t timestamp_ns_to,
    Eigen::Matrix<int64_t, 1, Eigen::Dynamic>* imu_timestamps,
    Eigen::Matrix<double, 6, Eigen::Dynamic>* imu_measurements)
{
    CHECK_NOTNULL(imu_timestamps);
    CHECK_NOTNULL(imu_measurements);

    QueryResult query_result =
            isDataAvailableUpToImpl(timestamp_ns_from, timestamp_ns_to);
    if (query_result != QueryResult::kDataAvailable) {
        imu_timestamps->resize(Eigen::NoChange, 0);
        imu_measurements->resize(Eigen::NoChange, 0);
        return query_result;
    }
//只有当数据可以利用时才会进行
    // Copy the data with timestamp_up_to <= timestamps_buffer from the buffer.
    Aligned<std::vector, vio::ImuMeasurement> between_values;
    CHECK(//找到图片时间戳之间的imu数据，两头不等于
            buffer_.getValuesBetweenTimes(
                    timestamp_ns_from, timestamp_ns_to, &between_values));

    if (between_values.empty()) {
        LOG(WARNING) << "Too few IMU measurements available between time "
                     << timestamp_ns_from << "[ns] and " << timestamp_ns_to
                     << "[ns].";
        imu_timestamps->resize(Eigen::NoChange, 0);
        imu_measurements->resize(Eigen::NoChange, 0);
        return QueryResult::kTooFewMeasurementsAvailable;
    }

    // The first and last index will be replaced with the interpolated values.//这里明白了为啥上面不要两头的imu数据
    const size_t num_measurements = between_values.size() + 2u;//这里的2是考虑到了图片两端的时间戳
    imu_timestamps->resize(Eigen::NoChange, num_measurements);
    imu_measurements->resize(Eigen::NoChange, num_measurements);

    for (size_t idx = 1u; idx < num_measurements - 1u; ++idx) {//先把中间数据给了
        (*imu_timestamps)(idx) = between_values[idx - 1u].timestamp;
        (*imu_measurements).col(idx) = between_values[idx - 1u].imu_data;
    }

    // Interpolate border values.
    int64_t pre_border_timestamp, post_border_timestamp;
    vio::ImuMeasurement pre_border_value, post_border_value;
    vio::ImuData interpolated_measurement;

    // Interpolate lower border.
    CHECK(
            buffer_.getValueAtOrBeforeTime(
                    timestamp_ns_from, &pre_border_timestamp, &pre_border_value));
    CHECK_EQ(pre_border_timestamp, pre_border_value.timestamp);
    CHECK(
            buffer_.getValueAtOrAfterTime(
                    timestamp_ns_from, &post_border_timestamp, &post_border_value));
    CHECK_EQ(post_border_timestamp, post_border_value.timestamp);
    linearInterpolate(pre_border_value.timestamp, pre_border_value.imu_data,
                      post_border_value.timestamp, post_border_value.imu_data,
                      timestamp_ns_from, &interpolated_measurement);
    (*imu_timestamps).leftCols<1>()(0) = timestamp_ns_from;
    (*imu_measurements).leftCols<1>() = interpolated_measurement;

    // Interpolate upper border.
    CHECK(
            buffer_.getValueAtOrBeforeTime(
                    timestamp_ns_to, &pre_border_timestamp, &pre_border_value));
    CHECK_EQ(pre_border_timestamp, pre_border_value.timestamp);
    CHECK(
            buffer_.getValueAtOrAfterTime(
                    timestamp_ns_to, &post_border_timestamp, &post_border_value));
    CHECK_EQ(post_border_timestamp, post_border_value.timestamp);
    linearInterpolate(pre_border_value.timestamp, pre_border_value.imu_data,
                      post_border_value.timestamp, post_border_value.imu_data,
                      timestamp_ns_to, &interpolated_measurement);
    (*imu_timestamps).rightCols<1>()(0) = timestamp_ns_to;
    (*imu_measurements).rightCols<1>() = interpolated_measurement;

    return query_result;
}

//首先调用这个
ImuMeasurementBuffer::QueryResult
ImuMeasurementBuffer::getImuDataInterpolatedBordersBlocking(
    int64_t timestamp_ns_from, int64_t timestamp_ns_to,
    int64_t wait_timeout_nanoseconds,
    Eigen::Matrix<int64_t, 1, Eigen::Dynamic>* imu_timestamps,
    Eigen::Matrix<double, 6, Eigen::Dynamic>* imu_measurements)
    {
        CHECK_NOTNULL(imu_timestamps);
        CHECK_NOTNULL(imu_measurements);

        // Wait for the IMU buffer to contain the required measurements within a
        // timeout.
        //等待IMU缓冲区在超时内包含所需的测量值。
        int64_t time_start = aslam::time::nanoSecondsSinceEpoch();
        int64_t total_elapsed_time_nanoseconds = aslam::time::getInvalidTime();
        QueryResult query_result;
        {
            std::unique_lock<std::mutex> lock(m_buffer_);
            //如果没有得到imu数据，就等imu数据过来
            while ((query_result = isDataAvailableUpToImpl(timestamp_ns_from, timestamp_ns_to)) !=
                   QueryResult::kDataAvailable)
            {
                cv_new_measurement_.wait_for(
                        lock, std::chrono::nanoseconds(wait_timeout_nanoseconds));

                if (shutdown_) {
                    imu_timestamps->resize(Eigen::NoChange, 0);
                    imu_measurements->resize(Eigen::NoChange, 0);
                    return QueryResult::kQueueShutdown;
                }

                // Check if we hit the max. time allowed to wait for the required data.
                int64_t total_elapsed_time_nanoseconds =
                        aslam::time::nanoSecondsSinceEpoch() - time_start;
                if (total_elapsed_time_nanoseconds >= wait_timeout_nanoseconds) {
                    imu_timestamps->resize(Eigen::NoChange, 0);
                    imu_measurements->resize(Eigen::NoChange, 0);
                    LOG(WARNING) << "Timeout reached while trying to get the requested "
                                 << "IMU data. Requested range: " << timestamp_ns_from
                                 << " to " << timestamp_ns_to << ".";
                    if (query_result == QueryResult::kDataNotYetAvailable) {
                        LOG(WARNING) << "The relevant IMU data is not yet available.";
                    } else if (query_result == QueryResult::kDataNeverAvailable) {
                        LOG(WARNING) << "The relevant IMU data will never be available. "
                                     << "Either the buffer is too small or a sync issue "
                                     << "occurred.";
                    } else {
                        LOG(FATAL) << "Unknown query result error.";
                    }
                    return query_result;
                }
            }
        }
        statistics::StatsCollector imu_pop_time("Wait time for IMU data [ms]");
        imu_pop_time.AddSample(
                aslam::time::to_milliseconds(total_elapsed_time_nanoseconds));
        return getImuDataInterpolatedBorders(//这里会存储所有的图片时间戳之间的imu数据，并且会对前后两个图片的时间戳的时间部分进行插值
                timestamp_ns_from, timestamp_ns_to, imu_timestamps, imu_measurements);
    }

}  // namespace vio_common
