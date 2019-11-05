#include "rovioli/datasource-rostopic.h"

#include <string>

#include <aslam/common/time.h>
#include <boost/bind.hpp>
#include <maplab-common/accessors.h>
#include <vio-common/rostopic-settings.h>

#include "rovioli/ros-helpers.h"

DECLARE_bool(rovioli_zero_initial_timestamps);

namespace rovioli
{

DataSourceRostopic::DataSourceRostopic(
    const vio_common::RosTopicSettings& ros_topics)
    : shutdown_requested_(false),//初始化接收话题的选项
      ros_topics_(ros_topics),
      image_transport_(node_handle_) {}

DataSourceRostopic::~DataSourceRostopic() {}

void DataSourceRostopic::startStreaming() {
  registerSubscribers(ros_topics_);
}

void DataSourceRostopic::shutdown() {
  shutdown_requested_ = true;
}

void DataSourceRostopic::registerSubscribers(
    const vio_common::RosTopicSettings& ros_topics)//ros接收消息回调
{
    // Camera subscriber.
    const size_t num_cameras = ros_topics.camera_topic_cam_index_map.size();
    sub_images_.reserve(num_cameras);

    for (const std::pair<std::string, size_t>& topic_camidx ://接收所有的图片消息
            ros_topics.camera_topic_cam_index_map)
    {
        boost::function<void(const sensor_msgs::ImageConstPtr&)> image_callback =//定义一个消息回调函数
                boost::bind(
                        &DataSourceRostopic::imageCallback, this, _1, topic_camidx.second);//_1是消息

        constexpr size_t kRosSubscriberQueueSizeImage = 20u;
        image_transport::Subscriber image_sub = image_transport_.subscribe(
                topic_camidx.first, kRosSubscriberQueueSizeImage, image_callback);//话题名字，队列里图片个数，图片回调函数
        sub_images_.push_back(image_sub);
    }

    // IMU subscriber.
    constexpr size_t kRosSubscriberQueueSizeImu = 1000u;
    boost::function<void(const sensor_msgs::ImuConstPtr&)> imu_callback =
            boost::bind(&DataSourceRostopic::imuMeasurementCallback, this, _1);
    sub_imu_ = node_handle_.subscribe(
            ros_topics.imu_topic, kRosSubscriberQueueSizeImu, imu_callback);
}

void DataSourceRostopic::imageCallback(//rostopic用的图片消息回调函数
    const sensor_msgs::ImageConstPtr& image_message, size_t camera_idx)
{
    if (shutdown_requested_)
    {
        return;
    }

    vio::ImageMeasurement::Ptr image_measurement =//将ros的图片转成需要的数据格式
            convertRosImageToMaplabImage(image_message, camera_idx);

//    void invoke##SENSOR_NAME##Callbacks(const MEASUREMENT_TYPE& measurement) \
//      const
//    {                                                              \
//    for (const SENSOR_NAME##Callback& callback /* NOLINT */ :              \
//         SENSOR_NAME##_callbacks_) {                                       \
//      callback(measurement);                                               \
//    }                                                                      \
//  }

    // Shift timestamps to start at 0.
    if (!FLAGS_rovioli_zero_initial_timestamps ||
        shiftByFirstTimestamp(&(image_measurement->timestamp)))//把时间戳整体前移
    {
        invokeImageCallbacks(image_measurement);//像在发布消息
    }
}

void DataSourceRostopic::imuMeasurementCallback(
    const sensor_msgs::ImuConstPtr& msg) {
  if (shutdown_requested_) {
    return;
  }

  vio::ImuMeasurement::Ptr imu_measurement = convertRosImuToMaplabImu(msg);

  // Shift timestamps to start at 0.
  if (!FLAGS_rovioli_zero_initial_timestamps ||
      shiftByFirstTimestamp(&(imu_measurement->timestamp))) {
    invokeImuCallbacks(imu_measurement);
  }
}

}  // namespace rovioli
