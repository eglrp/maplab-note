#ifndef ROVIOLI_DATASOURCE_FLOW_H_
#define ROVIOLI_DATASOURCE_FLOW_H_
#include <memory>

#include <aslam/cameras/ncamera.h>
#include <message-flow/message-flow.h>
#include <sensors/imu.h>
#include <vio-common/rostopic-settings.h>
#include <vio-common/vio-types.h>

#include "rovioli/datasource-factory.h"
#include "rovioli/flow-topics.h"

namespace rovioli {

class DataSourceFlow {
 public:
  explicit DataSourceFlow(
      const aslam::NCamera& camera_system, const vi_map::Imu& imu_sensor) //输入多相机系统，imu传感器
      {
    datasource_.reset(
        createAndConfigureDataSourcefromGFlags(camera_system, imu_sensor));
    CHECK(datasource_);
  }

  ~DataSourceFlow() {
    shutdown();
  }

  void attachToMessageFlow(message_flow::MessageFlow* flow)
  {

      //flow->registerPublisher<message_flow_topics::IMAGE_MEASUREMENTS>());\

      CHECK_NOTNULL(flow);//初始化，就是存储各种传感器callback函数,都存到对应的SENSOR_NAME##_callbacks_里
      datasource_->registerImageCallback(
              flow->registerPublisher<message_flow_topics::IMAGE_MEASUREMENTS>());
      datasource_->registerImuCallback(
              flow->registerPublisher<message_flow_topics::IMU_MEASUREMENTS>());
  }

  void startStreaming() {
    datasource_->startStreaming();
  }

  void shutdown() {
    datasource_->shutdown();
  }

  void registerEndOfDataCallback(const std::function<void()>& cb) {
    CHECK(cb);
    datasource_->registerEndOfDataCallback(cb);
  }

 private:
  std::unique_ptr<DataSource> datasource_;
};

}  // namespace rovioli

#endif  // ROVIOLI_DATASOURCE_FLOW_H_
