#ifndef ROVIOLI_DATASOURCE_H_
#define ROVIOLI_DATASOURCE_H_

#include <functional>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <maplab-common/macros.h>
#include <vio-common/vio-types.h>

namespace rovioli {

#define DECLARE_SENSOR_CALLBACK(SENSOR_NAME, MEASUREMENT_TYPE)             \
 public: /* NOLINT */                                                      \
  typedef std::function<void(MEASUREMENT_TYPE)> SENSOR_NAME##Callback;     \
  /* NOLINT */ void register##SENSOR_NAME##Callback(                       \
      const SENSOR_NAME##Callback& cb) { /* NOLINT */                      \
    CHECK(cb);                                                             \
    SENSOR_NAME##_callbacks_.emplace_back(cb);                             \
  }                                                                        \
  void invoke##SENSOR_NAME##Callbacks(const MEASUREMENT_TYPE& measurement) \
      const
      {                                                              \
    for (const SENSOR_NAME##Callback& callback /* NOLINT */ :              \
         SENSOR_NAME##_callbacks_) {                                       \
      callback(measurement);                                               \
    }                                                                      \
  }                                                                        \
                                                                           \
 private: /* NOLINT */                                                     \
  std::vector<SENSOR_NAME##Callback> SENSOR_NAME##_callbacks_;

class CallbackManager {

//    #define DECLARE_SENSOR_CALLBACK(SENSOR_NAME, MEASUREMENT_TYPE)             \
// public: /* NOLINT */                                                      \
//  typedef std::function<void(MEASUREMENT_TYPE)> SENSOR_NAME##Callback; //定义一个函数叫 ImageCallback   \
//
//void registerImageCallback( const ImageCallback& cb)
//        {

        //std::function<void(const typename MessageTopicDefinition::message_type&)>其实就是ImageCallback;
//
//
//        }
//  /* NOLINT */ void register##SENSOR_NAME##Callback(                       \
//      const SENSOR_NAME##Callback& cb)
// { /* NOLINT */                      \
//    CHECK(cb);                                                             \
//    SENSOR_NAME##_callbacks_.emplace_back(cb);    //注册各种回调函数，然后把函数push进  SENSOR_NAME##_callbacks_里                       \
//  }                                                                        \
//  void invoke##SENSOR_NAME##Callbacks(const MEASUREMENT_TYPE& measurement) \
//      const {                                                              \
//    for (const SENSOR_NAME##Callback& callback /* NOLINT */ :              \
//         SENSOR_NAME##_callbacks_) {                                       \
//      callback(measurement);                                               \
//    }                                                                      \
//  }                                                                        \
//                                                                           \
// private: /* NOLINT */                                                     \
//  std::vector<SENSOR_NAME##Callback> SENSOR_NAME##_callbacks_;



  DECLARE_SENSOR_CALLBACK(Image, vio::ImageMeasurement::Ptr);
  DECLARE_SENSOR_CALLBACK(Imu, vio::ImuMeasurement::Ptr);
};

class DataSource : public CallbackManager {
 public:
  MAPLAB_POINTER_TYPEDEFS(DataSource);
  MAPLAB_DISALLOW_EVIL_CONSTRUCTORS(DataSource);

  virtual ~DataSource() {}

  // Has all data been released to the output queues.
  virtual void startStreaming() = 0;
  virtual void shutdown() = 0;

  virtual bool allDataStreamed() const = 0;
  virtual std::string getDatasetName() const = 0;

  virtual void registerEndOfDataCallback(const std::function<void()>& cb) {
    CHECK(cb);
    end_of_data_callbacks_.emplace_back(cb);
  }
  void invokeEndOfDataCallbacks() const {
    for (const std::function<void()>& cb : end_of_data_callbacks_) {
      cb();
    }
  }

  // If this is the first timestamp we receive, we store it and shift all
  // subsequent timestamps. Will return false for any timestamps that are
  // smaller than the first timestamp received.
  //如果这是我们接收到的第一个时间戳，我们将存储它并转移所有后续的时间戳。对于小于接收到的第一个时间戳的任何时间戳，将返回false。
  bool shiftByFirstTimestamp(int64_t* timestamp_ns)
  {
    CHECK_NOTNULL(timestamp_ns);
    CHECK_GE(*timestamp_ns, 0);
    {
      std::lock_guard<std::mutex> lock(timestamp_mutex_);
      if (timestamp_at_start_ns_ == -1)
      {//这个是第一个时间戳到来时要干的事情，
        timestamp_at_start_ns_ = *timestamp_ns;
        *timestamp_ns = 0;
        VLOG(2)
            << "Set the first timestamp that was received to "
            << timestamp_at_start_ns_
            << "ns, all subsequent timestamp will be shifted by that amount.";
      } else {
        if (*timestamp_ns < timestamp_at_start_ns_) {
          LOG(WARNING) << "Received timestamp that is earlier than the first "
                       << "timestamp of the data source! First timestamp: "
                       << timestamp_at_start_ns_
                       << "ns, received timestamp: " << *timestamp_ns << "ns.";
          return false;//如果新来的时间戳小于当前的时间戳，那么会
        }
        *timestamp_ns = *timestamp_ns - timestamp_at_start_ns_;//减去漂移的时间，就是整体时间前移
      }
    }

    CHECK_GE(timestamp_at_start_ns_, 0);
    CHECK_GE(*timestamp_ns, 0);
    return true;
  }

 protected:
  DataSource() = default;

 private:
  std::vector<std::function<void()>> end_of_data_callbacks_;

  std::mutex timestamp_mutex_;
  int64_t timestamp_at_start_ns_ = -1;
};

}  // namespace rovioli

#endif  // ROVIOLI_DATASOURCE_H_
