#ifndef ROVIOLI_ROVIO_ESTIMATE_H_
#define ROVIOLI_ROVIO_ESTIMATE_H_

#include <Eigen/Core>
#include <aslam/common/pose-types.h>
#include <maplab-common/macros.h>
#include <vio-common/vio-types.h>

namespace rovioli {
struct RovioEstimate {
  MAPLAB_POINTER_TYPEDEFS(RovioEstimate);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double timestamp_s;//多相机时间戳中最小的那个，单位是秒
  vio::ViNodeState vinode;//vio状态

  aslam::Transformation T_G_M;//当前任务坐标系到世界坐标系的变换
  bool has_T_G_M;//是否有这个变换的信息
};

//因为maplab可以多任务建图，每一个任务都是有自己的坐标系，定义为M，所有的任务又共同在一个世界坐标系下，定义为G
//class ViNodeState {
//private:
//    /// The pose taking points from the body frame to the world frame.
//    aslam::Transformation T_M_I_;//本体系在自己任务坐标系中的位姿
//    /// The velocity (m/s).
//    Eigen::Vector3d v_M_I_;//本体系在自己任务坐标系中的速度
//    /// The accelerometer bias (m/s^2).
//    Eigen::Vector3d acc_bias_;//加速度bias
//    /// The gyroscope bias (rad/s).
//    Eigen::Vector3d gyro_bias_;//陀螺仪bias
//
//    /// Transformation of IMU wrt UTM reference frame.
//    aslam::Transformation T_UTM_I_;
//    /// Transformation of the body frame wrt the UTM reference frame.
//    aslam::Transformation T_UTM_B_;
//    };

}  // namespace rovioli
#endif  // ROVIOLI_ROVIO_ESTIMATE_H_
