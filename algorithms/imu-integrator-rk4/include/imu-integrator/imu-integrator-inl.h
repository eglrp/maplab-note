#ifndef IMU_INTEGRATOR_IMU_INTEGRATOR_INL_H_
#define IMU_INTEGRATOR_IMU_INTEGRATOR_INL_H_

#include <cmath>
#include <iomanip>
#include <limits>

#include <glog/logging.h>

#include <maplab-common/geometry.h>
#include <maplab-common/quaternion-math.h>

#include "imu-integrator/common.h"

namespace imu_integrator {

//typedef Eigen::Matrix<double, imu_integrator::kStateSize, 1>，在这里ScalarType是double

    //输入当前状态,相邻两次时刻k到k+1的加速度和速度数据的标称值，dt
    //输出下一个状态，误差的传播矩阵，噪声协方差
template <typename ScalarType>
void ImuIntegratorRK4::integrate(
    const Eigen::Matrix<ScalarType, kStateSize, 1>& current_state,
    const Eigen::Matrix<ScalarType, 2 * kImuReadingSize, 1>&
        debiased_imu_readings,
    const ScalarType delta_time_seconds,
    Eigen::Matrix<ScalarType, kStateSize, 1>* next_state,
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize>* next_phi,
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize>* next_cov)
    const {
  // The (next_phi, next_cov) pair is optional; both pointers have to be null to
  // skip calculations.
  LOG_IF(FATAL, static_cast<bool>(next_phi) != static_cast<bool>(next_cov))
      << "next_phi and next_cov have to be either both valid or be null";
  bool calculate_phi_cov = (next_phi != nullptr) && (next_cov != nullptr);

  ScalarType o5 = static_cast<ScalarType>(0.5);//o5是0.5

  next_state->setZero();
  Eigen::Matrix<ScalarType, kImuReadingSize, 1> imu_readings_k1 =//k时刻的imu数据，为了算k1
      debiased_imu_readings.template block<kImuReadingSize, 1>(0, 0);

  Eigen::Matrix<ScalarType, kImuReadingSize, 1> imu_readings_k23;
  interpolateImuReadings(//相邻两次时刻k到k+1的加速度和速度数据的标称值，dt，0.5dt，返回线性插值的imu数据，这里是插值0.5dt处的，为了算K2,K3
      debiased_imu_readings, delta_time_seconds, o5 * delta_time_seconds,
      &imu_readings_k23);

  Eigen::Matrix<ScalarType, kImuReadingSize, 1> imu_readings_k4 =//后一时刻的状态,为了算k4
      debiased_imu_readings.template block<kImuReadingSize, 1>(
          kImuReadingSize, 0);

  //RK4离散积分
  Eigen::Matrix<ScalarType, kStateSize, 1> state_der1;//K1
  Eigen::Matrix<ScalarType, kStateSize, 1> state_der2;//K2
  Eigen::Matrix<ScalarType, kStateSize, 1> state_der3;//K3
  Eigen::Matrix<ScalarType, kStateSize, 1> state_der4;//K4

  //分别计算k1-4
  //这里dy是状态，dx是t
  getStateDerivativeRungeKutta(imu_readings_k1, current_state, &state_der1);

  getStateDerivativeRungeKutta(
      imu_readings_k23,
      static_cast<const Eigen::Matrix<ScalarType, kStateSize, 1> >(
          current_state + o5 * delta_time_seconds * state_der1),
      &state_der2);
  getStateDerivativeRungeKutta(
      imu_readings_k23,
      static_cast<const Eigen::Matrix<ScalarType, kStateSize, 1> >(
          current_state + o5 * delta_time_seconds * state_der2),
      &state_der3);
  getStateDerivativeRungeKutta(
      imu_readings_k4,
      static_cast<const Eigen::Matrix<ScalarType, kStateSize, 1> >(
          current_state + delta_time_seconds * state_der3),
      &state_der4);

  // Calculate final state using RK4.
  *next_state = current_state +//利用RK4数值积分求出k+1时刻的p v q bias
                delta_time_seconds * (state_der1 + ScalarType(2) * state_der2 +
                                      ScalarType(2) * state_der3 + state_der4) /
                    ScalarType(6);

  if (calculate_phi_cov)
  {
    next_phi->setZero();
    next_cov->setZero();

    const ScalarType* state_q_ptr = next_state->head(4).data();
    Eigen::Quaternion<ScalarType> B_q_G(state_q_ptr);//q_I_M
    B_q_G.normalize();//仍然要标准化

    // Now calculate state transition matrix and covariance.
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize> cov_der1;
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize> cov_der2;
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize> cov_der3;
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize> cov_der4;

    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize> transition_der1;
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize> transition_der2;
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize> transition_der3;
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize> transition_der4;

    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize> current_cov =
        Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize>::Zero();
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize>
        current_transition = Eigen::Matrix<ScalarType, kErrorStateSize,
                                           kErrorStateSize>::Identity();


    //这里利用LK4求状态的cov和F矩阵

    getCovarianceTransitionDerivativesRungeKutta(
        imu_readings_k1, current_state, current_cov, current_transition,
        &cov_der1, &transition_der1);

    Eigen::Matrix<ScalarType, kStateSize, 1> current_state_intermediate;
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize>
        current_cov_intermediate;
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize>
        current_transition_intermediate;

    ScalarType o5 = static_cast<ScalarType>(0.5);
    current_state_intermediate =
        current_state + o5 * delta_time_seconds * state_der1;
    current_cov_intermediate = current_cov + o5 * delta_time_seconds * cov_der1;
    current_transition_intermediate =
        current_transition + o5 * delta_time_seconds * transition_der1;
    getCovarianceTransitionDerivativesRungeKutta(
        imu_readings_k23, current_state_intermediate, current_cov_intermediate,
        current_transition_intermediate, &cov_der2, &transition_der2);

    current_state_intermediate =//y+0.5h
        current_state + o5 * delta_time_seconds * state_der2;
    current_cov_intermediate = current_cov + o5 * delta_time_seconds * cov_der2;
    current_transition_intermediate =
        current_transition + o5 * delta_time_seconds * transition_der2;
    getCovarianceTransitionDerivativesRungeKutta(
        imu_readings_k23, current_state_intermediate, current_cov_intermediate,
        current_transition_intermediate, &cov_der3, &transition_der3);

    current_state_intermediate =
        current_state + delta_time_seconds * state_der3;
    current_cov_intermediate = current_cov + delta_time_seconds * cov_der3;
    current_transition_intermediate =
        current_transition + delta_time_seconds * transition_der3;
    getCovarianceTransitionDerivativesRungeKutta(
        imu_readings_k4, current_state_intermediate, current_cov_intermediate,
        current_transition_intermediate, &cov_der4, &transition_der4);


    //由连续的状态方程利用rk4数值积分方法转成离散的形式

    *next_cov = current_cov +//k+1状态噪声的不确定度
                delta_time_seconds *
                    (cov_der1 + static_cast<ScalarType>(2) * cov_der2 +
                     static_cast<ScalarType>(2) * cov_der3 + cov_der4) /
                    static_cast<ScalarType>(6);
    *next_phi = current_transition;//k+1时刻的F

    next_phi->template block<3, 15>(0, 0) +=
        delta_time_seconds *
        (transition_der1.template block<3, 15>(0, 0) +
         ScalarType(2) * transition_der2.template block<3, 15>(0, 0) +
         ScalarType(2) * transition_der3.template block<3, 15>(0, 0) +
         transition_der4.template block<3, 15>(0, 0)) /
        ScalarType(6.0);
    next_phi->template block<3, 15>(6, 0) +=
        delta_time_seconds *
        (transition_der1.template block<3, 15>(6, 0) +
         ScalarType(2) * transition_der2.template block<3, 15>(6, 0) +
         ScalarType(2) * transition_der3.template block<3, 15>(6, 0) +
         transition_der4.template block<3, 15>(6, 0)) /
        ScalarType(6.0);
    next_phi->template block<3, 15>(12, 0) +=
        delta_time_seconds *
        (transition_der1.template block<3, 15>(12, 0) +
         ScalarType(2) * transition_der2.template block<3, 15>(12, 0) +
         ScalarType(2) * transition_der3.template block<3, 15>(12, 0) +
         transition_der4.template block<3, 15>(12, 0)) /
        ScalarType(6.0);
  }
}


//输入的是k+a*dt时刻的imu插值得到的测量值
template <typename ScalarType>
void ImuIntegratorRK4::getStateDerivativeRungeKutta(
    const Eigen::Matrix<ScalarType, kImuReadingSize, 1>& debiased_imu_readings,
    const Eigen::Matrix<ScalarType, kStateSize, 1>& current_state,//k+a*dt时刻y的数值近似值
    Eigen::Matrix<ScalarType, kStateSize, 1>* state_derivative) const {//state_derivative是k+a*dt时刻y的变化率
  CHECK_NOTNULL(state_derivative);

  Eigen::Quaternion<ScalarType> B_q_G(current_state.head(4).data());//q_I_M （JPL格式）
  // As B_q_G is calculated using linearization, it may not be normalized
  // -> we need to do it explicitly before passing to quaternion object.
  ScalarType o5 = static_cast<ScalarType>(0.5);
  B_q_G.normalize();//先进行四元数的归一化
  Eigen::Matrix<ScalarType, 3, 3> B_R_G;//R_I_M
  common::toRotationMatrixJPL(B_q_G.coeffs(), &B_R_G);
  const Eigen::Matrix<ScalarType, 3, 3> G_R_B = B_R_G.transpose();//R_M_I

  const Eigen::Matrix<ScalarType, 3, 1> acc_meas(//k+a*dt时刻的加速度测量值
      debiased_imu_readings.template block<3, 1>(kAccelReadingOffset, 0)
          .data());
  const Eigen::Matrix<ScalarType, 3, 1> gyr_meas(//k+a*dt时刻的陀螺仪测量值
      debiased_imu_readings.template block<3, 1>(kGyroReadingOffset, 0).data());

  Eigen::Matrix<ScalarType, 4, 4> gyro_omega;
  gyroOmegaJPL(gyr_meas, &gyro_omega);//计算陀螺仪测量值的虚四元数对应的左乘算子

  //这里的导数是用的JPL格式的状态方程推出来的，具体可以看一下s-msckf的论文
  Eigen::Matrix<ScalarType, 4, 1> q_dot = o5 * gyro_omega * B_q_G.coeffs();//q导数
  Eigen::Matrix<ScalarType, 3, 1> v_dot =//v导数
      G_R_B * acc_meas -
      Eigen::Matrix<ScalarType, 3, 1>(
          ScalarType(0), ScalarType(0), ScalarType(gravity_acceleration_));
  Eigen::Matrix<ScalarType, 3, 1> p_dot =
      current_state.template block<3, 1>(kStateVelocityOffset, 0);

  state_derivative->setZero();  // Bias derivatives are zero.
  state_derivative->template block<4, 1>(kStateOrientationOffset, 0) = q_dot;
  state_derivative->template block<3, 1>(kStateVelocityOffset, 0) = v_dot;
  state_derivative->template block<3, 1>(kStatePositionOffset, 0) = p_dot;
}

template <typename ScalarType>
void ImuIntegratorRK4::getCovarianceTransitionDerivativesRungeKutta(
    const Eigen::Matrix<ScalarType, kImuReadingSize, 1>& debiased_imu_readings,//k+a*dt时刻的imu插值得到的测量值
    const Eigen::Matrix<ScalarType, kStateSize, 1>& current_state,//k+a*dt时刻y的数值近似值
    const Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize>&
        current_cov,//k+a*dt时刻y的协方差
    const Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize>&
        current_transition,//k+a*dt时刻y的状态转移矩阵
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize>* cov_derivative,
    Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize>*
        transition_derivative) const {
  CHECK_NOTNULL(cov_derivative);
  CHECK_NOTNULL(transition_derivative);

  Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize> phi_cont =
      Eigen::Matrix<ScalarType, kErrorStateSize, kErrorStateSize>::Zero();

  Eigen::Quaternion<ScalarType> B_q_G(current_state.head(4).data());
  // As B_q_G is calculated using linearization, it may not be normalized
  // -> we need to do it explicitly before passing to quaternion object.
  B_q_G.normalize();//还是先标准 q_I_M
  Eigen::Matrix<ScalarType, 3, 3> B_R_G;
  common::toRotationMatrixJPL(B_q_G.coeffs(), &B_R_G);
  const Eigen::Matrix<ScalarType, 3, 3> G_R_B = B_R_G.transpose();

  //加速度和陀螺仪测量值
  Eigen::Matrix<ScalarType, 3, 1> acc_meas(
      debiased_imu_readings.template block<3, 1>(kAccelReadingOffset, 0)
          .data());
  const Eigen::Matrix<ScalarType, 3, 1> gyr_meas(
      debiased_imu_readings.template block<3, 1>(kGyroReadingOffset, 0).data());

  //他们的反对称矩阵
  const Eigen::Matrix<ScalarType, 3, 3> gyro_skew;
  common::skew(gyr_meas, gyro_skew);

  const Eigen::Matrix<ScalarType, 3, 3> acc_skew;
  common::skew(acc_meas, acc_skew);


  //xIdot =F ̃xI+Gn,phi_cont是F
  phi_cont.template block<3, 3>(0, 3) =
      -Eigen::Matrix<ScalarType, 3, 3>::Identity();
  phi_cont.template block<3, 3>(12, 6) =
      Eigen::Matrix<ScalarType, 3, 3>::Identity();
  phi_cont.template block<3, 3>(0, 0) = -gyro_skew;
  phi_cont.template block<3, 3>(6, 9) = -G_R_B;
  phi_cont.template block<3, 3>(6, 0) = -G_R_B * acc_skew;

  // Compute *transition_derivative = phi_cont * current_transition blockwise.
  //这里直接对行进行赋值，直接矩阵乘也可以
  transition_derivative->setZero();
  transition_derivative->template block<3, 15>(0, 0) =
      phi_cont.template block<3, 3>(0, 0) *
          current_transition.template block<3, 15>(0, 0) -
      current_transition.template block<3, 15>(3, 0);
  transition_derivative->template block<3, 15>(6, 0) =
      phi_cont.template block<3, 3>(6, 0) *
          current_transition.template block<3, 15>(0, 0) +
      phi_cont.template block<3, 3>(6, 9) *
          current_transition.template block<3, 15>(9, 0);
  transition_derivative->template block<3, 15>(12, 0) =
      current_transition.template block<3, 15>(6, 0);

  Eigen::Matrix<ScalarType, 15, 15> phi_cont_cov =
      Eigen::Matrix<ScalarType, 15, 15>::Zero();
  phi_cont_cov.template block<3, 15>(0, 0) =
      phi_cont.template block<3, 3>(0, 0) *
          current_cov.template block<3, 15>(0, 0) -
      current_cov.template block<3, 15>(3, 0);
  phi_cont_cov.template block<3, 15>(6, 0) =
      phi_cont.template block<3, 3>(6, 0) *
          current_cov.template block<3, 15>(0, 0) +
      phi_cont.template block<3, 3>(6, 9) *
          current_cov.template block<3, 15>(9, 0);
  phi_cont_cov.template block<3, 15>(12, 0) =
      current_cov.template block<3, 15>(6, 0);
  *cov_derivative = phi_cont_cov + phi_cont_cov.transpose();

  // Relevant parts of Gc * Qc * Gc'.
  cov_derivative->diagonal().template segment<3>(0) +=
      Eigen::Matrix<ScalarType, 3, 1>::Constant(
          static_cast<ScalarType>(gyro_noise_sigma_squared_));
  cov_derivative->diagonal().template segment<3>(3) +=
      Eigen::Matrix<ScalarType, 3, 1>::Constant(
          static_cast<ScalarType>(gyro_bias_sigma_squared_));
  cov_derivative->diagonal().template segment<3>(6) +=
      Eigen::Matrix<ScalarType, 3, 1>::Constant(
          static_cast<ScalarType>(acc_noise_sigma_squared_));
  cov_derivative->diagonal().template segment<3>(9) +=
      Eigen::Matrix<ScalarType, 3, 1>::Constant(
          static_cast<ScalarType>(acc_bias_sigma_squared_));
}

template <typename ScalarType>
void ImuIntegratorRK4::interpolateImuReadings(
    const Eigen::Matrix<ScalarType, 2 * kImuReadingSize, 1>& imu_readings,
    const ScalarType delta_time_seconds,
    const ScalarType increment_step_size_seconds,
    Eigen::Matrix<ScalarType, kImuReadingSize, 1>* interpolated_imu_readings)
    const {
  CHECK_NOTNULL(interpolated_imu_readings);
  CHECK_GE(delta_time_seconds, 0.0);




  if (delta_time_seconds < std::numeric_limits<ScalarType>::epsilon()) {//如果dt小于一个很小的数
    *interpolated_imu_readings =//不用插值，直接选用k时刻的imu数据
        imu_readings.template block<kImuReadingSize, 1>(0, 0);
    return;
  }

  *interpolated_imu_readings =//对imu进行线性插值
          // imuk           imu插值            imuk+1
          // |--------------|-----------------|
          //这里第一段虚线的间隔时间就是increment_step_size_seconds
      imu_readings.template block<kImuReadingSize, 1>(0, 0) +
      (imu_readings.template block<kImuReadingSize, 1>(kImuReadingSize, 0) -
       imu_readings.template block<kImuReadingSize, 1>(0, 0)) *
          (increment_step_size_seconds / delta_time_seconds);
}

template <typename ScalarType>
void ImuIntegratorRK4::gyroOmegaJPL(
    const Eigen::Matrix<ScalarType, 3, 1>& gyro_readings,
    Eigen::Matrix<ScalarType, 4, 4>* omega_matrix) const {
  CHECK_NOTNULL(omega_matrix);//q.w（）=0的一个虚部为gyro_readings的虚四元数左乘算子

  const ScalarType scalar_type_zero = static_cast<ScalarType>(0.);

  *omega_matrix << scalar_type_zero, gyro_readings[2], -gyro_readings[1],
      gyro_readings[0], -gyro_readings[2], scalar_type_zero, gyro_readings[0],
      gyro_readings[1], gyro_readings[1], -gyro_readings[0], scalar_type_zero,
      gyro_readings[2], -gyro_readings[0], -gyro_readings[1], -gyro_readings[2],
      scalar_type_zero;
}

}  // namespace imu_integrator

#endif  // IMU_INTEGRATOR_IMU_INTEGRATOR_INL_H_
