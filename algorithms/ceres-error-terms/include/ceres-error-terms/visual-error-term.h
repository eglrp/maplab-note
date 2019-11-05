#ifndef CERES_ERROR_TERMS_VISUAL_ERROR_TERM_H_
#define CERES_ERROR_TERMS_VISUAL_ERROR_TERM_H_

#include <memory>

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/sized_cost_function.h>
#include <maplab-common/pose_types.h>

#include "ceres-error-terms/common.h"
#include "ceres-error-terms/parameterization/quaternion-param-jpl.h"
#include "ceres-error-terms/visual-error-term-base.h"

namespace ceres_error_terms {
// Note: this error term accepts rotations expressed as quaternions
// in JPL convention [x, y, z, w]. This convention corresponds to the internal
// coefficient storage of Eigen so you can directly pass pointer to your
// Eigen quaternion data, e.g. your_eigen_quaternion.coeffs().data().
//注意:这个错误项接受在JPL约定中表示为四元数的旋转[x, y, z, w]。这个约定对应于特征值的内部系数存储
// ，因此您可以直接将指针传递给特征值四元数数据，例如您的_eigen_quaternion.coeffs().data()。
//不同的投影模型和畸变模型
/* 雅克比矩阵的对应
    enum {
        kIdxLandmarkP,3维
        kIdxLandmarkBasePose,7维
        kIdxLandmarkMissionBasePose,7维
        kIdxImuMissionBasePose,7维
        kIdxImuPose,7维
        kIdxCameraToImuQ,4维
        kIdxCameraToImuP,3维
        kIdxCameraIntrinsics,4维
        kIdxCameraDistortion,4 维
    };
    */
template <typename CameraType, typename DistortionType>
class VisualReprojectionError
    : public ceres::SizedCostFunction<//针孔，radtan而言<2,3,7,7,7,7,4,3,4,4>
          visual::kResidualSize, visual::kPositionBlockSize,
          visual::kPoseBlockSize, visual::kPoseBlockSize,
          visual::kPoseBlockSize, visual::kPoseBlockSize,
          visual::kOrientationBlockSize, visual::kPositionBlockSize,
          CameraType::parameterCount(), DistortionType::parameterCount()>,
          //CameraType比如针孔模型就是4（fu，fv，cx，cy）,DistortionType比如radtan是4（k1,k2,p1,p2）
      public VisualCostFunction {
 public:
  typedef VisualCostFunction Base;

  // Construct a cost function representing the reprojection error. Sigma is
  // standard deviation (in pixels).标准差
  VisualReprojectionError(
      const Eigen::Vector2d& measurement, double pixel_sigma,
      visual::VisualErrorType error_term_type, const CameraType* camera)
      : Base(pixel_sigma),
        measurement_(measurement),//观测值
        error_term_type_(error_term_type),//误差类型
        camera_ptr_(camera) {//相机类型
    CHECK(camera);
    CHECK(isValidVisualErrorTermType(error_term_type_));
  }

  virtual ~VisualReprojectionError() {}

  virtual bool Evaluate(
      double const* const* parameters, double* residuals,
      double** jacobians) const;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  // Don't change the ordering of the enum elements, they have to be the
  // same as the order of the parameter blocks.
  enum {
    kIdxLandmarkP,
    kIdxLandmarkBasePose,
    kIdxLandmarkMissionBasePose,
    kIdxImuMissionBasePose,
    kIdxImuPose,
    kIdxCameraToImuQ,
    kIdxCameraToImuP,
    kIdxCameraIntrinsics,
    kIdxCameraDistortion
  };

  // The representation for Jacobians computed by this object.
  typedef Eigen::Vector2d SwitchJacobian;

  typedef Eigen::Matrix<double, visual::kResidualSize,
                        visual::kOrientationBlockSize, Eigen::RowMajor>
      OrientationJacobian;

  typedef Eigen::Matrix<double, visual::kResidualSize,
                        visual::kPositionBlockSize, Eigen::RowMajor>
      PositionJacobian;

  typedef Eigen::Matrix<double, visual::kResidualSize, visual::kPoseBlockSize,
                        Eigen::RowMajor>
      PoseJacobian;

  typedef Eigen::Matrix<double, visual::kResidualSize,
                        CameraType::parameterCount(), Eigen::RowMajor>
      IntrinsicsJacobian;

  typedef Eigen::Matrix<double, visual::kResidualSize, Eigen::Dynamic,
                        Eigen::RowMajor>
      DistortionJacobian;

  Eigen::Vector2d measurement_;
  const visual::VisualErrorType error_term_type_;
  const CameraType* camera_ptr_;
};

}  // namespace ceres_error_terms

#include "ceres-error-terms/visual-error-term-inl.h"

#endif  // CERES_ERROR_TERMS_VISUAL_ERROR_TERM_H_
