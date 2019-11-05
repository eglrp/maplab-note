#ifndef CERES_ERROR_TERMS_VISUAL_ERROR_TERM_FACTORY_INL_H_
#define CERES_ERROR_TERMS_VISUAL_ERROR_TERM_FACTORY_INL_H_

#include <limits>
#include <vector>

#include <glog/logging.h>

#include <aslam/cameras/camera-pinhole.h>
#include <aslam/cameras/camera-unified-projection.h>
#include <aslam/cameras/camera.h>
#include <aslam/cameras/distortion-equidistant.h>
#include <aslam/cameras/distortion-fisheye.h>
#include <aslam/cameras/distortion-null.h>
#include <aslam/cameras/distortion-radtan.h>
#include <aslam/cameras/distortion.h>

#include "ceres-error-terms/common.h"

namespace ceres_error_terms {

    //构建costfunction,ErrorTerm是ceres_error_terms::VisualReprojectionError
template <template <typename, typename> class ErrorTerm>
ceres::CostFunction* createVisualCostFunction(
    const Eigen::Vector2d& measurement, double pixel_sigma,
    ceres_error_terms::visual::VisualErrorType error_term_type,
    aslam::Camera* camera)
{
    CHECK_NOTNULL(camera);
    ceres::CostFunction* error_term = nullptr;
    switch (camera->getType())
    {//针对不同的投影模型和畸变模型选择不同的ErrorTerm
        case aslam::Camera::Type::kPinhole://针孔相机模型
        {
            aslam::PinholeCamera* derived_camera =
                    static_cast<aslam::PinholeCamera*>(camera);
            aslam::Distortion::Type distortion_type =
                    camera->getDistortion().getType();
            switch (distortion_type)
            {
                case aslam::Distortion::Type::kNoDistortion:
                    error_term =
                            new ErrorTerm<aslam::PinholeCamera, aslam::NullDistortion>(
                                    measurement, pixel_sigma, error_term_type, derived_camera);
                    break;
                case aslam::Distortion::Type::kEquidistant:
                    error_term =
                            new ErrorTerm<aslam::PinholeCamera, aslam::EquidistantDistortion>(
                                    measurement, pixel_sigma, error_term_type, derived_camera);
                    break;
                case aslam::Distortion::Type::kRadTan:
                    error_term =
                            new ErrorTerm<aslam::PinholeCamera, aslam::RadTanDistortion>(
                                    measurement, pixel_sigma, error_term_type, derived_camera);
                    break;
                case aslam::Distortion::Type::kFisheye:
                    error_term =
                            new ErrorTerm<aslam::PinholeCamera, aslam::FisheyeDistortion>(
                                    measurement, pixel_sigma, error_term_type, derived_camera);
                    break;
                default:
                    LOG(FATAL) << "Invalid camera distortion type for ceres error term: "
                               << static_cast<int>(distortion_type);
            }
            break;
        }
        case aslam::Camera::Type::kUnifiedProjection://
        {
            aslam::UnifiedProjectionCamera* derived_camera =
                    static_cast<aslam::UnifiedProjectionCamera*>(camera);
            aslam::Distortion::Type distortion_type =
                    camera->getDistortion().getType();
            switch (distortion_type)
            {
                case aslam::Distortion::Type::kNoDistortion:
                    error_term = new ErrorTerm<aslam::UnifiedProjectionCamera,
                            aslam::NullDistortion>(
                            measurement, pixel_sigma, error_term_type, derived_camera);
                    break;
                case aslam::Distortion::Type::kEquidistant:
                    error_term = new ErrorTerm<aslam::UnifiedProjectionCamera,
                            aslam::EquidistantDistortion>(
                            measurement, pixel_sigma, error_term_type, derived_camera);
                    break;
                case aslam::Distortion::Type::kRadTan:
                    error_term = new ErrorTerm<aslam::UnifiedProjectionCamera,
                            aslam::RadTanDistortion>(
                            measurement, pixel_sigma, error_term_type, derived_camera);
                    break;
                case aslam::Distortion::Type::kFisheye:
                    error_term = new ErrorTerm<aslam::UnifiedProjectionCamera,
                            aslam::FisheyeDistortion>(
                            measurement, pixel_sigma, error_term_type, derived_camera);
                    break;
                default:
                    LOG(FATAL) << "Invalid camera distortion type for ceres error term: "
                               << static_cast<int>(distortion_type);
            }
            break;
        }
        default:
            LOG(FATAL) << "Invalid camera projection type for ceres error term: "
                       << static_cast<int>(camera->getType());
    }
    return error_term;
}

//三种情况需要不同的控制优化策略
void replaceUnusedArgumentsOfVisualCostFunctionWithDummies(
    ceres_error_terms::visual::VisualErrorType error_term_type,
    std::vector<double*>* error_term_argument_list,
    std::vector<double*>* dummies_to_set_constant)
{
    CHECK_NOTNULL(error_term_argument_list);
    CHECK_NOTNULL(dummies_to_set_constant)->clear();

    CHECK_EQ(error_term_argument_list->size(), 9u);
    for (const double* argument : *error_term_argument_list) {
        CHECK_NOTNULL(argument);
    }

    // Initialize dummy variables to infinity so that any usage mistakes can be
    // detected.将虚拟变量初始化为无穷大，以便发现任何使用错误
    static Eigen::Matrix<double, 7, 1> dummy_7d_landmark_base_pose =
            Eigen::Matrix<double, 7, 1>::Constant(std::numeric_limits<double>::max());
    static Eigen::Matrix<double, 7, 1> dummy_7d_landmark_mission_base_pose =
            Eigen::Matrix<double, 7, 1>::Constant(std::numeric_limits<double>::max());
    static Eigen::Matrix<double, 7, 1> dummy_7d_imu_mission_base_pose =
            Eigen::Matrix<double, 7, 1>::Constant(std::numeric_limits<double>::max());
    static Eigen::Matrix<double, 7, 1> dummy_7d_imu_pose =
            Eigen::Matrix<double, 7, 1>::Constant(std::numeric_limits<double>::max());

    if (error_term_type == visual::VisualErrorType::kLocalKeyframe) //当前节点就是第一次观测到这个地图点的节点
    {//在局部任务情况下，不需要基帧和关键帧姿态，只需要地图点在这个节点body系中的坐标，相机外参，相机内参，相机畸变
        // The baseframes and keyframe poses are not necessary in the local
        // mission case.
        (*error_term_argument_list)[1] = dummy_7d_landmark_base_pose.data();
        (*error_term_argument_list)[2] = dummy_7d_landmark_mission_base_pose.data();
        (*error_term_argument_list)[3] = dummy_7d_imu_mission_base_pose.data();
        (*error_term_argument_list)[4] = dummy_7d_imu_pose.data();
        dummies_to_set_constant->push_back(dummy_7d_landmark_base_pose.data());
        dummies_to_set_constant->push_back(
                dummy_7d_landmark_mission_base_pose.data());
        dummies_to_set_constant->push_back(dummy_7d_imu_mission_base_pose.data());
        dummies_to_set_constant->push_back(dummy_7d_imu_pose.data());
    } else if (error_term_type == visual::VisualErrorType::kLocalMission) //当前节点和第一次观测到这个地图点的节点在同一个任务下
    {//都在一个任务下，所以也不需要基帧
        // The baseframes are not necessary in the local mission case.
        (*error_term_argument_list)[2] = dummy_7d_landmark_base_pose.data();
        (*error_term_argument_list)[3] = dummy_7d_landmark_mission_base_pose.data();
        dummies_to_set_constant->push_back(dummy_7d_landmark_base_pose.data());
        dummies_to_set_constant->push_back(
                dummy_7d_landmark_mission_base_pose.data());
    } else if (error_term_type == visual::VisualErrorType::kGlobal)
    {
        // Nothing to replace.
    } else {
        LOG(FATAL) << "Unknown error term type: " << error_term_type;
    }
}

}  // namespace ceres_error_terms

#endif  // CERES_ERROR_TERMS_VISUAL_ERROR_TERM_FACTORY_INL_H_
