#include <memory>
#include <utility>

#include <aslam/cameras/camera-pinhole.h>

#include <aslam/cameras/camera-factory.h>
#include <aslam/common/types.h>

namespace aslam {
std::ostream& operator<<(std::ostream& out, const PinholeCamera& camera) {
  camera.printParameters(out, std::string(""));
  return out;
}

PinholeCamera::PinholeCamera()
    : Base(Eigen::Vector4d::Zero(), 0, 0, Camera::Type::kPinhole) {}

PinholeCamera::PinholeCamera(const Eigen::VectorXd& intrinsics,
                             uint32_t image_width, uint32_t image_height,
                             aslam::Distortion::UniquePtr& distortion)
  : Base(intrinsics, distortion, image_width, image_height, Camera::Type::kPinhole) {
  CHECK(intrinsicsValid(intrinsics));
}

PinholeCamera::PinholeCamera(const Eigen::VectorXd& intrinsics, uint32_t image_width,
                             uint32_t image_height)
    : Base(intrinsics, image_width, image_height, Camera::Type::kPinhole) {
  CHECK(intrinsicsValid(intrinsics));
}

PinholeCamera::PinholeCamera(double focallength_cols, double focallength_rows,
                             double imagecenter_cols, double imagecenter_rows, uint32_t image_width,
                             uint32_t image_height, aslam::Distortion::UniquePtr& distortion)
    : PinholeCamera(
        Eigen::Vector4d(focallength_cols, focallength_rows, imagecenter_cols, imagecenter_rows),
        image_width, image_height, distortion) {}

PinholeCamera::PinholeCamera(double focallength_cols, double focallength_rows,
                             double imagecenter_cols, double imagecenter_rows, uint32_t image_width,
                             uint32_t image_height)
    : PinholeCamera(
        Eigen::Vector4d(focallength_cols, focallength_rows, imagecenter_cols, imagecenter_rows),
        image_width, image_height) {}

bool PinholeCamera::operator==(const Camera& other) const {
  // Check that the camera models are the same.
  const PinholeCamera* rhs = dynamic_cast<const PinholeCamera*>(&other);
  if (!rhs)
    return false;

  // Verify that the base members are equal.
  if (!Camera::operator==(other))
    return false;

  // Compare the distortion model (if distortion is set for both).
  if ( !(*(this->distortion_) == *(rhs->distortion_)) )
    return false;

  return true;
}

bool PinholeCamera::backProject3(const Eigen::Ref<const Eigen::Vector2d>& keypoint,
                                 Eigen::Vector3d* out_point_3d) const {
  CHECK_NOTNULL(out_point_3d);

  Eigen::Vector2d kp = keypoint;//输入的是像素坐标
  //转换成归一化坐标
  kp[0] = (kp[0] - cu()) / fu();
  kp[1] = (kp[1] - cv()) / fv();
  //去畸变
  distortion_->undistort(&kp);
  //得到去畸变后的归一化坐标
  (*out_point_3d)[0] = kp[0];
  (*out_point_3d)[1] =kp [1];
  (*out_point_3d)[2] = 1;

  // Always valid for the pinhole model.
  return true;
}
//输入的是这个地图点在当前观测到这个点的相机中的坐标，内参，畸变，
    // 重投影以后这个地图点的像素坐标，残差对于fci的雅克比，残差对于内参的雅克比，残差畸变的雅克比
    //还会根据重投影的像素坐标和3d点坐标评判重投影结果
const ProjectionResult PinholeCamera::project3Functional(
    const Eigen::Ref<const Eigen::Vector3d>& point_3d,
    const Eigen::VectorXd* intrinsics_external,
    const Eigen::VectorXd* distortion_coefficients_external,
    Eigen::Vector2d* out_keypoint,
    Eigen::Matrix<double, 2, 3>* out_jacobian_point,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian_intrinsics,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian_distortion) const
{
    CHECK_NOTNULL(out_keypoint);

    // Determine the parameter source. (if nullptr, use internal)
    const Eigen::VectorXd* intrinsics;
    if (!intrinsics_external)
        intrinsics = &getParameters();
    else
        intrinsics = intrinsics_external;//如果有内参就直接赋值
    CHECK_EQ(intrinsics->size(), kNumOfParams) << "intrinsics: invalid size!";

    const Eigen::VectorXd* distortion_coefficients;
    if(!distortion_coefficients_external) {
        distortion_coefficients = &getDistortion().getParameters();
    } else {
        distortion_coefficients = distortion_coefficients_external;//如果有畸变参数就直接赋值
    }

    const double& fu = (*intrinsics)[0];
    const double& fv = (*intrinsics)[1];
    const double& cu = (*intrinsics)[2];
    const double& cv = (*intrinsics)[3];

    // Project the point.
    const double& x = point_3d[0];
    const double& y = point_3d[1];
    const double& z = point_3d[2];

    const double rz = 1.0 / z;//逆深度
    (*out_keypoint)[0] = x * rz;//这里存储的是归一化坐标
    (*out_keypoint)[1] = y * rz;

    // Distort the point and get the Jacobian wrt. keypoint.
    Eigen::Matrix2d J_distortion = Eigen::Matrix2d::Identity();
    if(out_jacobian_distortion)
    {
        // Calculate the Jacobian w.r.t to the distortion parameters,
        // if requested (and distortion set).
        //我只针对radtan这部分进行注释
        //残差是重投影的像素坐标-观测到的像素坐标
        //应用链式法则 残差对畸变的雅克比 = 残差对归一化坐标的雅克比 × 归一化坐标对于畸变的雅克比（因为畸变是对归一化坐标做的处理）
        distortion_->distortParameterJacobian(distortion_coefficients,//畸变参数
                                              *out_keypoint,//归一化坐标
                                              out_jacobian_distortion);//归一化坐标对于畸变的雅克比
        //残差对于归一化坐标的雅克比是[fu,0;0,fv]，乘上归一化坐标对于畸变的雅克比
        out_jacobian_distortion->row(0) *= fu;
        out_jacobian_distortion->row(1) *= fv;
    }

    if(out_jacobian_point)
    {
        // Distortion active and we want the Jacobian.
        //J_distortion是去畸变的归一化坐标对于带畸变的归一化的雅克比
        distortion_->distortUsingExternalCoefficients(distortion_coefficients,
                                                      out_keypoint,
                                                      &J_distortion);
    } else {
        // Distortion active but Jacobian NOT wanted.
        distortion_->distortUsingExternalCoefficients(distortion_coefficients,
                                                      out_keypoint,
                                                      nullptr);
    }

    if(out_jacobian_point)
    {
        // Jacobian including distortion
        const double rz2 = rz * rz;

         //链式法则， 残差对点带畸变的归一化坐标的雅克比 = 残差对xyz的雅克比× xyz 对于畸变的雅克比
         //u重 = fx * (X/Z) + cx
         //v重 = fy * (Y/Z) + cy
         //Ju_X = fx / Z
         //Ju_Y = 0
         //Ju_Z = - fx / Z2
         //Jv_X = 0
         //Jv_Y = fy / Z
         //Jv_Z = - fy / Z2



        const double duf_dx =  fu * J_distortion(0, 0) * rz;
        const double duf_dy =  fu * J_distortion(0, 1) * rz;
        //为什么（0,2）的雅克比是这个？ x * J_distortion(0, 0) + y * J_distortion(0, 1)
        //2*2 怎么扩展成3×3的
        const double duf_dz = -fu * (x * J_distortion(0, 0) + y * J_distortion(0, 1)) * rz2;
        const double dvf_dx =  fv * J_distortion(1, 0) * rz;
        const double dvf_dy =  fv * J_distortion(1, 1) * rz;
        const double dvf_dz = -fv * (x * J_distortion(1, 0) + y * J_distortion(1, 1)) * rz2;

        (*out_jacobian_point) << duf_dx, duf_dy, duf_dz,
                dvf_dx, dvf_dy, dvf_dz;
    }

    // Calculate the Jacobian w.r.t to the intrinsic parameters, if requested.
    if(out_jacobian_intrinsics) {
        //这里计算的是残差对于内参4个参数的雅克比
        out_jacobian_intrinsics->resize(2, kNumOfParams);
        const double duf_dfu = (*out_keypoint)[0];
        const double duf_dfv = 0.0;
        const double duf_dcu = 1.0;
        const double duf_dcv = 0.0;
        const double dvf_dfu = 0.0;
        const double dvf_dfv = (*out_keypoint)[1];
        const double dvf_dcu = 0.0;
        const double dvf_dcv = 1.0;

        (*out_jacobian_intrinsics) << duf_dfu, duf_dfv, duf_dcu, duf_dcv,
                dvf_dfu, dvf_dfv, dvf_dcu, dvf_dcv;
    }

    // Normalized image plane to camera plane.
    (*out_keypoint)[0] = fu * (*out_keypoint)[0] + cu;//由归一化坐标转化成像素坐标
    (*out_keypoint)[1] = fv * (*out_keypoint)[1] + cv;

    return evaluateProjectionResult(*out_keypoint, point_3d);
}

Eigen::Vector2d PinholeCamera::createRandomKeypoint() const {
  Eigen::Vector2d out;
  out.setRandom();
  // Unit tests often fail when the point is near the border. Keep the point
  // away from the border.
  double border = std::min(imageWidth(), imageHeight()) * 0.1;

  out(0) = border + std::abs(out(0)) * (imageWidth() - border * 2.0);
  out(1) = border + std::abs(out(1)) * (imageHeight() - border * 2.0);

  return out;
}

Eigen::Vector3d PinholeCamera::createRandomVisiblePoint(double depth) const {
  CHECK_GT(depth, 0.0) << "Depth needs to be positive!";
  Eigen::Vector3d point_3d;

  Eigen::Vector2d y = createRandomKeypoint();
  backProject3(y, &point_3d);
  point_3d /= point_3d.norm();

  // Muck with the depth. This doesn't change the pointing direction.
  return point_3d * depth;
}

void PinholeCamera::getBorderRays(Eigen::MatrixXd& rays) const {
  rays.resize(4, 8);
  Eigen::Vector4d ray;
  backProject4(Eigen::Vector2d(0.0, 0.0), &ray);
  rays.col(0) = ray;
  backProject4(Eigen::Vector2d(0.0, imageHeight() * 0.5), &ray);
  rays.col(1) = ray;
  backProject4(Eigen::Vector2d(0.0, imageHeight() - 1.0), &ray);
  rays.col(2) = ray;
  backProject4(Eigen::Vector2d(imageWidth() - 1.0, 0.0), &ray);
  rays.col(3) = ray;
  backProject4(Eigen::Vector2d(imageWidth() - 1.0, imageHeight() * 0.5), &ray);
  rays.col(4) = ray;
  backProject4(Eigen::Vector2d(imageWidth() - 1.0, imageHeight() - 1.0), &ray);
  rays.col(5) = ray;
  backProject4(Eigen::Vector2d(imageWidth() * 0.5, 0.0), &ray);
  rays.col(6) = ray;
  backProject4(Eigen::Vector2d(imageWidth() * 0.5, imageHeight() - 1.0), &ray);
  rays.col(7) = ray;
}

bool PinholeCamera::areParametersValid(const Eigen::VectorXd& parameters) {
  return (parameters.size() == parameterCount()) &&
         (parameters[0] > 0.0)  && //fu
         (parameters[1] > 0.0)  && //fv
         (parameters[2] > 0.0)  && //cu
         (parameters[3] > 0.0);    //cv
}

bool PinholeCamera::intrinsicsValid(const Eigen::VectorXd& intrinsics) {
  return areParametersValid(intrinsics);
}

void PinholeCamera::printParameters(std::ostream& out, const std::string& text) const {
  Camera::printParameters(out, text);
  out << "  focal length (cols,rows): "
      << fu() << ", " << fv() << std::endl;
  out << "  optical center (cols,rows): "
      << cu() << ", " << cv() << std::endl;

  out << "  distortion: ";
  distortion_->printParameters(out, text);
}
const double PinholeCamera::kMinimumDepth = 1e-10;
}  // namespace aslam
