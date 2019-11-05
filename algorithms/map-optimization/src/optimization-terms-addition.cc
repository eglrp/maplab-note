#include "map-optimization/optimization-terms-addition.h"

#include <memory>

#include <ceres-error-terms/inertial-error-term.h>
#include <ceres-error-terms/visual-error-term-factory.h>
#include <ceres-error-terms/visual-error-term.h>
#include <ceres/ceres.h>
#include <maplab-common/progress-bar.h>
#include <vi-map-helpers/vi-map-queries.h>
#include <vi-map/landmark-quality-metrics.h>

namespace map_optimization {

    //添加特征点(这个特征点是被某个节点（vertex_ptr）中的某一个相机（frame_idx）观测到)的视觉因子，残差等
bool addVisualTermForKeypoint(
    const int keypoint_idx, const int frame_idx,
    const bool fix_landmark_positions, const bool fix_intrinsics,
    const bool fix_extrinsics_rotation, const bool fix_extrinsics_translation,
    const std::shared_ptr<ceres::LocalParameterization>& pose_parameterization,
    const std::shared_ptr<ceres::LocalParameterization>&
        baseframe_parameterization,
    const std::shared_ptr<ceres::LocalParameterization>&
        camera_parameterization,
    vi_map::Vertex* vertex_ptr, OptimizationProblem* problem)
{
    CHECK_NOTNULL(vertex_ptr);
    CHECK_NOTNULL(problem);

    CHECK(pose_parameterization != nullptr);
    CHECK(baseframe_parameterization != nullptr);
    CHECK(camera_parameterization != nullptr);

    OptimizationStateBuffer* buffer =//得到储存节点的位姿，各个任务基准帧的位姿，各个任务的相机系统的外参标定的OptimizationStateBuffer
            CHECK_NOTNULL(problem->getOptimizationStateBufferMutable());
    vi_map::VIMap* map = CHECK_NOTNULL(problem->getMapMutable());
    ceres_error_terms::ProblemInformation* problem_information =//构造了一个优化问题
            CHECK_NOTNULL(problem->getProblemInformationMutable());

    const aslam::VisualFrame& visual_frame =//看到这个特征点的具体相机
            vertex_ptr->getVisualFrame(frame_idx);
    CHECK_GE(keypoint_idx, 0);//检查一个特征点id
    CHECK_LT(
            keypoint_idx,
            static_cast<int>(visual_frame.getNumKeypointMeasurements()));

    const vi_map::LandmarkId landmark_id =//得到特征点对应的地图点id
            vertex_ptr->getObservedLandmarkId(frame_idx, keypoint_idx);

    // The keypoint must have a valid association with a landmark.
    CHECK(landmark_id.isValid());//2d-3d必须要对应上

    vi_map::Vertex& landmark_store_vertex =//根据这个地图点id找到第一次观测到这个地图点的节点
            map->getLandmarkStoreVertex(landmark_id);
    vi_map::Landmark& landmark = map->getLandmark(landmark_id);//得到地图点

    const aslam::Camera::Ptr camera_ptr = vertex_ptr->getCamera(frame_idx);//得到这一个相机对应的相机模型
    CHECK(camera_ptr != nullptr);

    const Eigen::Vector2d& image_point_distorted =//得到带畸变的特征点像素坐标, 当前帧看到的
            vertex_ptr->getVisualFrame(frame_idx).getKeypointMeasurement(
                    keypoint_idx);
    const double image_point_uncertainty =//每个特征点的不确定度？,这个是怎么得到的，前端给的么？
            vertex_ptr->getVisualFrame(frame_idx).getKeypointMeasurementUncertainty(
                    keypoint_idx);

    // As defined here: http://en.wikipedia.org/wiki/Huber_Loss_Function
    double huber_loss_delta = 3.0;
    //enum VisualErrorType { kLocalKeyframe, kLocalMission, kGlobal };三种
    ceres_error_terms::visual::VisualErrorType error_term_type;
    if (vertex_ptr->id() != landmark_store_vertex.id()) //如果这个节点不是第一个观测到这个地图点的节点
    {
        // Verify if the landmark and keyframe belong to the same mission.
        if (vertex_ptr->getMissionId() == landmark_store_vertex.getMissionId())
        {//如果当前节点和首次观测到这个地图点的节点是同一个任务下的节点，视觉误差定义成kLocalMission
            error_term_type =
                    ceres_error_terms::visual::VisualErrorType::kLocalMission;
        } else
        {//如果当前节点和首次观测到这个地图点的节点不是同一个任务下的节点，视觉误差定义成kGlobal
            error_term_type = ceres_error_terms::visual::VisualErrorType::kGlobal;
            huber_loss_delta = 10.0;
        }
    } else
    {//如果这个节点是第一个观测到这个地图点的节点,就把视觉误差类型定成kLocalKeyframe，也就是说第一次观测到这个地图点的帧是关键帧
        error_term_type =
                ceres_error_terms::visual::VisualErrorType::kLocalKeyframe;
    }

    // 相机的畸变类型
    // enum class Type {
    //    kNoDistortion = 0,
    //    kEquidistant = 1,
    //    kFisheye = 2,
    //    kRadTan = 3
    //  };
    double* distortion_params = nullptr;
    if (camera_ptr->getDistortion().getType() != aslam::Distortion::Type::kNoDistortion)
    {
        distortion_params = camera_ptr->getDistortionMutable()->getParametersMutable();
        CHECK_NOTNULL(distortion_params);
    }

    vi_map::MissionBaseFrameId observer_baseframe_id = //当前这个节点所在任务的基准帧的id
            map->getMissionForVertex(vertex_ptr->id()).getBaseFrameId();
    vi_map::MissionBaseFrameId store_baseframe_id =//首次观测到这个特征点的节点所在任务的基准帧id
            map->getMissionForVertex(landmark_store_vertex.id()).getBaseFrameId();
    double* observer_baseframe_q_GM__G_p_GM =//当前这个节点所在任务的基准帧的pose(q是JPL)
            buffer->get_baseframe_q_GM__G_p_GM_JPL(observer_baseframe_id);
    double* landmark_store_baseframe_q_GM__G_p_GM =//首次观测到这个特征点的节点所在任务的基准帧的pose(q是JPL格式)
            buffer->get_baseframe_q_GM__G_p_GM_JPL(store_baseframe_id);

    double* vertex_q_IM__M_p_MI =//当前这个节点的本体pose(q是JPL)
            buffer->get_vertex_q_IM__M_p_MI_JPL(vertex_ptr->id());
    double* landmark_store_vertex_q_IM__M_p_MI =//首次观测到这个特征点的节点的本体pose(q是JPL)
            buffer->get_vertex_q_IM__M_p_MI_JPL(landmark_store_vertex.id());

    const aslam::CameraId& camera_id = camera_ptr->getId();
    CHECK(camera_id.isValid());
    double* camera_q_CI =//这个相机和body坐标系中的外参（q是JPL）
            buffer->get_camera_extrinsics_q_CI__C_p_CI_JPL(camera_id);
    // The visual error term requires the camera rotation and translation
    // to be feeded separately. Shifting by 4 = the quaternione size.
    double* camera_C_p_CI = camera_q_CI + 4;//这里+4是因为相机的p和q是分开优化的，所以camera_C_p_CI是p,需要跳过q存储的部分

    std::shared_ptr<ceres::CostFunction> visual_term_cost
            (//构建ceres的costfunction
                    ceres_error_terms::createVisualCostFunction<ceres_error_terms::VisualReprojectionError>
                            (
                            image_point_distorted, image_point_uncertainty, error_term_type,
                            camera_ptr.get()));

    std::vector<double*> cost_term_args =
            {
                    landmark.get_p_B_Mutable(),
                    landmark_store_vertex_q_IM__M_p_MI,
                    landmark_store_baseframe_q_GM__G_p_GM,
                    observer_baseframe_q_GM__G_p_GM,
                    vertex_q_IM__M_p_MI,
                    camera_q_CI,
                    camera_C_p_CI,
                    camera_ptr->getParametersMutable(),
                    camera_ptr->getDistortionMutable()->getParametersMutable()
            };

    // Certain types of visual cost terms (as indicated by error_term_type) do not
    // use all of the pointer arguments. Ceres, however, requires us to provide
    // valid pointers so we replace unnecessary arguments with dummy variables
    // filled with NaNs. The function also returns the pointers of the dummies
    // used so that we can set them constant below.
    //某些类型的可视成本项(如error_term_type所示)不使用所有指针参数。但是Ceres要求我们提供有效的指针，
    // 所以我们用填充了NaNs的伪变量替换了不必要的参数。该函数还返回所使用的伪指针，以便我们可以将它们设置为常量。
    std::vector<double*> dummies_to_set_constant;
    //error_term_type有enum VisualErrorType { kLocalKeyframe, kLocalMission, kGlobal };三种，
    // 需要针对不同的情况不优化某些量
    ceres_error_terms::replaceUnusedArgumentsOfVisualCostFunctionWithDummies(
            error_term_type, &cost_term_args, &dummies_to_set_constant);

    for (double* dummy : dummies_to_set_constant) {
        problem_information->setParameterBlockConstant(dummy);//设置不优化的量
    }

    std::shared_ptr<ceres::LossFunction> loss_function(//设置huber核函数
            new ceres::LossFunctionWrapper(
                    new ceres::HuberLoss(huber_loss_delta * image_point_uncertainty),
                    ceres::TAKE_OWNERSHIP));

    problem_information->addResidualBlock(//构造视觉重投影误差,输入残差类型，视觉costfunction，核函数，优化变量
            ceres_error_terms::ResidualType::kVisualReprojectionError,
            visual_term_cost, loss_function, cost_term_args);

    if (error_term_type !=
        ceres_error_terms::visual::VisualErrorType::kLocalKeyframe)
    {//如果当前节点是首次观测到这个地图点的节点，那么它就对应于kLocalKeyframe这种情况，
        // 这时候不需要优化基帧和关键帧姿态，所以也就不用添加局部参数化
        problem_information->setParameterization(
                landmark_store_vertex_q_IM__M_p_MI, pose_parameterization);
        problem_information->setParameterization(
                vertex_q_IM__M_p_MI, pose_parameterization);
//如果当前节点是和首次观测到这个地图点的节点在同一个任务下，那么它就对应于kLocalMission这种情况，
        // 这时候不需要优化基帧姿态，所以也就不用添加局部参数化
        if (error_term_type ==
            ceres_error_terms::visual::VisualErrorType::kGlobal) {
            problem_information->setParameterization(
                    landmark_store_baseframe_q_GM__G_p_GM, baseframe_parameterization);
            problem_information->setParameterization(
                    observer_baseframe_q_GM__G_p_GM, baseframe_parameterization);
        }
    }

    problem_information->setParameterization(//三种情况都需要添加相机位姿的局部参数化
            camera_q_CI, camera_parameterization);

    //这里可以手动设置一下固定哪些优化参数
    if (fix_landmark_positions) {
        problem_information->setParameterBlockConstant(landmark.get_p_B_Mutable());
    }
    if (fix_intrinsics) {
        problem_information->setParameterBlockConstant(
                camera_ptr->getParametersMutable());
        if (camera_ptr->getDistortion().getType() !=
            aslam::Distortion::Type::kNoDistortion) {
            problem_information->setParameterBlockConstant(
                    camera_ptr->getDistortionMutable()->getParametersMutable());
        }
    }
    if (fix_extrinsics_rotation) {
        problem_information->setParameterBlockConstant(camera_q_CI);
    }
    if (fix_extrinsics_translation) {
        problem_information->setParameterBlockConstant(camera_C_p_CI);
    }

    problem->getProblemBookkeepingMutable()->landmarks_in_problem.emplace(
            landmark_id, visual_term_cost.get());//在problem_books_中保存地图点和视觉cost_fun的信息
    return true;
}

void addVisualTermsForVertices(//增加所有节点中相机对地图点的的重投影视觉约束
    const bool fix_landmark_positions, const bool fix_intrinsics,
    const bool fix_extrinsics_rotation, const bool fix_extrinsics_translation,
    const size_t min_landmarks_per_frame,
    const std::shared_ptr<ceres::LocalParameterization>& pose_parameterization,
    const std::shared_ptr<ceres::LocalParameterization>&
        baseframe_parameterization,
    const std::shared_ptr<ceres::LocalParameterization>&
        camera_parameterization,
    const pose_graph::VertexIdList& vertices, OptimizationProblem* problem)
{
    CHECK_NOTNULL(problem);

    vi_map::VIMap* map = CHECK_NOTNULL(problem->getMapMutable());//得到地图
    const vi_map::MissionIdSet& missions_to_optimize = problem->getMissionIds();//得到当前所有的任务set

    for (const pose_graph::VertexId& vertex_id : vertices) //遍历某个任务下的所有的节点id
    {
        vi_map::Vertex& vertex = map->getVertex(vertex_id);//通过id找到这个节点
        const size_t num_frames = vertex.numFrames();//每个节点代表一个状态，n相机系统在每个节点下就有n帧
        for (size_t frame_idx = 0u; frame_idx < num_frames; ++frame_idx) //遍历所有相机
        {
            if (!vertex.isVisualFrameSet(frame_idx) ||//检查每个相机是否合法
                !vertex.isVisualFrameValid(frame_idx))
            {
                continue;
            }

            if (min_landmarks_per_frame > 0)
            {//如果对每帧的最小观测地图点有要求
                vi_map_helpers::VIMapQueries queries(*map);
                const size_t num_frame_good_landmarks =//输入的是当前节点和当前相机id，输出是当前这个节点的这帧观测到的好的地图点的数量
                        queries.getNumWellConstrainedLandmarks(vertex, frame_idx);
                if (num_frame_good_landmarks < min_landmarks_per_frame) {//如果数量小，就跳过
                    VLOG(3) << " Skipping this visual keyframe. Only "
                            << num_frame_good_landmarks
                            << " well constrained landmarks, but "
                            << min_landmarks_per_frame << " required";
                    continue;
                }
            }
            problem->getProblemBookkeepingMutable()->keyframes_in_problem.emplace(//记录要优化的节点
                    vertex_id);

            const aslam::VisualFrame& visual_frame = vertex.getVisualFrame(frame_idx);//这个节点中的这个相机
            //getNumKeypointMeasurements在maplab_ws/src/maplab/aslam_cv2/aslam_cv_frames/src/visual-frame.cc中
            // aslam::channels::get_VISUAL_KEYPOINT_MEASUREMENTS_Data(channels_);返回的是所有特征点的像素值Eigen::Matrix2Xd
            const size_t num_keypoints = visual_frame.getNumKeypointMeasurements();//特征点的数量

            for (size_t keypoint_idx = 0u; keypoint_idx < num_keypoints;//遍历所有的特征点
                 ++keypoint_idx)
            {
                const vi_map::LandmarkId landmark_id =//得到当前这个节点的这个相机的这个特征点对应的地图点id
                        vertex.getObservedLandmarkId(frame_idx, keypoint_idx);
                // Invalid landmark_id means that the keypoint is not actually
                // associated to an existing landmark object.
                //当2d没有对应的3d地图点信息时，会不合法
                if (!landmark_id.isValid()) {
                    continue;
                }

                //map中的landmark_index中的LandmarkToVertexMap存了这个地图点第一次被哪个节点观测到
                const vi_map::Vertex& landmark_store_vertex =//这个地图点第一次是被哪个节点观测到的
                        map->getLandmarkStoreVertex(landmark_id);

                // Skip if the landmark is stored in a mission that should not be
                // optimized.
                //如果这个节点不在需要优化的任务中，就跳过
                if (missions_to_optimize.count(landmark_store_vertex.getMissionId()) ==
                    0u) {
                    continue;
                }

                vi_map::Landmark& landmark = map->getLandmark(landmark_id);//通过id得到这个地图点

                // Skip if the current landmark is not well constrained.
                //选完关键帧以后已经进行过地图点质量的判定了,所以这里这是读取一下地图点的质量，如果不是kGood就跳过
                if (!vi_map::isLandmarkWellConstrained(*map, landmark)) {
                    continue;
                }

                //能来到这步的都是好点了
                //增加每一个特征点的视觉因子
                addVisualTermForKeypoint(
                        keypoint_idx, frame_idx, fix_landmark_positions, fix_intrinsics,
                        fix_extrinsics_rotation, fix_extrinsics_translation,
                        pose_parameterization, baseframe_parameterization,
                        camera_parameterization, &vertex, problem);
            }
        }
    }
}

void addVisualTerms(//添加视觉的因子
    const bool fix_landmark_positions, const bool fix_intrinsics,
    const bool fix_extrinsics_rotation, const bool fix_extrinsics_translation,
    const size_t min_landmarks_per_frame, OptimizationProblem* problem)
{
    CHECK_NOTNULL(problem);

    vi_map::VIMap* map = CHECK_NOTNULL(problem->getMapMutable());

    const OptimizationProblem::LocalParameterizations& parameterizations =
            problem->getLocalParameterizations();

    const vi_map::MissionIdSet& missions_to_optimize = problem->getMissionIds();//得到要优化的任务set
    for (const vi_map::MissionId& mission_id : missions_to_optimize) //遍历所有的任务
    {
        pose_graph::VertexIdList vertices;
        map->getAllVertexIdsInMissionAlongGraph(mission_id, &vertices);//找到每个任务的所有节点
        //增加所有节点中相机对地图点的的重投影视觉约束
        addVisualTermsForVertices(
                fix_landmark_positions, fix_intrinsics, fix_extrinsics_rotation,
                fix_extrinsics_translation, min_landmarks_per_frame,
                parameterizations.pose_parameterization,
                parameterizations.baseframe_parameterization,
                parameterizations.quaternion_parameterization, vertices, problem);
    }
}

void addInertialTerms(//增加imu的约束
    const bool fix_gyro_bias, const bool fix_accel_bias,
    const bool fix_velocity, const double gravity_magnitude,
    OptimizationProblem* problem)
{
    CHECK_NOTNULL(problem);

    vi_map::VIMap* map = CHECK_NOTNULL(problem->getMapMutable());
    const vi_map::SensorManager& sensor_manger = map->getSensorManager();//传感器管理器

    const OptimizationProblem::LocalParameterizations& parameterizations =
            problem->getLocalParameterizations();

    size_t num_residuals_added = 0;
    const vi_map::MissionIdSet& missions_to_optimize = problem->getMissionIds();//得到当前优化问题的所有任务set
    for (const vi_map::MissionId& mission_id : missions_to_optimize) //遍历所有的任务
    {
        pose_graph::EdgeIdList edges;
        map->getAllEdgeIdsInMissionAlongGraph(//当前任务所有kViwls类型的边
                mission_id, pose_graph::Edge::EdgeType::kViwls, &edges);

        const vi_map::Imu& imu_sensor =//imu_sensor里有imu重力大小，加速度阈值，协方差等等
                sensor_manger.getSensorForMission<vi_map::Imu>(mission_id);
        const vi_map::ImuSigmas& imu_sigmas = imu_sensor.getImuSigmas();//协方差

        num_residuals_added += addInertialTermsForEdges(//每条边都有imu数据，添加进imu因子
                fix_gyro_bias, fix_accel_bias, fix_velocity, gravity_magnitude,
                imu_sigmas, parameterizations.pose_parameterization, edges, problem);
    }

    VLOG(1) << "Added " << num_residuals_added << " inertial residuals.";
}

int addInertialTermsForEdges(
    const bool fix_gyro_bias, const bool fix_accel_bias,//一些固定参数的选项
    const bool fix_velocity, const double gravity_magnitude,//当地重力
    const vi_map::ImuSigmas& imu_sigmas,//imu的加速度和imu噪声以及bias的噪声
    const std::shared_ptr<ceres::LocalParameterization>& pose_parameterization,//局部参数化
    const pose_graph::EdgeIdList& edges, OptimizationProblem* problem)//这个任务里所有的imu测量边
{
    CHECK(pose_parameterization != nullptr);
    CHECK_NOTNULL(problem);

    vi_map::VIMap* map = CHECK_NOTNULL(problem->getMapMutable());//得到当前地图
    OptimizationStateBuffer* buffer =
            CHECK_NOTNULL(problem->getOptimizationStateBufferMutable());//要优化的状态存储器

    int num_residuals_added = 0;
    for (const pose_graph::EdgeId edge_id : edges) //遍历每一条边id
    {
        const vi_map::ViwlsEdge& inertial_edge =//找到这条imu测量边
                map->getEdgeAs<vi_map::ViwlsEdge>(edge_id);

        std::shared_ptr<ceres_error_terms::InertialErrorTerm> inertial_term_cost(//初始化imu的代价函数
                new ceres_error_terms::InertialErrorTerm(
                        inertial_edge.getImuData(), inertial_edge.getImuTimestamps(),
                        imu_sigmas.gyro_noise_density,
                        imu_sigmas.gyro_bias_random_walk_noise_density,
                        imu_sigmas.acc_noise_density,
                        imu_sigmas.acc_bias_random_walk_noise_density, gravity_magnitude));

        vi_map::Vertex& vertex_from = map->getVertex(inertial_edge.from());//边的是哪个节点出的（前状态）
        vi_map::Vertex& vertex_to = map->getVertex(inertial_edge.to());//边的是哪个节点入的（后状态）

        problem->getProblemBookkeepingMutable()->keyframes_in_problem.emplace(//记录被优化的节点id
                vertex_from.id());
        problem->getProblemBookkeepingMutable()->keyframes_in_problem.emplace(
                vertex_to.id());

        double* vertex_from_q_IM__M_p_MI =
                buffer->get_vertex_q_IM__M_p_MI_JPL(inertial_edge.from());//前状态的位姿（JPL格式）
        double* vertex_to_q_IM__M_p_MI =
                buffer->get_vertex_q_IM__M_p_MI_JPL(inertial_edge.to());//后状态的位姿（JPL格式）

        problem->getProblemInformationMutable()->addResidualBlock(//添加残差块,这里的类型是kInertial
                ceres_error_terms::ResidualType::kInertial, inertial_term_cost, nullptr,
                {vertex_from_q_IM__M_p_MI, vertex_from.getGyroBiasMutable(),
                 vertex_from.get_v_M_Mutable(), vertex_from.getAccelBiasMutable(),
                 vertex_to_q_IM__M_p_MI, vertex_to.getGyroBiasMutable(),
                 vertex_to.get_v_M_Mutable(), vertex_to.getAccelBiasMutable()});

        problem->getProblemInformationMutable()->setParameterization(//局部参数化
                vertex_from_q_IM__M_p_MI, pose_parameterization);
        problem->getProblemInformationMutable()->setParameterization(
                vertex_to_q_IM__M_p_MI, pose_parameterization);

        //设置一下需要fix的优化变量
        if (fix_gyro_bias) {
            problem->getProblemInformationMutable()->setParameterBlockConstant(
                    vertex_to.getGyroBiasMutable());
            problem->getProblemInformationMutable()->setParameterBlockConstant(
                    vertex_from.getGyroBiasMutable());
        }
        if (fix_accel_bias) {
            problem->getProblemInformationMutable()->setParameterBlockConstant(
                    vertex_to.getAccelBiasMutable());
            problem->getProblemInformationMutable()->setParameterBlockConstant(
                    vertex_from.getAccelBiasMutable());
        }
        if (fix_velocity) {
            problem->getProblemInformationMutable()->setParameterBlockConstant(
                    vertex_to.get_v_M_Mutable());
            problem->getProblemInformationMutable()->setParameterBlockConstant(
                    vertex_from.get_v_M_Mutable());
        }

        ++num_residuals_added;
    }

    return num_residuals_added;
}

}  // namespace map_optimization
