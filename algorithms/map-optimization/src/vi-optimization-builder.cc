#include "map-optimization/vi-optimization-builder.h"

#include <gflags/gflags.h>
#include <vi-map-helpers/mission-clustering-coobservation.h>

#include "map-optimization/optimization-state-fixing.h"

DEFINE_bool(
        ba_use_Lidar, false, "Whether or not to include visual error-terms.");

DEFINE_bool(
    ba_include_visual, true, "Whether or not to include visual error-terms.");
DEFINE_bool(
    ba_include_inertial, true, "Whether or not to include IMU error-terms.");

DEFINE_bool(
    ba_fix_ncamera_intrinsics, true,
    "Whether or not to fix the intrinsics of the ncamera(s).");
DEFINE_bool(
    ba_fix_ncamera_extrinsics_rotation, true,
    "Whether or not to fix the rotation extrinsics of the ncamera(s).");
DEFINE_bool(
    ba_fix_ncamera_extrinsics_translation, true,
    "Whether or not to fix the translation extrinsics of the ncamera(s).");
DEFINE_bool(
    ba_fix_landmark_positions, false,
    "Whether or not to fix the positions of the landmarks.");
DEFINE_bool(
    ba_fix_accel_bias, false,
    "Whether or not to fix the bias of the IMU accelerometer.");
DEFINE_bool(
    ba_fix_gyro_bias, false,
    "Whether or not to fix the bias of the IMU gyroscope.");
DEFINE_bool(
    ba_fix_velocity, false,
    "Whether or not to fix the velocity of the vertices.");

DEFINE_double(
    ba_latitude, common::locations::kLatitudeZurichDegrees,
    "Latitude to estimate the gravity magnitude.");
DEFINE_double(
    ba_altitude_meters, common::locations::kAltitudeZurichMeters,
    "Altitude in meters to estimate the gravity magnitude.");

DEFINE_int32(
    ba_min_landmark_per_frame, 0,
    "Minimum number of landmarks a frame must observe to be included in the "
    "problem.");

namespace map_optimization {

ViProblemOptions ViProblemOptions::initFromGFlags() //初始化vio优化器
{
  ViProblemOptions options;

  //一些优化时要固定的参数
  options.add_inertial_constraints = FLAGS_ba_include_inertial;//true
  options.fix_gyro_bias = FLAGS_ba_fix_gyro_bias;//false
  options.fix_accel_bias = FLAGS_ba_fix_accel_bias;//false
  options.fix_velocity = FLAGS_ba_fix_velocity;//false
  options.min_landmarks_per_frame = FLAGS_ba_min_landmark_per_frame;//0

  common::GravityProvider gravity_provider(
      FLAGS_ba_altitude_meters, FLAGS_ba_latitude);//FLAGS_ba_altitude_meters是392，FLAGS_ba_latitude是47.22
  options.gravity_magnitude = gravity_provider.getGravityMagnitude();

  // Visual constraints.
  options.add_visual_constraints = FLAGS_ba_include_visual;//true
  options.fix_intrinsics = FLAGS_ba_fix_ncamera_intrinsics;//true
  options.fix_extrinsics_rotation = FLAGS_ba_fix_ncamera_extrinsics_rotation;//true
  options.fix_extrinsics_translation =
      FLAGS_ba_fix_ncamera_extrinsics_translation;//true
  options.fix_landmark_positions = FLAGS_ba_fix_landmark_positions;//false看来是先优化bias，速度，还有路标点

  return options;
}

//构造vio优化问题
    OptimizationProblem* constructViProblem
            (
                    const vi_map::MissionIdSet& mission_ids, const ViProblemOptions& options,
                    vi_map::VIMap* map) {
        CHECK(map);
        CHECK(options.isValid());

        LOG_IF(
                FATAL,
                !options.add_visual_constraints && !options.add_inertial_constraints)
        << "Either enable visual or inertial constraints; otherwise don't call "
        << "this function.";
        //优化问题的构造
        //将节点位姿，基准帧位姿，相机外参等保存到state_buffer_中，q要转成JPL格式，因为后面的误差推导是用JPL形式来推导的
        OptimizationProblem* problem = new OptimizationProblem(map, mission_ids);
        if (options.add_visual_constraints) {//增加视觉约束
            addVisualTerms(//是否固定地图点位置，内参，外参R，t
                    options.fix_landmark_positions, options.fix_intrinsics,
                    options.fix_extrinsics_rotation, options.fix_extrinsics_translation,
                    options.min_landmarks_per_frame, problem);
        }
        if (options.add_inertial_constraints) {//增加imu约束,如果用激光的话，就不用去管这个
            addInertialTerms(
                    options.fix_gyro_bias, options.fix_accel_bias, options.fix_velocity,
                    options.gravity_magnitude, problem);
        }

        // Fixing open DoF of the visual(-inertial) problem. We assume that if there
        // is inertial data, that all missions will have them.
        const bool visual_only =//判断一下是不是纯视觉问题还是vio问题,maplab假设了所有任务应该要么都是vio，要么都是纯视觉
                options.add_visual_constraints && !options.add_inertial_constraints;

        // Determine and apply the gauge fixes.
        //vio和纯视觉系统是有不同的固定位姿以及尺度的策略的
        MissionClusterGaugeFixes fixes_of_mission_cluster;

        if(FLAGS_ba_use_Lidar)
        {
            fixes_of_mission_cluster.position_dof_fixed = true;
            fixes_of_mission_cluster.rotation_dof_fixed = FixedRotationDoF::kAll;
            fixes_of_mission_cluster.scale_fixed = true;
        }
        else if (!visual_only)
        {
            //如果是vio系统，就固定第一个任务的第一个节点的位置和yaw，并且认为尺度存在漂移的情况
            fixes_of_mission_cluster.position_dof_fixed = true;
            fixes_of_mission_cluster.rotation_dof_fixed = FixedRotationDoF::kYaw;
            fixes_of_mission_cluster.scale_fixed = false;
        } else {            //如果是纯视觉系统，就固定第一个任务的第一个节点的位姿，并且固定尺度
            fixes_of_mission_cluster.position_dof_fixed = true;
            fixes_of_mission_cluster.rotation_dof_fixed = FixedRotationDoF::kAll;
            fixes_of_mission_cluster.scale_fixed = true;
        }
        //之前对所有的任务进行了分簇，每一簇的子任务之间的存在共视的地图点的，但是簇与簇之间是不存在共视的地图点的
        const size_t num_clusters = problem->getMissionCoobservationClusters().size();
        std::vector<MissionClusterGaugeFixes> vi_cluster_fixes(
                num_clusters, fixes_of_mission_cluster);

        // Merge with already applied fixes (if necessary).如果之前已经对这个vec的簇的任务设置过fix，那么需要合并fix

        const std::vector<MissionClusterGaugeFixes>* already_applied_cluster_fixes =
                problem->getAppliedGaugeFixesForInitialVertices();
        if (already_applied_cluster_fixes)
        {//合并fix，起始就是两种fix选择中，哪个限制的自由度更少选哪个
            std::vector<MissionClusterGaugeFixes> merged_fixes;
            mergeGaugeFixes(
                    vi_cluster_fixes, *already_applied_cluster_fixes, &merged_fixes);
            //对首帧进行fix
            problem->applyGaugeFixesForInitialVertices(merged_fixes);
        } else {
            problem->applyGaugeFixesForInitialVertices(vi_cluster_fixes);
        }

        // Baseframes are fixed in the non mission-alignment problems.
        //对所有基准帧进行fix
        fixAllBaseframesInProblem(problem);
        fixAllPoseInProblem(problem);
        return problem;
    }

}  // namespace map_optimization
