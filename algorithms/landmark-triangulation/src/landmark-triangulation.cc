#include "landmark-triangulation/landmark-triangulation.h"

#include <functional>
#include <string>
#include <unordered_map>

#include <aslam/common/statistics/statistics.h>
#include <aslam/triangulation/triangulation.h>
#include <maplab-common/multi-threaded-progress-bar.h>
#include <maplab-common/parallel-process.h>
#include <vi-map/landmark-quality-metrics.h>
#include <vi-map/vi-map.h>


#include <iostream>

#include "landmark-triangulation/pose-interpolator.h"

namespace landmark_triangulation {
typedef AlignedUnorderedMap<aslam::FrameId, aslam::Transformation>
    FrameToPoseMap;

namespace {
void interpolateVisualFramePosesAllMissions(
    const vi_map::VIMap& map, FrameToPoseMap* interpolated_frame_poses) {
  CHECK_NOTNULL(interpolated_frame_poses)->clear();
  // Loop over all missions, vertices and frames and add the interpolated poses
  // to the map.
  size_t total_num_frames = 0u;
  vi_map::MissionIdList mission_ids;
  map.getAllMissionIds(&mission_ids);
  for (const vi_map::MissionId mission_id : mission_ids) {
    // Check if there is IMU data.
    std::unordered_map<pose_graph::VertexId, int64_t> vertex_to_time_map;
    PoseInterpolator imu_timestamp_collector;
    imu_timestamp_collector.getVertexToTimeStampMap(
        map, mission_id, &vertex_to_time_map);
    if (vertex_to_time_map.empty()) {
      VLOG(2) << "Couldn't find any IMU data to interpolate exact landmark "
                 "observer positions in "
              << "mission " << mission_id;
      continue;
    }

    pose_graph::VertexIdList vertex_ids;
    map.getAllVertexIdsInMissionAlongGraph(mission_id, &vertex_ids);

    // Compute upper bound for number of VisualFrames.
    const aslam::NCamera& ncamera =
        map.getSensorManager().getNCameraForMission(mission_id);
    const unsigned int upper_bound_num_frames =
        vertex_ids.size() * ncamera.numCameras();

    // Extract the timestamps for all VisualFrames.
    std::vector<aslam::FrameId> frame_ids(upper_bound_num_frames);
    Eigen::Matrix<int64_t, 1, Eigen::Dynamic> pose_timestamps(
        upper_bound_num_frames);
    unsigned int frame_counter = 0u;
    for (const pose_graph::VertexId& vertex_id : vertex_ids) {
      const vi_map::Vertex& vertex = map.getVertex(vertex_id);
      for (unsigned int frame_idx = 0u; frame_idx < vertex.numFrames();
           ++frame_idx) {
        if (vertex.isFrameIndexValid(frame_idx)) {
          CHECK_LT(frame_counter, upper_bound_num_frames);
          const aslam::VisualFrame& visual_frame =
              vertex.getVisualFrame(frame_idx);
          const int64_t timestamp = visual_frame.getTimestampNanoseconds();
          // Only interpolate if the VisualFrame timestamp and the vertex
          // timestamp do not match.
          std::unordered_map<pose_graph::VertexId, int64_t>::const_iterator it =
              vertex_to_time_map.find(vertex_id);
          if (it != vertex_to_time_map.end() && it->second != timestamp) {
            pose_timestamps(0, frame_counter) = timestamp;
            frame_ids[frame_counter] = visual_frame.getId();
            ++frame_counter;
          }
        }
      }
    }

    // Shrink time stamp and frame id arrays if necessary.
    if (upper_bound_num_frames > frame_counter) {
      frame_ids.resize(frame_counter);
      pose_timestamps.conservativeResize(Eigen::NoChange, frame_counter);
    }

    // Reserve enough space in the frame_to_pose_map to add the poses of the
    // current mission.
    total_num_frames += frame_counter;
    interpolated_frame_poses->reserve(total_num_frames);

    // Interpolate poses for all the VisualFrames.
    if (frame_counter > 0u) {
      VLOG(1) << "Interpolating the exact visual frame poses for "
              << frame_counter << " frames of mission " << mission_id;
      PoseInterpolator pose_interpolator;
      aslam::TransformationVector poses_M_I;
      pose_interpolator.getPosesAtTime(
          map, mission_id, pose_timestamps, &poses_M_I);
      CHECK_EQ(poses_M_I.size(), frame_counter);
      for (size_t frame_num = 0u; frame_num < frame_counter; ++frame_num) {
        interpolated_frame_poses->emplace(
            frame_ids[frame_num], poses_M_I.at(frame_num));
      }
      CHECK_EQ(interpolated_frame_poses->size(), total_num_frames);
    } else {
      VLOG(10) << "No frames found for mission " << mission_id
               << " that need to be interpolated.";
    }
  }
  if (total_num_frames == 0u) {
    VLOG(2) << "No frame pose in any of the missions needs interpolation!";
  }
}

    void retriangulateLandmarksOfVertex(
            const FrameToPoseMap& interpolated_frame_poses,
            pose_graph::VertexId storing_vertex_id, vi_map::VIMap* map)
    {
        CHECK_NOTNULL(map);
        vi_map::Vertex& storing_vertex = map->getVertex(storing_vertex_id);//从地图中通过id得到当前需要重新三角化的节点
        vi_map::LandmarkStore& landmark_store = storing_vertex.getLandmarks();//从节点中得到当前节点可能观察到的所有地图点
        // 其中LandmarkStore是一个地图点的管理器，里面记录了这个地图点的id以及这个地图点被其他帧观测的信息

        const aslam::Transformation& T_M_I_storing = storing_vertex.get_T_M_I();//得到当前节点作对应的body坐标系在当前任务坐标系下的位姿
        const aslam::Transformation& T_G_M_storing =
                const_cast<const vi_map::VIMap*>(map)//得到当前任务的基准帧相对于世界坐标系的坐标变换
                        ->getMissionBaseFrameForVertex(storing_vertex_id)
                        .get_T_G_M();

        std::cout<<"T_G_M_storing"<<std::endl<<T_G_M_storing.getRotationMatrix()<<std::endl<<T_G_M_storing.getPosition().transpose()<<std::endl;


        const aslam::Transformation T_G_I_storing = T_G_M_storing * T_M_I_storing;//得到当前节点作对应的body坐标系在世界坐标系下的位姿

        for (vi_map::Landmark& landmark : landmark_store)
        {//遍历每一个由这个节点首次看到的地图点
            // The following have one entry per measurement:
            Eigen::Matrix3Xd G_bearing_vectors;
            Eigen::Matrix3Xd p_G_C_vector;


            landmark.setQuality(vi_map::Landmark::Quality::kBad);//初始化质量时默认设成不好的质量

            const vi_map::KeypointIdentifierList& observations =
                    landmark.getObservations();//这个地图点所有的观测信息，就是哪一帧的哪一个特征点观测到了这个地图点
            if (observations.size() < 2u)
            {
                statistics::StatsCollector stats(
                        "Landmark triangulation failed too few observations.");
                stats.IncrementOne();
                continue;
            }

            G_bearing_vectors.resize(Eigen::NoChange, observations.size());//这是所有相机光心到这个地图点的射线
            p_G_C_vector.resize(Eigen::NoChange, observations.size());

            int num_measurements = 0;
            for (const vi_map::KeypointIdentifier& observation : observations)
            {//对所有的观测信息进行遍历
                const pose_graph::VertexId& observer_id = observation.frame_id.vertex_id;//得到当前观测点对应的节点id
                CHECK(map->hasVertex(observer_id))
                        << "Observer " << observer_id << " of store landmark "
                        << landmark.id() << " not in currently loaded map!";

                const vi_map::Vertex& observer =
                        const_cast<const vi_map::VIMap*>(map)->getVertex(observer_id);//通过节点id得到节点
                //得到节点id中哪一帧看到的这个地图点，因为考虑到可能是多目相机系统
                const aslam::VisualFrame& visual_frame =
                        observer.getVisualFrame(observation.frame_id.frame_index);
                //得到当前这个节点所在的任务的坐标系相对于世界坐标系的变换
                const aslam::Transformation& T_G_M_observer =
                        const_cast<const vi_map::VIMap*>(map)
                                ->getMissionBaseFrameForVertex(observer_id)
                                .get_T_G_M();

                // If there are precomputed/interpolated T_M_I, use those.
                //如果有预计算/插值T_M_I，使用它们。
                //这里这个插值pose是什么时候计算的不太清楚
                aslam::Transformation T_G_I_observer;
                FrameToPoseMap::const_iterator it =
                        interpolated_frame_poses.find(visual_frame.getId());
                if (it != interpolated_frame_poses.end()) {
                    //如果有插值pose话，直接用插值得到的pose和T_G_M得到世界坐标下当前节点的body坐标系的pose
                    const aslam::Transformation& T_M_I_observer = it->second;
                    T_G_I_observer = T_G_M_observer * T_M_I_observer;
                } else {
                    //如果没有的话，就用vio得到的pose和T_G_M得到世界坐标下当前节点的body坐标系的pose（这里是我猜的，因为没有看到成熟中哪里计算了插值信息）
                    const aslam::Transformation& T_M_I_observer = observer.get_T_M_I();
                    T_G_I_observer = T_G_M_observer * T_M_I_observer;
                }

                Eigen::Vector2d measurement =//得到这个节点中这个相机观测到这个地图点的坐标（像素坐标）
                        visual_frame.getKeypointMeasurement(observation.keypoint_index);

                Eigen::Vector3d C_bearing_vector;
                //这里对应了多种相机模型
                //输入，像素坐标，在这里是measurement
                //输出，归一化坐标，在这里就是C_bearing_vector，bool是因为有些模型将像素坐标投影到相机坐标系后需要判定一下投影的归一化坐标是否合法
                bool projection_result =observer.getCamera(observation.frame_id.frame_index)
                                ->backProject3(measurement, &C_bearing_vector);
                if (!projection_result) {
                    statistics::StatsCollector stats(
                            "Landmark triangulation failed proj failed.");
                    stats.IncrementOne();
                    continue;
                }
                //maplab把帧和相机信息是分开放的，ncamera.h存放了多相机系统的标定信息，所以这里需要把这帧对应的这个相机id找出来，
                // 比如我这帧使用左相机拍的，那么找到的cam_id就是对应着左相机的id，这样也就能找到左相机的标定信息
                const aslam::CameraId& cam_id =
                        observer.getCamera(observation.frame_id.frame_index)->getId();
                //得到这一帧的相机在世界坐标系下的位姿
                aslam::Transformation T_G_C =
                        (T_G_I_observer *
                         observer.getNCameras()->get_T_C_B(cam_id).inverse());
                G_bearing_vectors.col(num_measurements) =
                        T_G_C.getRotationMatrix() * C_bearing_vector;
                p_G_C_vector.col(num_measurements) = T_G_C.getPosition();//存的是相机位置
                ++num_measurements;
            }
            G_bearing_vectors.conservativeResize(Eigen::NoChange, num_measurements);
            p_G_C_vector.conservativeResize(Eigen::NoChange, num_measurements);

            if (num_measurements < 2) {
                statistics::StatsCollector stats("Landmark triangulation too few meas.");
                stats.IncrementOne();
                continue;
            }

            Eigen::Vector3d p_G_fi;
            //triangulation_result就是三角化是否成功
            aslam::TriangulationResult triangulation_result =
                    aslam::linearTriangulateFromNViews(
                            G_bearing_vectors, p_G_C_vector, &p_G_fi);

            //如果三角化成功了
            if (triangulation_result.wasTriangulationSuccessful())
            {
                //如果三角化成功，对这个地图点设置它在body坐标系下的位置
                landmark.set_p_B(T_G_I_storing.inverse() * p_G_fi);

                //对地图点的质量进行检查
                constexpr bool kReEvaluateQuality = true;
                //isLandmarkWellConstrained是去判断这个地图点是不是可靠的
                //判断依据：1.相机到路标点的距离2.两个相机观测一个路标点形成的夹角
                if (vi_map::isLandmarkWellConstrained(
                        *map, landmark, kReEvaluateQuality))
                {
                    statistics::StatsCollector stats_good("Landmark good");
                    stats_good.IncrementOne();
                    landmark.setQuality(vi_map::Landmark::Quality::kGood);
                } else
                    {
                    statistics::StatsCollector stats("Landmark bad after triangulation");
                    stats.IncrementOne();
                }
            } else {//如果三角化不成功
                statistics::StatsCollector stats("Landmark triangulation failed");
                stats.IncrementOne();
                if (triangulation_result.status() ==
                    aslam::TriangulationResult::UNOBSERVABLE) //三角化步骤失败时我看返回的都是这个
                {
                    statistics::StatsCollector stats(
                            "Landmark triangulation failed - unobservable");
                    stats.IncrementOne();
                } else if (
                        triangulation_result.status() ==
                        aslam::TriangulationResult::UNINITIALIZED) {
                    statistics::StatsCollector stats(
                            "Landmark triangulation failed - uninitialized");
                    stats.IncrementOne();
                }
            }
        }
    }

bool retriangulateLandmarksOfMission(//这里才是真正的重新三角化的步骤
    const vi_map::MissionId& mission_id,
    const FrameToPoseMap& interpolated_frame_poses, vi_map::VIMap* map) {//插值帧的姿态是什么意思呢？
  CHECK_NOTNULL(map); //检查地图的指针是否为空

  VLOG(1) << "Getting vertices of mission: " << mission_id;
  pose_graph::VertexIdList relevant_vertex_ids;
  map->getAllVertexIdsInMissionAlongGraph(mission_id, &relevant_vertex_ids);//通过图，得到这个任务的所有的节点

  const size_t num_vertices = relevant_vertex_ids.size();
  VLOG(1) << "Retriangulating landmarks of " << num_vertices << " vertices.";

  common::MultiThreadedProgressBar progress_bar;
  std::function<void(const std::vector<size_t>&)> retriangulator =
      [&relevant_vertex_ids, map, &progress_bar,
       &interpolated_frame_poses](const std::vector<size_t>& batch)
       {
        progress_bar.setNumElements(batch.size());
        size_t num_processed = 0u;
        for (size_t item : batch) {
          CHECK_LT(item, relevant_vertex_ids.size());
          retriangulateLandmarksOfVertex(//进行重新三角化，输入参数：所有帧的pose，要重新三角化的节点id，vimap地图
              interpolated_frame_poses, relevant_vertex_ids[item], map);
          progress_bar.update(++num_processed);
        }
      };

  static constexpr bool kAlwaysParallelize = false;
  const size_t num_threads = common::getNumHardwareThreads();
  common::ParallelProcess(
      num_vertices, retriangulator, kAlwaysParallelize, num_threads);
  return true;
}
}  // namespace

bool retriangulateLandmarks(vi_map::VIMap* map) {
  CHECK_NOTNULL(map);
  vi_map::MissionIdList all_mission_ids;
  map->getAllMissionIds(&all_mission_ids);

  FrameToPoseMap interpolated_frame_poses;
  interpolateVisualFramePosesAllMissions(*map, &interpolated_frame_poses);

  for (const vi_map::MissionId& mission_id : all_mission_ids) {
    retriangulateLandmarksOfMission(mission_id, interpolated_frame_poses, map);
  }
  return true;
}

bool retriangulateLandmarksOfMission(//在localization-map-creation重新三角化部分会进行调用
    const vi_map::MissionId& mission_id, vi_map::VIMap* map)
    {

    //typedef AlignedUnorderedMap<aslam::FrameId, aslam::Transformation> FrameToPoseMap;
    //这里我感觉FrameToPoseMap是每一帧对应的pose的map么？
  const FrameToPoseMap empty_frame_to_pose_map;
  return retriangulateLandmarksOfMission(
      mission_id, empty_frame_to_pose_map, map);
}

void retriangulateLandmarksOfVertex(
    const pose_graph::VertexId& storing_vertex_id, vi_map::VIMap* map) {
  CHECK_NOTNULL(map);
  FrameToPoseMap empty_frame_to_pose_map;
  retriangulateLandmarksOfVertex(
      empty_frame_to_pose_map, storing_vertex_id, map);
}

}  // namespace landmark_triangulation
