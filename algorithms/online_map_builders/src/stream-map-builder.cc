#include "online-map-builders/stream-map-builder.h"

#include <aslam/common/stl-helpers.h>
#include <aslam/frames/visual-nframe.h>
#include <glog/logging.h>
#include <vi-map-helpers/vi-map-manipulation.h>
#include <vi-map-helpers/vi-map-queries.h>
#include <vi-map/check-map-consistency.h>
#include <vi-map/vi-map.h>
#include <vio-common/vio-types.h>
#include <vio-common/vio-update.h>

DEFINE_bool(
    map_builder_save_image_as_resources, false,
    "Store the images associated with the visual frames to the map resource "
    "folder.");

namespace online_map_builders {

const vi_map::VIMap* StreamMapBuilder::constMap() const {
  return map_;
}

pose_graph::VertexId StreamMapBuilder::getRootVertexId() const {
  CHECK(mission_id_.isValid());
  return map_->getMission(mission_id_).getRootVertexId();
}

pose_graph::VertexId StreamMapBuilder::getLastVertexId() const {
  CHECK(last_vertex_.isValid());
  return last_vertex_;
}

StreamMapBuilder::StreamMapBuilder(
    const std::shared_ptr<aslam::NCamera>& camera_rig, vi_map::VIMap* map)
    : map_(CHECK_NOTNULL(map)),
      manipulation_(map),
      mission_id_(common::createRandomId<vi_map::MissionId>()),
      camera_rig_(camera_rig) {
  CHECK(camera_rig);
  map_->addNewMissionWithBaseframe(
      mission_id_, aslam::Transformation(),
      Eigen::Matrix<double, 6, 6>::Identity(), camera_rig,
      vi_map::Mission::BackBone::kViwls);
}

StreamMapBuilder::StreamMapBuilder(
    const std::shared_ptr<aslam::NCamera>& camera_rig,
    vi_map::Imu::UniquePtr imu, vi_map::VIMap* map)
    : map_(CHECK_NOTNULL(map)),
      manipulation_(map),
      mission_id_(common::createRandomId<vi_map::MissionId>()),
      camera_rig_(camera_rig) {
  CHECK(camera_rig);
  map_->addNewMissionWithBaseframe(
      mission_id_, aslam::Transformation(),
      Eigen::Matrix<double, 6, 6>::Identity(), camera_rig,
      vi_map::Mission::BackBone::kViwls);
  map_->getSensorManager().addSensor(std::move(imu), mission_id_);
}

void StreamMapBuilder::apply(const vio::VioUpdate& update) {
  CHECK(update.check());
  constexpr bool kDeepCopyNFrame = true;
  apply(update, kDeepCopyNFrame);
}

void StreamMapBuilder::apply(
    const vio::VioUpdate& update, bool deep_copy_nframe)
{
    CHECK(update.check());
    std::shared_ptr<aslam::VisualNFrame> nframe_to_insert;
    if (deep_copy_nframe)
    {
        const aslam::VisualNFrame& nframe_original =
                *update.keyframe_and_imudata->nframe;
        nframe_to_insert = aligned_shared<aslam::VisualNFrame>(
                nframe_original.getId(), nframe_original.getNumCameras());
        *nframe_to_insert = *update.keyframe_and_imudata->nframe;
    } else
        {//因为只是当前的信息，多相机的信息是会变化的，所以进行浅拷贝
        nframe_to_insert = update.keyframe_and_imudata->nframe;//当前的关键帧
    }
    CHECK(nframe_to_insert);

    if (!last_vertex_.isValid()) //还没有根节点时触发
    {

        addRootViwlsVertex(nframe_to_insert, update.vinode);// update.vinode里面是The state of a ViNode (pose, velocity and bias).
    } else {//如果有了根节点，加入的时候就还会加入这期间的imu时间戳，imu的测量值
        CHECK(mission_id_.isValid());
        addViwlsVertexAndEdge(
                nframe_to_insert, update.vinode,
                update.keyframe_and_imudata->imu_timestamps,
                update.keyframe_and_imudata->imu_measurements);

        if (update.localization_state == vio::LocalizationState::kLocalized ||//就是当前任务的位姿
            update.localization_state == vio::LocalizationState::kMapTracking) {
            map_->getMissionBaseFrameForMission(mission_id_).set_T_G_M(update.T_G_M);
        }
    }
}

//在图里加入根节点
void StreamMapBuilder::addRootViwlsVertex(
    const aslam::VisualNFrame::Ptr& nframe,//当前关键帧
    const vio::ViNodeState& vinode_state)//pose，速度，协方差
{
    CHECK(nframe);//
    CHECK(!last_vertex_.isValid()) << "Root vertex has already been set!";

    //加入这个vio地图节点
    pose_graph::VertexId root_vertex_id = addViwlsVertex(nframe, vinode_state);
    CHECK(root_vertex_id.isValid());

    CHECK(!constMap()->getMission(mission_id_).getRootVertexId().isValid())
    << "Root vertex has already been set for this mission.";
    map_->getMission(mission_id_).setRootVertexId(root_vertex_id);//设置当前任务根节点的id

    last_vertex_ = root_vertex_id;//最后一个节点
}

void StreamMapBuilder::addViwlsVertexAndEdge(
    const aslam::VisualNFrame::Ptr& nframe,
    const vio::ViNodeState& vinode_state,
    const Eigen::Matrix<int64_t, 1, Eigen::Dynamic>& imu_timestamps,
    const Eigen::Matrix<double, 6, Eigen::Dynamic>& imu_data) {
  CHECK(nframe);
  CHECK(last_vertex_.isValid())
      << "Need to have a previous vertex to connect to!";

  //
  pose_graph::VertexId new_vertex_id = addViwlsVertex(nframe, vinode_state);//加入新的节点，同时要和之前的节点去添加边

  addImuEdge(new_vertex_id, imu_timestamps, imu_data);//增加新的边

  if (kKeepNMostRecentImages > 0u) {//如果图片存储过多，就要release
    manipulation_.releaseOldVisualFrameImages(
        last_vertex_, kKeepNMostRecentImages);
  }
}


pose_graph::VertexId StreamMapBuilder::addViwlsVertex(
    const aslam::VisualNFrame::Ptr& nframe,//多相机系统
    const vio::ViNodeState& vinode_state)//pose, velocity and bias
{
    CHECK_EQ(camera_rig_.get(), nframe->getNCameraShared().get())
        << "Can only add nframes that correspond to the camera rig set in the "
        << "mission.";

    // Initialize all keypoint <-> landmark associations to invalid.
    std::vector<vi_map::LandmarkIdList> invalid_landmark_ids;//应该是存储的观测信息
    const size_t num_frames = nframe->getNumFrames();//几个相机
    for (size_t frame_idx = 0; frame_idx < num_frames; ++frame_idx)//遍历每个相机
    {
        if (nframe->isFrameSet(frame_idx))//检查一下这个相机的合法性
        {
            const aslam::VisualFrame& frame = nframe->getFrame(frame_idx);//得到当前相机的帧
            invalid_landmark_ids.emplace_back(frame.getNumKeypointMeasurements());//只是给每个vector初始化一个size
        }
    }

    // Create and add the new map vertex.
    pose_graph::VertexId vertex_id =//随机给了一个不重复的节点id
            common::createRandomId<pose_graph::VertexId>();
    //初始化节点
    //需要输入节点id，imubias(加速度计，陀螺仪)，相机个数，路标点集合，路标点集合
    vi_map::Vertex* map_vertex = new vi_map::Vertex(
            vertex_id, vinode_state.getImuBias(), nframe, invalid_landmark_ids,
            mission_id_);
    // Set pose and velocity.
    map_vertex->set_T_M_I(vinode_state.get_T_M_I());//设置当前body坐标系在任务坐标系中的位姿
    map_vertex->set_v_M(vinode_state.get_v_M_I());//当前的速度
    map_->addVertex(vi_map::Vertex::UniquePtr(map_vertex));//向地图中添加节点（实际是向位姿图中添加）

    // Optionally dump the image to disk.
    if (FLAGS_map_builder_save_image_as_resources)
    {
        CHECK(map_->hasMapFolder())
        << "Cannot store resources to a map that has no associated map folder, "
        << "please set the map folder in the VIMap constructor or by using "
        << "map.setMapFolder()!";
        map_->useMapResourceFolder();
        for (size_t frame_idx = 0u; frame_idx < nframe->getNumFrames();
             ++frame_idx) {
            if (nframe->isFrameSet(frame_idx)) {
                map_->storeRawImage(
                        nframe->getFrame(frame_idx).getRawImage(), frame_idx, map_vertex);
            }
        }
    }
    return vertex_id;
}

void StreamMapBuilder::removeAllVerticesAfterVertexId(
    const pose_graph::VertexId& vertex_id_from,
    pose_graph::VertexIdList* removed_vertex_ids) {
  CHECK_NOTNULL(removed_vertex_ids);
  manipulation_.removePosegraphAfter(vertex_id_from, removed_vertex_ids);
  last_vertex_ = vertex_id_from;
}

void StreamMapBuilder::addImuEdge(
    pose_graph::VertexId target_vertex_id,//新的节点
    const Eigen::Matrix<int64_t, 1, Eigen::Dynamic>& imu_timestamps,//这两个节点之间的imu测量
    const Eigen::Matrix<double, 6, Eigen::Dynamic>& imu_measurements)
{
    CHECK(last_vertex_.isValid());//和上一个节点去连接
    CHECK(target_vertex_id.isValid());
    CHECK_EQ(imu_timestamps.cols(), imu_measurements.cols());

    // Make sure the vertices are not yet connected.
    pose_graph::EdgeIdSet edges;
    constMap()->getVertex(last_vertex_).getOutgoingEdges(&edges);//得到之前节点所有出边，事实上应该是没有这个类型的出边的
    for (const pose_graph::EdgeId& outgoing_edge_id : edges)
    {
        CHECK(
                constMap()->getEdgeType(outgoing_edge_id) !=
                pose_graph::Edge::EdgeType::kViwls)
        << "Source vertex already has an outgoing ViwlsEdge.";
    }
    constMap()->getVertex(target_vertex_id).getIncomingEdges(&edges);
    for (const pose_graph::EdgeId& incoming_edge_id : edges) //得到现在节点的所有入边，事实上是没有这个类型的入边的
    {
        CHECK(
                constMap()->getEdgeType(incoming_edge_id) !=
                pose_graph::Edge::EdgeType::kViwls)
        << "Source vertex already has an incoming ViwlsEdge.";
    }

    // The vi_map data structures requires that the last IMU measurement of
    // the previous edge to be be duplicated as the first measurement of the
    // following edge.
    // vi_map数据结构要求将前一条边的最后一次IMU测量复制为下一条边的第一次测量，保证连续性
    pose_graph::EdgeIdList source_vertex_edge_incoming_ids;
    constMap()->getIncomingOfType(
            pose_graph::Edge::EdgeType::kViwls, last_vertex_,
            &source_vertex_edge_incoming_ids);
    //应该只会有一条边,source_vertex_edge_incoming_ids实际里只有一个id
    CHECK_LE(source_vertex_edge_incoming_ids.size(), 1u)
        << "More than one ViwlsEdge found.";

    const bool first_edge = source_vertex_edge_incoming_ids.empty();//这种情况只会发生在根节点身上，因为根节点是没有入边的
    if (!first_edge) //如果不是根节点，还需要检查一下imu数据连续性的问题
    {
        const vi_map::ViwlsEdge& prev_edge = map_->getEdgeAs<vi_map::ViwlsEdge>(//得到这个vio边
                source_vertex_edge_incoming_ids.front());
        const double kTolerance = 1.0e-12;
        const std::string kErrorMsg(
                "The vi_map data structures requires that the last IMU"
                "measurement of the previous edge to be be duplicated as the first "
                "measurement of the following edge.");
        CHECK_LT(//这里是在检查前一个节点的入边的最后一个imu数据应该就是当前这个边的第一个数据
                (prev_edge.getImuTimestamps().tail<1>() - imu_timestamps.head(1))
                        .cwiseAbs()
                        .maxCoeff() +
                (prev_edge.getImuData().rightCols<1>() -
                 imu_measurements.leftCols(1))
                        .cwiseAbs()
                        .maxCoeff(),
                kTolerance)
            << kErrorMsg;
    }

    // Add the edge.
    pose_graph::EdgeId edge_id = common::createRandomId<pose_graph::EdgeId>();//产生一个随机的边id
    map_->addEdge(//增加边，输入的是边的id，从哪个节点出来的，入哪个节点，这两个节点之间的imu数据
            aligned_unique<vi_map::ViwlsEdge>(
                    edge_id, last_vertex_, target_vertex_id, imu_timestamps,
                    imu_measurements));

    last_vertex_ = target_vertex_id;//当前节点变成上一个节点
}

bool StreamMapBuilder::checkConsistency() const {
  return vi_map::checkMapConsistency(*CHECK_NOTNULL(constMap()));
}

}  // namespace online_map_builders
