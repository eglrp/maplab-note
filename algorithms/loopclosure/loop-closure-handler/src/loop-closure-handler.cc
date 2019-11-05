#include "loop-closure-handler/loop-closure-handler.h"

#include <aslam/common/statistics/statistics.h>
#include <aslam/geometric-vision/pnp-pose-estimator.h>
#include <vi-map-helpers/vi-map-queries.h>
#include <vi-map/landmark-quality-metrics.h>

#include "loop-closure-handler/inlier-index-with-reprojection-error.h"

DEFINE_double(lc_ransac_pixel_sigma, 2.0, "Pixel sigma for ransac.");
DEFINE_int32(lc_min_inlier_count, 10, "Minimum inlier count for loop closure.");
DEFINE_double(
    lc_min_inlier_ratio, 0.2, "Minimum inlier ratio for loop closure.");
DECLARE_bool(lc_filter_underconstrained_landmarks);
DEFINE_int32(
    lc_num_ransac_iters, 100,
    "Maximum number of ransac iterations for absolute pose recovery.");
DEFINE_bool(
    lc_nonlinear_refinement_p3p, false,
    "If nonlinear refinement on all ransac inliers should be run.");
DECLARE_double(lc_switch_variable_variance);

DEFINE_double(
    lc_edge_covariance_scaler, 1e-7,
    "Scaling the covariance of loopclosure edges. It is identity by default.");
DEFINE_double(
    lc_edge_min_distance_meters, 1.0,
    "The minimum loop-closure gap distance such that a loop-closure edge is "
    "created.");
DEFINE_double(
    lc_edge_min_inlier_ratio, 0.5,
    "The minimum loop-closure inlier ratio to add a loop-closure edge.");
DEFINE_int32(
    lc_edge_min_inlier_count, 20,
    "The minimum loop-closure inlier count to add a loop-closure edge.");

namespace loop_closure_handler {

bool addLoopClosureEdge(
    const pose_graph::VertexId& query_vertex_id,
    const vi_map::LandmarkIdSet& commonly_observed_landmarks,
    const pose_graph::VertexId& vertex_id_from_structure_matches,
    const aslam::Transformation& T_G_I_lc_ransac, vi_map::VIMap* map) {
  CHECK(query_vertex_id.isValid());
  CHECK_NOTNULL(map)->hasVertex(vertex_id_from_structure_matches);
  CHECK(vertex_id_from_structure_matches != query_vertex_id);

  Eigen::Matrix2Xd measurements;
  std::vector<int> measurement_camera_indices;
  Eigen::Matrix3Xd G_landmark_positions;

  const vi_map::Vertex& vertex =
      map->getVertex(vertex_id_from_structure_matches);

  // Retrieve all observed landmarks to get an estimate of the maximum number
  // of correspondences we may find.
  vi_map::LandmarkIdList observed_landmark_ids;
  vertex.getAllObservedLandmarkIds(&observed_landmark_ids);

  // Resize the containers to the maximum possible size.
  measurements.resize(Eigen::NoChange, observed_landmark_ids.size());
  G_landmark_positions.resize(Eigen::NoChange, observed_landmark_ids.size());

  int index = 0;
  const size_t num_frames = vertex.numFrames();
  for (size_t frame_idx = 0u; frame_idx < num_frames; ++frame_idx) {
    if (!vertex.isVisualFrameSet(frame_idx) ||
        !vertex.isVisualFrameValid(frame_idx)) {
      continue;
    }

    const aslam::VisualFrame& visual_frame = vertex.getVisualFrame(frame_idx);
    const vi_map::LandmarkIdList& observed_landmarks =
        vertex.getFrameObservedLandmarkIds(frame_idx);
    CHECK_EQ(
        observed_landmarks.size(), visual_frame.getNumKeypointMeasurements());
    for (size_t i = 0u; i < observed_landmarks.size(); ++i) {
      if (commonly_observed_landmarks.count(observed_landmarks[i]) > 0u) {
        CHECK(observed_landmarks[i].isValid());

        CHECK_LT(index, measurements.cols());
        CHECK_LT(index, G_landmark_positions.cols());

        measurement_camera_indices.push_back(frame_idx);
        measurements.col(index) = visual_frame.getKeypointMeasurement(i);
        G_landmark_positions.col(index) =
            map->getLandmark_G_p_fi(observed_landmarks[i]);

        ++index;
      }
    }
  }

  // Resize the containers to the actual data size.
  measurements.conservativeResize(Eigen::NoChange, index);
  G_landmark_positions.conservativeResize(Eigen::NoChange, index);

  CHECK_EQ(
      measurements.cols(), static_cast<int>(measurement_camera_indices.size()));
  CHECK_EQ(measurements.cols(), G_landmark_positions.cols());
  CHECK_GT(measurements.cols(), 0);

  constexpr bool kUseRandomPnpSeed = true;
  aslam::geometric_vision::PnpPoseEstimator pose_estimator(//这里把优化给设成了false，为什么？
      FLAGS_lc_nonlinear_refinement_p3p, kUseRandomPnpSeed);

  aslam::NCamera::ConstPtr ncamera = vertex.getNCameras();//多相机数量
  CHECK(ncamera != nullptr);

  pose::Transformation T_G_Inn_ransac;
  std::vector<int> inliers;
  std::vector<double> inlier_distances_to_model;
  int num_iters;
  bool pnp_success = pose_estimator.absoluteMultiPoseRansacPinholeCam(
      measurements, measurement_camera_indices, G_landmark_positions,
      FLAGS_lc_ransac_pixel_sigma, FLAGS_lc_num_ransac_iters, ncamera,
      &T_G_Inn_ransac, &inliers, &inlier_distances_to_model, &num_iters);

  if (!pnp_success) {
    // We could not retrieve a pose for the vertex observing matched landmarks.
    // The LC edge cannot be added.
    return false;
  }

  const aslam::TransformationCovariance T_Inn_Iquery_covariance =
      FLAGS_lc_edge_covariance_scaler *
      aslam::TransformationCovariance::Identity();
  pose_graph::EdgeId loop_closure_edge_id;
  common::generateId(&loop_closure_edge_id);
  CHECK(loop_closure_edge_id.isValid());

  const aslam::Transformation T_Inn_G = T_G_Inn_ransac.inverse();
  const aslam::Transformation T_Inn_Iquery_lc = T_Inn_G * T_G_I_lc_ransac;

  const aslam::Transformation T_G_Iquery =
      map->getVertex_T_G_I(query_vertex_id);
  const aslam::Transformation T_Inn_Iquery_posegraph = T_Inn_G * T_G_Iquery;
  const double distance_lc_to_posegraph_meters_squared =
      (T_Inn_Iquery_lc.inverse() * T_Inn_Iquery_posegraph)
          .getPosition()
          .squaredNorm();

  if (distance_lc_to_posegraph_meters_squared <
      (FLAGS_lc_edge_min_distance_meters * FLAGS_lc_edge_min_distance_meters)) {
    VLOG(10) << "Skipping creation of loop-closure edge because the gap "
             << "distance is too small ("
             << std::sqrt(distance_lc_to_posegraph_meters_squared) << '/'
             << FLAGS_lc_edge_min_distance_meters << ").";
    return false;
  }

  const double kSwitchVariable = 1.0;
  CHECK_GT(FLAGS_lc_switch_variable_variance, 0.0);
  vi_map::Edge::UniquePtr loop_closure_edge(
      new vi_map::LoopClosureEdge(
          loop_closure_edge_id, vertex_id_from_structure_matches,
          query_vertex_id, kSwitchVariable, FLAGS_lc_switch_variable_variance,
          T_Inn_Iquery_lc, T_Inn_Iquery_covariance));

  VLOG(10) << "Added loop-closure edge between vertex "
           << query_vertex_id.hexString() << " and vertex "
           << vertex_id_from_structure_matches.hexString() << '.';

  map->addEdge(std::move(loop_closure_edge));

  return true;
}

LoopClosureHandler::LoopClosureHandler(
    vi_map::VIMap* map, LandmarkToLandmarkMap* landmark_id_old_to_new)
    : map_(CHECK_NOTNULL(map)), summary_map_(nullptr),
      landmark_id_old_to_new_(CHECK_NOTNULL(landmark_id_old_to_new)) {}

LoopClosureHandler::LoopClosureHandler(
    summary_map::LocalizationSummaryMap const* summary_map,
    LandmarkToLandmarkMap* landmark_id_old_to_new)
    : map_(nullptr), summary_map_(CHECK_NOTNULL(summary_map)),
      landmark_id_old_to_new_(CHECK_NOTNULL(landmark_id_old_to_new)) {}



// Assuming same query_keyframe in each of the constraints on the vector.
bool LoopClosureHandler::handleLoopClosure(
    const vi_map::LoopClosureConstraint& loop_closure_constraint,
    bool merge_matching_landmarks, bool add_loopclosure_edges, int* num_inliers,
    double* inlier_ratio, pose::Transformation* T_G_I_ransac,
    vi_map::LoopClosureConstraint* inlier_constraints,
    MergedLandmark3dPositionVector* landmark_pairs_merged,
    pose_graph::VertexId* vertex_id_closest_to_structure_matches,
    std::mutex* map_mutex, bool use_random_pnp_seed) const//use_random_pnp_seed默认是true
    {
        CHECK_NOTNULL(map_);
        CHECK_NOTNULL(num_inliers);
        CHECK_NOTNULL(inlier_ratio);
        CHECK_NOTNULL(T_G_I_ransac);
        CHECK_NOTNULL(inlier_constraints);
        CHECK_NOTNULL(landmark_pairs_merged);
        // Note: vertex_id_closest_to_structure_matches is optional and may be NULL.
        T_G_I_ransac->setIdentity();

        const pose_graph::VertexId& query_vertex_id =
                loop_closure_constraint.query_vertex_id;
        vi_map::Vertex& query_vertex = map_->getVertex(query_vertex_id);

        inlier_constraints->query_vertex_id = query_vertex_id;

        std::vector<vi_map::LandmarkIdList> query_vertex_observed_landmark_ids;
        query_vertex.getAllObservedLandmarkIds(&query_vertex_observed_landmark_ids);

        return handleLoopClosure(
                query_vertex.getVisualNFrame(), query_vertex_observed_landmark_ids,
                query_vertex_id, loop_closure_constraint.structure_matches,
                merge_matching_landmarks, add_loopclosure_edges, num_inliers,
                inlier_ratio, T_G_I_ransac, &inlier_constraints->structure_matches,
                landmark_pairs_merged, vertex_id_closest_to_structure_matches, map_mutex,
                use_random_pnp_seed);
    }

    //输入多相机系统，每个相机观测到的地图点的对应的id,这个是随机生成的id，还没有赋值的节点，多相机的所有匹配，是否要汇合地图点（默认是false），是否要添加闭环边（false）
    //输出内点数量，内点比例，求出的节点在世界坐标系下的位姿，?,?,?,?
bool LoopClosureHandler::handleLoopClosure(
    const aslam::VisualNFrame& query_vertex_n_frame,
    const std::vector<vi_map::LandmarkIdList>& query_vertex_landmark_ids,
    const pose_graph::VertexId& query_vertex_id,
    const vi_map::VertexKeyPointToStructureMatchList& structure_matches,
    bool merge_matching_landmarks, bool add_loopclosure_edges, int* num_inliers,
    double* inlier_ratio, pose::Transformation* T_G_I_ransac,
    vi_map::VertexKeyPointToStructureMatchList* inlier_structure_matches,
    MergedLandmark3dPositionVector* landmark_pairs_merged,
    pose_graph::VertexId* vertex_id_closest_to_structure_matches,
    std::mutex* map_mutex, bool use_random_pnp_seed) const
    {
        CHECK_NOTNULL(num_inliers);
        CHECK_NOTNULL(inlier_ratio);
        CHECK_NOTNULL(T_G_I_ransac);
        CHECK_NOTNULL(inlier_structure_matches)->clear();
        CHECK_NOTNULL(landmark_pairs_merged);
        CHECK_NOTNULL(map_mutex);
        // Note: vertex_id_closest_to_structure_matches is optional and may be NULL.
        T_G_I_ransac->setIdentity();

        CHECK_EQ(
                static_cast<unsigned int>(query_vertex_n_frame.getNumFrames()),
                query_vertex_landmark_ids.size());//多相机的数量应该是一致的

        // Make sure only one of those options is selected. We can't merge landmarks
        // and add loopclosure edges at the same time.//确保只选择其中一个选项。我们不能同时合并界标和添加环闭边缘。
        CHECK(!merge_matching_landmarks || !add_loopclosure_edges);

        *num_inliers = 0;
        *inlier_ratio = 0.0;

        statistics::StatsCollector stats_total_calls(
                "0.0 Loop closure: Total query frames handled");
        stats_total_calls.IncrementOne();

        // Bail for cases where there is no hope to reach the min num inliers.
        int total_matches = structure_matches.size();//所有相机总的匹配的数量
        if (total_matches < FLAGS_lc_min_inlier_count) {//最小不应该小于10个
            return false;
        }

        Eigen::Matrix2Xd measurements;
        std::vector<int> measurement_camera_indices;
        Eigen::Matrix3Xd G_landmark_positions;
        measurements.resize(Eigen::NoChange, total_matches);
        G_landmark_positions.resize(Eigen::NoChange, total_matches);
        measurement_camera_indices.resize(total_matches);


//        typedef std::vector<
//                std::pair<FrameKeypointIndexPair, vi_map::LandmarkId>>
//                KeypointToLandmarkVector;
//
        // Ordered containers s.t. inliers vector returned from P3P makes sense.
        KeypointToLandmarkVector query_keypoint_idx_to_map_landmark_pairs;//存储的是vec,里面的每一个元素，key是特征点的索引，value是地图点的id，也就是2d-3d匹配
        LandmarkToLandmarkVector query_landmark_to_map_landmark_pairs;//存储自己的3d点索引到以前的地图点索引的对应

        query_keypoint_idx_to_map_landmark_pairs.resize(total_matches);
        query_landmark_to_map_landmark_pairs.resize(total_matches);

        int col_idx = 0;

        //for debug

        for (const vi_map::VertexKeyPointToStructureMatch& structure_match :
                structure_matches) //遍历所有相机中的匹配信息
        {
            //现在都是针对某一个匹配而言
            map_mutex->lock();
            //如果要汇合地图点的话，可能当前闭环检测到的地图点已经变成了别的地图点。这里返回的是汇合以后的地图点的id，但是当不汇合时，这个就是原来的地图点的id
            vi_map::LandmarkId db_landmark_id = getLandmarkIdAfterMerges(
                    structure_match.landmark_result);

            // The loop-closure backend should not return invalid landmarks.
            CHECK(db_landmark_id.isValid())
            << "Found invalid landmark in result set.";

            // The loop-closure backend should not return underconstrained landmarks.
            // if the lc_filter_underconstrained_landmarks flag is true.
            if (map_ != nullptr)
            {
                // This check makes only sense if we have a full map available.,这里
                if (FLAGS_lc_filter_underconstrained_landmarks &&
                    !vi_map::isLandmarkWellConstrained(
                            *map_, map_->getLandmark(db_landmark_id)))
                {
                    LOG(FATAL)
                            << "Found not well constrained landmark in result set. Cached "
                            << "quality: "
                            << static_cast<int>(
                                    map_->getLandmark(db_landmark_id).getQuality())
                            << ",  evaluated quality: "
                            << vi_map::isLandmarkWellConstrained(
                                    *map_, map_->getLandmark(db_landmark_id))
                            << ", landmark id: " << structure_match.landmark_result.hexString();
                }
            }

            CHECK_LT(
                    col_idx,
                    static_cast<int>(query_keypoint_idx_to_map_landmark_pairs.size()));
            CHECK(query_vertex_n_frame.isFrameSet(structure_match.frame_index_query));
            measurements.col(col_idx) =
                    query_vertex_n_frame.getFrame(structure_match.frame_index_query)
                            .getKeypointMeasurement(structure_match.keypoint_index_query);//正在寻找匹配的这个相机中的这个特征点的id
            G_landmark_positions.col(col_idx) = getLandmark_p_G_fi(db_landmark_id);//得到它所匹配的3d坐标，也就是2d-3d匹配
            map_mutex->unlock();

            // Set the frame correspondence to the correct frame for multi-camera
            // systems. We do this for the single-camera case as well.
            // 为多相机系统设置正确的帧对应。我们对单摄像头的情况也是这样做的。
            measurement_camera_indices[col_idx] = structure_match.frame_index_query;//存储的是看到这个特征点的相机id
            std::string camera_stats_string =
                    "LC camera index " + std::to_string(structure_match.frame_index_query);
            statistics::StatsCollector stats(camera_stats_string);
            stats.IncrementOne();


//            struct FrameKeypointIndexPair {
//                unsigned int frame_idx;
//                unsigned int keypoint_idx;
//
//                FrameKeypointIndexPair() : frame_idx(-1), keypoint_idx(-1) {}
//                FrameKeypointIndexPair(unsigned int _frame_idx, unsigned int _keypoint_idx)
//                        : frame_idx(_frame_idx), keypoint_idx(_keypoint_idx) {}
//
//                inline bool operator==(const FrameKeypointIndexPair& lhs) const {
//                    bool is_same = true;
//                    is_same &= frame_idx == lhs.frame_idx;
//                    is_same &= keypoint_idx == lhs.keypoint_idx;
//                    return is_same;
//                }
//            };存储观测点的索引

            FrameKeypointIndexPair query_observation(
                    structure_match.frame_index_query,;//看到这个特征点的相机id
                    structure_match.keypoint_index_query);//正在寻找匹配的这个相机中的这个特征点的id
            query_keypoint_idx_to_map_landmark_pairs[col_idx] =
                    std::make_pair(query_observation, db_landmark_id);//存入2d-3d匹配信息，每一个2d点都会有一个3d匹配。col_idx就是点的索引

            CHECK_LT(
                    structure_match.keypoint_index_query,
                    query_vertex_landmark_ids[structure_match.frame_index_query]
                            .size());
            vi_map::LandmarkId query_landmark_id =//找到当前这个特征点对应的自己的地图点id，这个就是随机生成的
                    query_vertex_landmark_ids[structure_match.frame_index_query]
                    [structure_match.keypoint_index_query];
            query_landmark_to_map_landmark_pairs[col_idx] =//存储自己的3d点和它重定位检测到的3d点的索引
                    std::make_pair(query_landmark_id, db_landmark_id);

            ++col_idx;
        }
        // Bail for cases where there is no hope to reach the min num inliers.
        int valid_matches = col_idx;
        statistics::StatsCollector valid_matches_stats("LC num valid matches");
        valid_matches_stats.AddSample(valid_matches);
        if (valid_matches < FLAGS_lc_min_inlier_count) {//因为剔除了一部分不好的地图点，所以还会再去判断一下地图点的数量
            VLOG(2) << "Bailing out because too few inliers. (#valid matches: "
                    << valid_matches
                    << " vs. min_inlier_count: " << FLAGS_lc_min_inlier_count << ")";
            statistics::StatsCollector stats("LC bailed because too few inliers.");
            stats.IncrementOne();
            return false;
        }

        measurements.conservativeResize(Eigen::NoChange, col_idx);
        G_landmark_positions.conservativeResize(Eigen::NoChange, col_idx);
        query_keypoint_idx_to_map_landmark_pairs.resize(col_idx);
        query_landmark_to_map_landmark_pairs.resize(col_idx);
        measurement_camera_indices.resize(col_idx);

        aslam::geometric_vision::PnpPoseEstimator pose_estimator(//这里不去非线性优化p3p，这是为什么
                FLAGS_lc_nonlinear_refinement_p3p, use_random_pnp_seed);
        std::vector<int> inliers;
        std::vector<double> inlier_distances_to_model;
        int num_iters;

        aslam::NCamera::ConstPtr ncamera = query_vertex_n_frame.getNCameraShared();//得到多相机系统
        CHECK(ncamera != nullptr);
        //输入测量值，观测到这个点的是哪一个相机，这个观测点对应的地图点的坐标，像素方差的阈值，FLAGS_lc_num_ransac_iters的迭代次数，多相机系统，
        //输出位姿求解结果，内点索引，内点到模型的距离，迭代次数
        pose_estimator.absoluteMultiPoseRansacPinholeCam(
                measurements, measurement_camera_indices, G_landmark_positions,
                FLAGS_lc_ransac_pixel_sigma, FLAGS_lc_num_ransac_iters, ncamera,
                T_G_I_ransac, &inliers, &inlier_distances_to_model, &num_iters);
        CHECK_EQ(inliers.size(), inlier_distances_to_model.size());

        KeypointToInlierIndexWithReprojectionErrorMap
                keypoint_to_best_structure_match;
        //输入内点索引,内点到模型的距离，所有相机的匹配，n相机系统
        getBestStructureMatchForEveryKeypoint(
                inliers, inlier_distances_to_model, structure_matches,
                query_vertex_n_frame, &keypoint_to_best_structure_match);

        CHECK_LE(keypoint_to_best_structure_match.size(), inliers.size());
        *num_inliers = static_cast<int>(keypoint_to_best_structure_match.size());

        VLOG(3) << "\tnum_inliers " << *num_inliers << " num iters " << num_iters;
        statistics::StatsCollector stats_inlier_count("LC RANSAC inliers");
        stats_inlier_count.AddSample(*num_inliers);

        if (*num_inliers < FLAGS_lc_min_inlier_count) {
            statistics::StatsCollector stats("LC too few RANSAC inliers");
            stats.IncrementOne();

            return false;
        }

        statistics::StatsCollector stats(
                "LC passed RANSAC inlier threshold (lc_min_inlier_count)");
        stats.IncrementOne();

        CHECK_GT(G_landmark_positions.cols(), 0);
        *inlier_ratio = static_cast<double>(*num_inliers) /
                        static_cast<double>(G_landmark_positions.cols());
        VLOG(4) << "\tinlier_ratio " << *inlier_ratio;

        statistics::StatsCollector stats_inlier_ratio("LC RANSAC inlier ratio");
        stats_inlier_ratio.AddSample(*inlier_ratio);

        if (*inlier_ratio < FLAGS_lc_min_inlier_ratio) {
            statistics::StatsCollector statistics_ransac_fail_inlier_ratio(
                    "LC ransac fail inlier_ratio");
            statistics_ransac_fail_inlier_ratio.AddSample(*inlier_ratio);
            statistics::StatsCollector statistics_ransac_fail_num_inliers(
                    "LC ransac fail num_inliers");
            statistics_ransac_fail_num_inliers.AddSample(*num_inliers);
            return false;
        }

        Eigen::Matrix3Xd landmark_positions;
        landmark_positions.resize(Eigen::NoChange, *num_inliers);
        inlier_structure_matches->resize(static_cast<size_t>(*num_inliers));

        int inlier_sequential_idx = 0;
        for (const KeypointToInlierIndexWithReprojectionErrorMap::value_type&
                    keypoint_identifier_with_inlier_index :
                keypoint_to_best_structure_match) {
            const int inlier_index =
                    keypoint_identifier_with_inlier_index.second.getInlierIndex();
            CHECK_GE(inlier_index, 0);
            CHECK_LT(inlier_index, G_landmark_positions.cols());
            landmark_positions.block<3, 1>(0, inlier_sequential_idx) =
                    G_landmark_positions.block<3, 1>(0, inlier_index);
            (*inlier_structure_matches)[inlier_sequential_idx] =
                    structure_matches[inlier_index];
            ++inlier_sequential_idx;
        }

        statistics::StatsCollector statistics_ransac_success_inlier_ratio(
                "LC ransac success inlier_ratio");
        statistics_ransac_success_inlier_ratio.AddSample(*inlier_ratio);
        statistics::StatsCollector statistics_ransac_success_num_inliers(
                "LC ransac success num_inliers");
        statistics_ransac_success_num_inliers.AddSample(*num_inliers);
        VLOG(10) << "Found loop-closure for query vertex "
                 << query_vertex_id.hexString();

        if (merge_matching_landmarks) {
            CHECK_NOTNULL(map_);
            std::lock_guard<std::mutex> map_lock(*map_mutex);

            // This case should be only handled if a valid query_vertex_id is
            // provided.
            CHECK(query_vertex_id.isValid())
            << "Merging landmark is not possible "
            << "if no valid query_vertex_id is provided.";

            // Also reassociates keypoints of the query frame.
            mergeLandmarks(
                    inliers, query_landmark_to_map_landmark_pairs, landmark_pairs_merged);

            // Some of the query frame keypoints may have invalid landmark ids
            // (which means the landmark object don't exist right now), but they
            // were matched to an existing map landmark. We should handle that
            // separately, as it's not true landmark merge.
            vi_map::Vertex& query_vertex = map_->getVertex(query_vertex_id);
            updateQueryKeyframeInvalidLandmarkAssociations(
                    inliers, query_keypoint_idx_to_map_landmark_pairs, &query_vertex);
        }
        vi_map::LandmarkIdSet commonly_observed_landmarks;
        if (vertex_id_closest_to_structure_matches != nullptr) {
            CHECK(!merge_matching_landmarks)
            << "Retrieving the vertex id closest to "
            << "the structure-matches does not work if landmarks are being merged "
            << "too.";
            *vertex_id_closest_to_structure_matches =
                    vi_map_helpers::getVertexIdWithMostOverlappingLandmarks(
                            query_vertex_id, *inlier_structure_matches, *map_,
                            &commonly_observed_landmarks);
            CHECK(vertex_id_closest_to_structure_matches->isValid());
        }
        if (add_loopclosure_edges) {
            CHECK(!merge_matching_landmarks);
            if (query_vertex_id.isValid() && map_ != nullptr) {
                if (*inlier_ratio >= FLAGS_lc_edge_min_inlier_ratio &&
                    *num_inliers >= FLAGS_lc_edge_min_inlier_count) {
                    pose_graph::VertexId lc_edge_target_vertex_id;
                    if (vertex_id_closest_to_structure_matches == nullptr) {
                        CHECK(commonly_observed_landmarks.empty());
                        lc_edge_target_vertex_id =
                                vi_map_helpers::getVertexIdWithMostOverlappingLandmarks(
                                        query_vertex_id, *inlier_structure_matches, *map_,
                                        &commonly_observed_landmarks);
                    } else {
                        // vertex_id_closest_to_structure_matches was already retrieved
                        // before.
                        lc_edge_target_vertex_id = *vertex_id_closest_to_structure_matches;
                    }
                    CHECK(lc_edge_target_vertex_id.isValid());
                    CHECK(!commonly_observed_landmarks.empty());
                    std::lock_guard<std::mutex> map_lock(*map_mutex);
                    addLoopClosureEdge(
                            query_vertex_id, commonly_observed_landmarks,
                            lc_edge_target_vertex_id, *T_G_I_ransac, map_);
                }
            }
        }

        VLOG(4) << "\transac success. Ransac pts: " << G_landmark_positions.cols()
                << " inliers: " << inliers.size()
                << " inlier ratio: " << *inlier_ratio << '.';
        return true;
    }

void LoopClosureHandler::updateQueryKeyframeInvalidLandmarkAssociations(
    const std::vector<int>& inliers,
    const KeypointToLandmarkVector& query_keypoint_idx_to_landmark_pairs,
    vi_map::Vertex* query_vertex) const {
  CHECK_NOTNULL(query_vertex);

  statistics::StatsCollector stats_reassociations(
      "0.3.2 Loop closure: Invalid query landmarks, reassociated");

  for (unsigned int i = 0; i < inliers.size(); ++i) {
    const int query_frame_idx =
        query_keypoint_idx_to_landmark_pairs[inliers[i]].first.frame_idx;
    const int query_keypoint_idx =
        query_keypoint_idx_to_landmark_pairs[inliers[i]].first.keypoint_idx;
    const vi_map::LandmarkId query_landmark_id =
        query_vertex->getObservedLandmarkId(
            query_frame_idx, query_keypoint_idx);
    vi_map::LandmarkId map_landmark_id =
        query_keypoint_idx_to_landmark_pairs[inliers[i]].second;

    if (!query_landmark_id.isValid()) {
      stats_reassociations.IncrementOne();

      // The map landmark could have been merged and removed already. Check
      // if we should not replace the id to the newer one.
      map_landmark_id = getLandmarkIdAfterMerges(map_landmark_id);

      query_vertex->setObservedLandmarkId(
          query_frame_idx, query_keypoint_idx, map_landmark_id);

      vi_map::Vertex& map_landmark_vertex =
          map_->getLandmarkStoreVertex(map_landmark_id);

      map_landmark_vertex.getLandmarks()
          .getLandmark(map_landmark_id).addObservation(
              query_vertex->id(), query_frame_idx, query_keypoint_idx);
    }
  }
}

void LoopClosureHandler::mergeLandmarks(
    const std::vector<int>& inliers,
    const LandmarkToLandmarkVector& query_landmark_to_map_landmark_pairs,
    MergedLandmark3dPositionVector* landmark_pairs_actually_merged) const {
  CHECK_NOTNULL(landmark_pairs_actually_merged);

  statistics::StatsCollector stats_p3p_inliers(
      "0.1 Loop closure: total P3P inliers");
  statistics::StatsCollector stats_same_ids(
      "0.2 Loop closure: Same query/map landmark IDs");
  statistics::StatsCollector stats_valid_query_landmark(
      "0.3.1 Loop closure: Valid query landmarks");
  statistics::StatsCollector stats_total_merge_calls(
      "0.4 Loop closure: Total merge calls");

  for (unsigned int i = 0; i < inliers.size(); ++i) {
    vi_map::LandmarkId query_landmark_to_be_deleted =
        query_landmark_to_map_landmark_pairs[inliers[i]].first;
    vi_map::LandmarkId map_landmark =
        query_landmark_to_map_landmark_pairs[inliers[i]].second;
    CHECK(map_landmark.isValid());
    stats_p3p_inliers.IncrementOne();

    // Perform the full merge only if the query landmark exists. Otherwise
    // we will only update the reference in vertex using
    // updateQueryKeyframeInvalidLandmarkAssociations method.
    if (query_landmark_to_be_deleted.isValid()) {
      CHECK(map_landmark.isValid());

      stats_valid_query_landmark.IncrementOne();

      // If we need to merge landmark into itself, we skip merging.
      if (query_landmark_to_be_deleted == map_landmark) {
        stats_same_ids.IncrementOne();
        continue;
      }

      // Check if this landmark has already been merged with a new one.
      LandmarkToLandmarkMap::iterator it_landmark_already_changed =
          landmark_id_old_to_new_->find(query_landmark_to_be_deleted);
      if (it_landmark_already_changed != landmark_id_old_to_new_->end()) {
        // Thats all we need to do, the vertices etc. have already been
        // updated before using the back-references.
        continue;
      }

      // Also verify if the map landmark was not changed before in this
      // call to mergeLandmarks.
      map_landmark = getLandmarkIdAfterMerges(map_landmark);

      // If we need to merge landmark into itself, we skip merging. Do this
      // for the 2nd time because the assignment above may cause this situation
      // to arise.
      if (query_landmark_to_be_deleted == map_landmark) {
        stats_same_ids.IncrementOne();
        continue;
      }
      CHECK_NE(query_landmark_to_be_deleted, map_landmark);

      // Store the changed id in the local map.
      (*landmark_id_old_to_new_)[query_landmark_to_be_deleted] = map_landmark;

      Eigen::Vector3d p_G_landmark_query =
          map_->getLandmark_G_p_fi(query_landmark_to_be_deleted);
      Eigen::Vector3d p_G_landmark_map =
          map_->getLandmark_G_p_fi(map_landmark);

      stats_total_merge_calls.IncrementOne();
      map_->mergeLandmarks(query_landmark_to_be_deleted, map_landmark);

      landmark_pairs_actually_merged->emplace_back(
          p_G_landmark_query, p_G_landmark_map);
    }
  }
}

}  // namespace loop_closure_handler
