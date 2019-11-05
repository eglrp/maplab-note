#include "feature-tracking/vo-feature-tracking-pipeline.h"

#include <aslam/common/timer.h>
#include <aslam/geometric-vision/match-outlier-rejection-twopt.h>
#include <aslam/tracker/feature-tracker-gyro.h>
#include <aslam/visualization/basic-visualization.h>
#include <maplab-common/conversions.h>
#include <visualization/common-rviz-visualization.h>

DEFINE_double(
    swe_feature_tracker_two_pt_ransac_threshold, 1.0 - cos(0.5 * kDegToRad),
    "Threshold for the 2-pt RANSAC used for feature tracking outlier "
    "removal. The error is defined as (1 - cos(alpha)) where alpha is "
    "the angle between the predicted and measured bearing vectors.");

DEFINE_double(
    swe_feature_tracker_two_pt_ransac_max_iterations, 200,
    "Max iterations for the 2-pt RANSAC used for feature tracking "
    "outlier removal.");

DEFINE_bool(
    swe_feature_tracker_deterministic, false,
    "If true, deterministic RANSAC outlier rejection is used.");//如果为真，则使用确定性的RANSAC离群拒绝。
DEFINE_bool(
    detection_visualize_keypoints, false,
    "Visualize the raw keypoint detections to a ros topic.");

namespace feature_tracking {

void VOFeatureTrackingPipeline::trackFeaturesNFrame(
    const aslam::Transformation& T_Bk_Bkp1, aslam::VisualNFrame* nframe_k,
    aslam::VisualNFrame* nframe_kp1) {
  CHECK_NOTNULL(nframe_kp1);
  CHECK_NOTNULL(nframe_k);
  CHECK(nframe_k->getNCameraShared());
  CHECK(nframe_kp1->getNCameraShared());
  CHECK_EQ(nframe_k->getNCamera().getId(), nframe_kp1->getNCamera().getId());
  CHECK_EQ(
      nframe_k->getNCameraShared().get(), nframe_kp1->getNCameraShared().get());
  aslam::FrameToFrameMatchesList inlier_matches_kp1_k;
  aslam::FrameToFrameMatchesList outlier_matches_kp1_k;
  trackFeaturesNFrame(
      T_Bk_Bkp1.getRotation().inverse(), nframe_kp1, nframe_k,
      &inlier_matches_kp1_k, &outlier_matches_kp1_k);
}


//n相机追踪
    //输入这段时间内的旋转,传入的旋转是qbkb1，当前的多相机状态，之前的多相机状态
    //输出。。。
void VOFeatureTrackingPipeline::trackFeaturesNFrame(
    const aslam::Quaternion& q_Bkp1_Bk, aslam::VisualNFrame* nframe_kp1,
    aslam::VisualNFrame* nframe_k,
    aslam::FrameToFrameMatchesList* inlier_matches_kp1_k,
    aslam::FrameToFrameMatchesList* outlier_matches_kp1_k)
{
    CHECK_NOTNULL(nframe_kp1);
    CHECK_NOTNULL(nframe_k);
    CHECK_NOTNULL(inlier_matches_kp1_k)->clear();
    CHECK_NOTNULL(outlier_matches_kp1_k)->clear();
    CHECK_GT(
            nframe_kp1->getMinTimestampNanoseconds(),
            nframe_k->getMinTimestampNanoseconds());
    timing::Timer timer_eval("SweFeatureTracking::trackFeaturesNFrame");
//相机数量
    const size_t num_cameras = nframe_kp1->getNumCameras();
    CHECK_EQ(num_cameras, trackers_.size());
    CHECK_EQ(num_cameras, track_managers_.size());
    CHECK(ncamera_.get() == nframe_kp1->getNCameraShared().get());

    // Track features for each camera in its own thread.
    inlier_matches_kp1_k->resize(num_cameras);//内点
    outlier_matches_kp1_k->resize(num_cameras);//外点

    CHECK(thread_pool_);
    for (size_t camera_idx = 0u; camera_idx < num_cameras; ++camera_idx)
    {//遍历所有相机
        aslam::VisualFrame* frame_kp1 =//当前的这个相机
                nframe_kp1->getFrameShared(camera_idx).get();
        aslam::VisualFrame* frame_k = nframe_k->getFrameShared(camera_idx).get();//前一时刻的这个相机
        //输入的是四元数旋转qbkb1，相机id，当前相机，前一时刻的相机
        thread_pool_->enqueue(
                &VOFeatureTrackingPipeline::trackFeaturesSingleCamera, this, q_Bkp1_Bk,
                camera_idx, frame_kp1, frame_k, &(*inlier_matches_kp1_k)[camera_idx],
                &(*outlier_matches_kp1_k)[camera_idx]);
    }
    thread_pool_->waitForEmptyQueue();

    timer_eval.Stop();
}
    //输入的是四元数旋转qbkb1，相机id，当前相机，前一时刻的相机
void VOFeatureTrackingPipeline::trackFeaturesSingleCamera(
    const aslam::Quaternion& q_Bkp1_Bk, const size_t camera_idx,
    aslam::VisualFrame* frame_kp1, aslam::VisualFrame* frame_k,
    aslam::FrameToFrameMatches* inlier_matches_kp1_k,
    aslam::FrameToFrameMatches* outlier_matches_kp1_k)
{
    timing::Timer timer("swe-feature-tracker: trackFeaturesSingleCamera");
    CHECK_LE(camera_idx, track_managers_.size());
    CHECK_NOTNULL(frame_k);
    CHECK_NOTNULL(frame_kp1);
    CHECK_NOTNULL(inlier_matches_kp1_k);
    CHECK_NOTNULL(outlier_matches_kp1_k);
    inlier_matches_kp1_k->clear();
    outlier_matches_kp1_k->clear();

    // Initialize keypoints and descriptors in frame_k, if there aren't any.
    if (!has_feature_extraction_been_performed_on_first_nframe_) //就是首个状态的特征是在第二次状态来的时候提取的
    {
        detectors_extractors_[camera_idx]->detectAndExtractFeatures(frame_k);//对首个状态的这个相机提取特征还有描述子等
        has_feature_extraction_been_performed_on_first_nframe_ = true;//对首个状态的这个相机提取特征还有描述子等
    }
    detectors_extractors_[camera_idx]->detectAndExtractFeatures(frame_kp1);//对当前时刻的这个相机提取特征点和描述子等

    if (FLAGS_detection_visualize_keypoints)
    {//如果要发出的话，默认不发出
        cv::Mat image;
        cv::cvtColor(frame_kp1->getRawImage(), image, cv::COLOR_GRAY2BGR);

        aslam_cv_visualization::drawKeypoints(*CHECK_NOTNULL(frame_kp1), &image);
        const std::string topic = feature_tracking_ros_base_topic_ +
                                  "/keypoints_raw_cam" + std::to_string(camera_idx);
        visualization::RVizVisualizationSink::publish(topic, image);
    }

    CHECK(frame_k->hasKeypointMeasurements());
    CHECK(frame_k->hasDescriptors());
    CHECK(frame_kp1->hasKeypointMeasurements());
    CHECK(frame_kp1->hasDescriptors());

    // Get the relative motion of the camera using the extrinsics of the camera
    // system.
    const aslam::Quaternion& q_C_B =
            ncamera_->get_T_C_B(camera_idx).getRotation();
    aslam::Quaternion q_Ckp1_Ck = q_C_B * q_Bkp1_Bk * q_C_B.inverse();//转到了相机坐标系下

    statistics::StatsCollector stat_tracking("keypoint tracking (1 image) in ms");
    timing::Timer timer_tracking("descriptor matching");
    aslam::FrameToFrameMatchesWithScore matches_with_score_kp1_k;

    //输入qckc1,前一帧，后一帧，匹配得分
    trackers_[camera_idx]->track(//跟踪，尚有存在疑问的地方
            q_Ckp1_Ck, *frame_k, frame_kp1, &matches_with_score_kp1_k);
    stat_tracking.AddSample(timer_tracking.Stop() * 1000);

    // Remove outlier matches.
    aslam::FrameToFrameMatchesWithScore inlier_matches_with_score_kp1_k;
    aslam::FrameToFrameMatchesWithScore outlier_matches_with_score_kp1_k;

    statistics::StatsCollector stat_ransac("Twopt RANSAC (1 image) in ms");
    timing::Timer timer_ransac(
            "swe-feature-tracker: trackFeaturesSingleCamera - ransac");

    //对前后的匹配进行ransac处理
    //输入当前帧，前一帧，他们两个之间的旋转，匹配的得分，
    //输入当前帧，之前帧，qckc1，当前帧和,剔除外点
    bool ransac_success = aslam::geometric_vision::
    rejectOutlierFeatureMatchesTranslationRotationSAC(
            *frame_kp1, *frame_k, q_Ckp1_Ck, matches_with_score_kp1_k,
            FLAGS_swe_feature_tracker_deterministic,
            FLAGS_swe_feature_tracker_two_pt_ransac_threshold,
            FLAGS_swe_feature_tracker_two_pt_ransac_max_iterations,
            &inlier_matches_with_score_kp1_k, &outlier_matches_with_score_kp1_k);

    stat_ransac.AddSample(timer_ransac.Stop() * 1000);

    LOG_IF(WARNING, !ransac_success)
    << "Match outlier rejection RANSAC failed on camera " << camera_idx
    << ".";
    const size_t num_outliers = outlier_matches_with_score_kp1_k.size();
    VLOG_IF(3, num_outliers > 0) << "Removed " << num_outliers << " outliers of "
                                 << matches_with_score_kp1_k.size()
                                 << " matches on camera " << camera_idx << ".";

    // Assign track ids.
    timing::Timer timer_track_manager(
            "swe-feature-tracker: trackFeaturesSingleCamera - track manager");

    //更新轨迹在这里，输入内点匹配，当前帧，之前帧
    track_managers_[camera_idx]->applyMatchesToFrames(
            inlier_matches_with_score_kp1_k, frame_kp1, frame_k);

    aslam::convertMatchesWithScoreToMatches<aslam::FrameToFrameMatchWithScore,
            aslam::FrameToFrameMatch>(
            inlier_matches_with_score_kp1_k, inlier_matches_kp1_k);
    aslam::convertMatchesWithScoreToMatches<aslam::FrameToFrameMatchWithScore,
            aslam::FrameToFrameMatch>(
            outlier_matches_with_score_kp1_k, outlier_matches_kp1_k);

    if (visualize_keypoint_matches_) {
        cv::Mat image;
        aslam_cv_visualization::visualizeMatches(
                *frame_kp1, *frame_k, inlier_matches_with_score_kp1_k, &image);
        const std::string topic = feature_tracking_ros_base_topic_ +
                                  "/keypoint_matches_camera_" +
                                  std::to_string(camera_idx);
        visualization::RVizVisualizationSink::publish(topic, image);

        cv::Mat outlier_image;
        aslam_cv_visualization::visualizeMatches(
                *frame_kp1, *frame_k, outlier_matches_with_score_kp1_k, &outlier_image);
        const std::string outlier_topic = feature_tracking_ros_base_topic_ +
                                          "/keypoint_outlier_matches_camera_" +
                                          std::to_string(camera_idx);
        visualization::RVizVisualizationSink::publish(outlier_topic, outlier_image);
    }
    timer_track_manager.Stop();
}

//初始化orb特征点提取器，brisk描述子提取器
void VOFeatureTrackingPipeline::initialize(
    const aslam::NCamera::ConstPtr& ncamera)
{
    CHECK(ncamera);
    ncamera_ = ncamera;//多相机信息
    // Create a thread pool.
    const size_t num_cameras = ncamera_->numCameras();//多少个相机
    thread_pool_.reset(new aslam::ThreadPool(num_cameras));

    // Create a feature tracker.
    detectors_extractors_.reserve(num_cameras);
    trackers_.reserve(num_cameras);
    track_managers_.reserve(num_cameras);

    for (size_t cam_idx = 0u; cam_idx < num_cameras; ++cam_idx)
    {
        detectors_extractors_.emplace_back(//orb的特征点检测器，描述子是用的别的
                new FeatureDetectorExtractor(ncamera_->getCamera(cam_idx)));
        //输入当前相机，最小边框，描述子
        trackers_.emplace_back(//看样子是基于陀螺仪的这么一个追踪
                new aslam::GyroTracker(
                        ncamera_->getCamera(cam_idx),
                        detectors_extractors_.back()
                                ->detector_settings_.min_tracking_distance_to_image_border_px,
                        detectors_extractors_.back()->getExtractorPtr()));
        track_managers_.emplace_back(new aslam::SimpleTrackManager);
    }
}

VOFeatureTrackingPipeline::VOFeatureTrackingPipeline()
    : has_feature_extraction_been_performed_on_first_nframe_(false) {}

VOFeatureTrackingPipeline::VOFeatureTrackingPipeline(
    const aslam::NCamera::ConstPtr& ncamera)
    : has_feature_extraction_been_performed_on_first_nframe_(false)//不在第一帧提取特征
    {
  initialize(ncamera);//初始化
}

VOFeatureTrackingPipeline::~VOFeatureTrackingPipeline() {
  if (thread_pool_) {
    thread_pool_->stop();
  }
}
}  // namespace feature_tracking
