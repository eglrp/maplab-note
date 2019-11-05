#include <array>

#include "feature-tracking/feature-detection-extraction.h"

#include <aslam/common/statistics/statistics.h>
#include <aslam/common/timer.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/tracker/tracking-helpers.h>
#include <brisk/brisk.h>
#include <gflags/gflags.h>
#include <opencv/highgui.h>
#include <opencv2/core/version.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "feature-tracking/grided-detector.h"

DEFINE_bool(use_grided_detections, true, "Use multiple detectors on a grid?");

namespace feature_tracking {

FeatureDetectorExtractor::FeatureDetectorExtractor(const aslam::Camera& camera)
    : camera_(camera)//因为特征提取器是针对单个相机而言的，所以这里记录的是单个相机的信息
{
    initialize();//初始化orb特征提取器和描述子
}

void FeatureDetectorExtractor::initialize()
{
  CHECK_LT(
      2 * detector_settings_.min_tracking_distance_to_image_border_px,
      camera_.imageWidth());
  CHECK_LT(
      2 * detector_settings_.min_tracking_distance_to_image_border_px,
      camera_.imageHeight());

  // No distance to the edges is required for the grided detector.网格化探测器不需要与边缘保持距离。,所以这里是0
  const size_t orb_detector_edge_threshold =
      FLAGS_use_grided_detections
          ? 0u
          : detector_settings_.orb_detector_edge_threshold;

  detector_ = cv::ORB::create(
      detector_settings_.orb_detector_number_features,
      detector_settings_.orb_detector_scale_factor,
      detector_settings_.orb_detector_pyramid_levels,
      orb_detector_edge_threshold, detector_settings_.orb_detector_first_level,
      detector_settings_.orb_detector_WTA_K,
      detector_settings_.orb_detector_score_type,
      detector_settings_.orb_detector_patch_size,
      detector_settings_.orb_detector_fast_threshold);

  switch (extractor_settings_.descriptor_type) {
    case SweFeatureTrackingExtractorSettings::DescriptorType::kBrisk:
      extractor_ = new brisk::BriskDescriptorExtractor(
          extractor_settings_.rotation_invariant,
          extractor_settings_.scale_invariant);
      break;
    case SweFeatureTrackingExtractorSettings::DescriptorType::kOcvFreak:
      extractor_ = cv::xfeatures2d::FREAK::create(
          extractor_settings_.rotation_invariant,
          extractor_settings_.scale_invariant,
          extractor_settings_.freak_pattern_scale,
          detector_settings_.orb_detector_pyramid_levels);
      break;
    default:
      LOG(FATAL) << "Unknown descriptor type.";
      break;
  }
}

cv::Ptr<cv::DescriptorExtractor> FeatureDetectorExtractor::getExtractorPtr()
    const {
  return extractor_;
}



//提取当前帧的特征点和描述子,这里提取特征点用的是orb类里的fast，然后描述子使用的是freak或者orb
void FeatureDetectorExtractor::detectAndExtractFeatures(
    aslam::VisualFrame* frame) const
{
    CHECK_NOTNULL(frame);
    CHECK(frame->hasRawImage())
    << "Can only detect keypoints if the frame has a raw image";
    CHECK_EQ(
            camera_.getId(),
            CHECK_NOTNULL(frame->getCameraGeometry().get())->getId());

    timing::Timer timer_detection("keypoint detection");

    std::vector<cv::KeyPoint> keypoints_cv;
    const cv::Mat& image = frame->getRawImage();//得到这帧图像

    if (FLAGS_use_grided_detections)
    {
        // Grided detection to ensure a certain distribution of keypoints across
        // the image.//特征分布均匀性,这里利用orb提取器提取fast特征点
        constexpr size_t kNumGridCols = 3;
        constexpr size_t kNumGridRows = 2;
        detectKeypointsGrided(
                detector_, image, /*detection_mask=*/cv::Mat(),
                detector_settings_.max_feature_count,
                detector_settings_.detector_nonmaxsuppression_radius,
                detector_settings_.detector_nonmaxsuppression_ratio_threshold,
                kNumGridCols, kNumGridRows, &keypoints_cv);
    } else {
        detector_->detect(image, keypoints_cv);

        if (detector_settings_.detector_use_nonmaxsuppression) {
            timing::Timer timer_nms("non-maximum suppression");
            localNonMaximumSuppression(
                    camera_.imageHeight(),
                    detector_settings_.detector_nonmaxsuppression_radius,
                    detector_settings_.detector_nonmaxsuppression_ratio_threshold,
                    &keypoints_cv);

            statistics::StatsCollector stat_nms(
                    "non-maximum suppression (1 image) in ms");
            stat_nms.AddSample(timer_nms.Stop() * 1000);
        }
        cv::KeyPointsFilter::retainBest(
                keypoints_cv, detector_settings_.max_feature_count);
    }

    // The ORB detector tries to always return a constant number of keypoints.
    // If we get into an environment with very few good keypoint candidates
    // the detector adapts it's score such that it even detects keypoints that
    // are dust on the camera. Therefore, we put a lower bound on the threshold.
    size_t num_removed_keypoints = keypoints_cv.size();

    std::vector<cv::KeyPoint>::iterator it_erase_from = std::remove_if(
            keypoints_cv.begin(), keypoints_cv.end(), [this](const cv::KeyPoint& kp) {
                return kp.response <= detector_settings_.orb_detector_score_lower_bound;
            });
    keypoints_cv.erase(it_erase_from, keypoints_cv.end());

    num_removed_keypoints -= keypoints_cv.size();
    VLOG(4) << "Number of removed low score keypoints: " << num_removed_keypoints;

    if (extractor_settings_.flip_descriptor) {
        for (cv::KeyPoint& keypoint : keypoints_cv) {
            keypoint.angle = 180.0f;
        }
    } else {
        // We are doing this because:
        // - Stefan Leutenegger's BRISK implementation uses orientation information
        //   (in case of enabled rotational invariance) of OpenCV keypoints if the
        //   angles are not -1 and the ORB detector assigns orientations to
        //   keypoints.
        //   I don't know which orientation assignment is more robust so feel free
        //   to remove this if you know better. I think there is no significant
        //   performance difference.
        // - Also remove this if you use descriptors (e.g. OpenCV BRISK or ORB)
        //   that rely on orientation information provided by some detectors
        //   (e.g. ORB detector).
        for (cv::KeyPoint& keypoint : keypoints_cv) {
            keypoint.angle = -1.0f;
        }
    }
    timer_detection.Stop();

    timing::Timer timer_extraction("descriptor extraction");

    // Compute the descriptors.
    cv::Mat descriptors_cv;
    if (!keypoints_cv.empty()) {//这里直接用的就是cv的
        extractor_->compute(frame->getRawImage(), keypoints_cv, descriptors_cv);
    } else {
        descriptors_cv = cv::Mat(0, 0, CV_8UC1);
    }

    timer_extraction.Stop();

    // Note: It is important that the values are set even if there are no
    // keypoints as downstream code may rely on the keypoints being set.
    //将cv提取出的关键点和描述子放到视觉帧的channels_中,  aslam::channels::ChannelGroup
    // channels_是可以用来存储各种数据的，比如这帧的描述子，观测，等等
    aslam::insertCvKeypointsAndDescriptorsIntoEmptyVisualFrame(
            keypoints_cv, descriptors_cv, detector_settings_.keypoint_uncertainty_px,
            frame);
}
}  // namespace feature_tracking
