#ifndef FEATURE_TRACKING_FEATURE_TRACKING_TYPES_H_
#define FEATURE_TRACKING_FEATURE_TRACKING_TYPES_H_

#include <string>

namespace feature_tracking {

struct SimpleBriskFeatureTrackingSettings {
  SimpleBriskFeatureTrackingSettings();
  virtual ~SimpleBriskFeatureTrackingSettings() = default;
  const size_t num_octaves;
  const double uniformity_radius;
  const double absolute_threshold;
  const size_t max_number_of_keypoints;
  const bool rotation_invariant;
  const bool scale_invariant;
  const bool copy_images;
  const int matching_descriptor_hamming_distance_threshold;
  const double matching_image_space_distance_threshold_px;
};

struct SweFeatureTrackingExtractorSettings {
  enum class DescriptorType { kOcvFreak, kBrisk };
  SweFeatureTrackingExtractorSettings();
  DescriptorType convertStringToDescriptorType(
      const std::string& descriptor_string);
  /// Type of descriptor used by SWE.
  DescriptorType descriptor_type;

  /// Common settings of all descriptors.
  bool rotation_invariant;//是否要具有旋转不变性
  bool scale_invariant;//是否尺度不变性
  bool flip_descriptor;//是否要对特征点进行反转

  /// FREAK settings.
  float freak_pattern_scale;
};

struct SweFeatureTrackingDetectorSettings {
  SweFeatureTrackingDetectorSettings();

  // Settings for the non-maximum suppression algorithm.
  bool detector_use_nonmaxsuppression;//特征检测后是否应用非最大抑制?
  float detector_nonmaxsuppression_radius;//以被查询的关键点为中心的半径，其他关键点可以被抑制
  float detector_nonmaxsuppression_ratio_threshold;//如果关键点的响应低于此阈值乘以查询的关键点的响应，则禁止它们。较低的阈值应使抑制更强。然而，这将导致更多的关键点彼此接近。

  // ORB detector settings.
  // An adaption of the FAST detector designed for the ORB descriptor.
  // Suitable for real-time applications.
  int orb_detector_number_features;//提取多少点
  float orb_detector_scale_factor;
  int orb_detector_pyramid_levels;
  int orb_detector_edge_threshold;
  // It should be 0 according to the OpenCV documentation.
  int orb_detector_first_level;
  // The number of points that produce each element of the oriented
  // BRIEF descriptor. Check OCV documentation for more information.
  int orb_detector_WTA_K;
  // The default HARRIS_SCORE means that Harris algorithm is used to rank
  // features. FAST_SCORE is alternative value of the parameter that produces
  // slightly less stable keypoints, but it is a little faster to compute.
  int orb_detector_score_type;
  // Size of the patch used by the oriented BRIEF descriptor. Of course, on
  // smaller pyramid layers the perceived image area covered by a feature will
  // be larger.
  int orb_detector_patch_size;
  // Lower bound for the keypoint score. Keypoints with a lower score will
  // be removed.
  float orb_detector_score_lower_bound;//就是检测特征点的阈值
  int orb_detector_fast_threshold;


  // Maximum number of keypoint to detect.
  size_t max_feature_count;

  // Enforce a minimal distance to the image border for feature tracking.
  // Hence, features can be detected close to the image border but
  // might not be tracked if the predicted location of the keypoint in the
  // next frame is closer to the image border than this value.
  //在图像边界上设置一个最小的距离来跟踪特征。因此，可以在靠近图像边界的地方检测到特征，但如果下一帧中关键点的预测位置比这个值更靠近图像边界，则可能无法跟踪特征。
  size_t min_tracking_distance_to_image_border_px;

  double keypoint_uncertainty_px;
};

}  // namespace feature_tracking

#endif  // FEATURE_TRACKING_FEATURE_TRACKING_TYPES_H_
