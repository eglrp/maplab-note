#include "matching-based-loopclosure/detector-settings.h"

#include <descriptor-projection/flags.h>
#include <glog/logging.h>
#include <loopclosure-common/flags.h>
#include <loopclosure-common/types.h>

DEFINE_string(
    lc_detector_engine,
    matching_based_loopclosure::kMatchingLDInvertedMultiIndexString,
    "Which loop-closure engine to use");
DEFINE_string(
    lc_scoring_function, matching_based_loopclosure::kAccumulationString,
    "Type of scoring function to be used for scoring keyframes.");
DEFINE_double(
    lc_min_image_time_seconds, 10.0,
    "Minimum time between matching images to allow a loop closure.");
DEFINE_uint64(
    lc_min_verify_matches_num, 10u,
    "The minimum number of matches needed to verify geometry.");
DEFINE_double(
    lc_fraction_best_scores, 0.25,
    "Fraction of best scoring "
    "keyframes/vertices that are considered for covisibility filtering.");
DEFINE_int32(
    lc_num_neighbors, -1,
    "Number of neighbors to retrieve for loop-closure. -1 auto.");
DEFINE_int32(
    lc_num_words_for_nn_search, 10,
    "Number of nearest words to retrieve in the inverted index.");

namespace matching_based_loopclosure {

    //匹配器初始化
MatchingBasedEngineSettings::MatchingBasedEngineSettings()
    : projection_matrix_filename(FLAGS_lc_projection_matrix_filename),
      projected_quantizer_filename(FLAGS_lc_projected_quantizer_filename),//这两个都是文件存储位置
      num_closest_words_for_nn_search(FLAGS_lc_num_words_for_nn_search),//要在倒排索引中检索的最近的单词数
      min_image_time_seconds(FLAGS_lc_min_image_time_seconds),//小于这个时间间隔的图片之间是不会去检测匹配的，不然就会产生跟踪的效果了
      min_verify_matches_num(FLAGS_lc_min_verify_matches_num),//验证几何形状所需的最小匹配数。
      fraction_best_scores(FLAGS_lc_fraction_best_scores),//最佳得分的分数"关键帧顶点，考虑用于共可见性过滤。
      num_nearest_neighbors(FLAGS_lc_num_neighbors) //确定为每个查询描述符检索的最近邻描述符的数量。这个默认值取决于地图中地标/观察值的数量。将其设置为较大的值会增加召回率，但可能会降低精确度。
{
    CHECK_GT(num_closest_words_for_nn_search, 0);
    CHECK_GE(min_image_time_seconds, 0.0);
    CHECK_GE(min_verify_matches_num, 0u);
    CHECK_GT(fraction_best_scores, 0.f);
    CHECK_LT(fraction_best_scores, 1.f);
    CHECK_GE(num_nearest_neighbors, -1);

    setKeyframeScoringFunctionType(FLAGS_lc_scoring_function);//用于对关键帧进行评分的评分函数的类型。,叠加得分
    setDetectorEngineType(FLAGS_lc_detector_engine);//存储索引的方式，默认是inverted_multi_index

    const std::string loop_closure_files_path = getLoopClosureFilePath();//应该是得到闭环所在包的位置
    //下面这两个都没设，都是空的，所以都用的是true
    const bool use_default_projection_matrix_filepath =
            projection_matrix_filename.empty();
    const bool use_default_projected_quantizer_filepath =
            projected_quantizer_filename.empty();
    //在maplab_ws/src/maplab/algorithms/loopclosure/loopclosure-common/src/flags.cc
    // 中定义了使用的哪种描述子,默认是kFeatureDescriptorBRISK
    if (FLAGS_feature_descriptor_type == loop_closure::kFeatureDescriptorFREAK) {
        if (use_default_projection_matrix_filepath) {
            projection_matrix_filename =
                    std::string(loop_closure_files_path) + "/projection_matrix_freak.dat";
        }
        if (use_default_projected_quantizer_filepath) {
            projected_quantizer_filename =
                    std::string(loop_closure_files_path) +
                    "/inverted_multi_index_quantizer_freak.dat";
        }
    } else {
        CHECK_EQ(
                FLAGS_feature_descriptor_type, loop_closure::kFeatureDescriptorBRISK);
        if (use_default_projection_matrix_filepath) {//默认用这个
            projection_matrix_filename =
                    std::string(loop_closure_files_path) + "/projection_matrix_brisk.dat";
        }
        if (use_default_projected_quantizer_filepath) {
            projected_quantizer_filename =
                    std::string(loop_closure_files_path) +
                    "/inverted_multi_index_quantizer_brisk.dat";
        }
    }
}

void MatchingBasedEngineSettings::setKeyframeScoringFunctionType(
    const std::string& scoring_function_string)//得分评估方程的初始化
{
    scoring_function_type_string = scoring_function_string;
    if (scoring_function_string == kAccumulationString) //maplab默认的计算方式，累加
    {
        keyframe_scoring_function_type = KeyframeScoringFunctionType::kAccumulation;
    } else if (scoring_function_string == kProbabilisticString)//基于概率的方式
    {
        keyframe_scoring_function_type =
                KeyframeScoringFunctionType::kProbabilistic;
    } else {
        LOG(FATAL) << "Unknown scoring function type: " << scoring_function_string;
    }
}


/*
 * enum class DetectorEngineType {//这个是搜索匹配的存储索引方式
    kMatchingLDKdTree,
    kMatchingLDInvertedIndex,
    kMatchingLDInvertedMultiIndex,
    kMatchingLDInvertedMultiIndexProductQuantization,
  };
 */

void MatchingBasedEngineSettings::setDetectorEngineType(
    const std::string& detector_engine_string)
{
    detector_engine_type_string = detector_engine_string;
    if (detector_engine_string == kMatchingLDKdTreeString) {
        detector_engine_type = DetectorEngineType::kMatchingLDKdTree;
    } else if (detector_engine_string == kMatchingLDInvertedIndexString) {
        detector_engine_type = DetectorEngineType::kMatchingLDInvertedIndex;
    } else if (detector_engine_string == kMatchingLDInvertedMultiIndexString) {
        detector_engine_type = DetectorEngineType::kMatchingLDInvertedMultiIndex;
    } else if (
            detector_engine_string ==
            kMatchingLDInvertedMultiIndexProductQuantizationString) {
        detector_engine_type =
                DetectorEngineType::kMatchingLDInvertedMultiIndexProductQuantization;
    } else {
        LOG(FATAL) << "Unknown loop detector engine type: "
                   << detector_engine_string;
    }
}

std::string MatchingBasedEngineSettings::getLoopClosureFilePath()
{
  const char* loop_closure_files_path = getenv("MAPLAB_LOOPCLOSURE_DIR");//读取闭环文件夹
  CHECK_NE(loop_closure_files_path, static_cast<char*>(NULL))
      << "MAPLAB_LOOPCLOSURE_DIR environment variable is not set.\n"
      << "Source the Maplab environment from your workspace:\n"
      << "source devel/setup.bash";
  return loop_closure_files_path;
}

}  // namespace matching_based_loopclosure
