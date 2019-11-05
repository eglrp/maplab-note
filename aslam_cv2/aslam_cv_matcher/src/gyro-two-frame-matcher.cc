#include "aslam/matcher/gyro-two-frame-matcher.h"

#include <aslam/common/statistics/statistics.h>

namespace aslam {
//初始化匹配器，输入的是qbkb1，当前帧，前一帧，相机的高度,预测的下一帧的特征点的像素坐标，是否帧的预测上了，匹配结果
    GyroTwoFrameMatcher::GyroTwoFrameMatcher(
            const Quaternion& q_Ckp1_Ck,
            const VisualFrame& frame_kp1,
            const VisualFrame& frame_k,
            const uint32_t image_height,
            const Eigen::Matrix2Xd& predicted_keypoint_positions_kp1,
            const std::vector<unsigned char>& prediction_success,
            FrameToFrameMatchesWithScore* matches_with_score_kp1_k)
            : frame_kp1_(frame_kp1), frame_k_(frame_k), q_Ckp1_Ck_(q_Ckp1_Ck),
              predicted_keypoint_positions_kp1_(predicted_keypoint_positions_kp1),
              prediction_success_(prediction_success),
              kDescriptorSizeBytes(frame_kp1.getDescriptorSizeBytes()),
              kNumPointsKp1(frame_kp1.getKeypointMeasurements().cols()),
              kNumPointsK(frame_k.getKeypointMeasurements().cols()),
              kImageHeight(image_height),
              matches_kp1_k_(matches_with_score_kp1_k),
              is_keypoint_kp1_matched_(kNumPointsKp1, false),
              iteration_processed_keypoints_kp1_(kNumPointsKp1, false) {
        CHECK(frame_kp1.isValid());
        CHECK(frame_k.isValid());
        CHECK(frame_kp1.hasDescriptors());
        CHECK(frame_k.hasDescriptors());
        CHECK(frame_kp1.hasKeypointMeasurements());
        CHECK(frame_k.hasKeypointMeasurements());
        CHECK_GT(frame_kp1.getTimestampNanoseconds(), frame_k.getTimestampNanoseconds());
        CHECK_NOTNULL(matches_kp1_k_)->clear();
        CHECK_EQ(kNumPointsKp1, frame_kp1.getDescriptors().cols()) <<
                                                                   "Number of keypoints and descriptors in frame k+1 is not the same.";
        CHECK_EQ(kNumPointsK, frame_k.getDescriptors().cols()) <<
                                                               "Number of keypoints and descriptors in frame k is not the same.";
        CHECK_LE(kDescriptorSizeBytes*8, 512u) << "Usually binary descriptors' size "
                                                  "is less or equal to 512 bits. Adapt the following check if this "
                                                  "framework uses larger binary descriptors.";
        CHECK_GT(kImageHeight, 0u);
        CHECK_EQ(iteration_processed_keypoints_kp1_.size(), kNumPointsKp1);
        CHECK_EQ(is_keypoint_kp1_matched_.size(), kNumPointsKp1);
        CHECK_EQ(prediction_success_.size(), predicted_keypoint_positions_kp1_.cols());

        descriptors_kp1_wrapped_.reserve(kNumPointsKp1);
        keypoints_kp1_sorted_by_y_.reserve(kNumPointsKp1);
        descriptors_k_wrapped_.reserve(kNumPointsK);
        matches_kp1_k_->reserve(kNumPointsK);
        corner_row_LUT_.reserve(kImageHeight);
    }

//主要做的就是储存前后两帧的描述子，然后对当前帧的特征点按y的大小进行排序
    void GyroTwoFrameMatcher::initialize() {
        // Prepare descriptors for efficient matching.
        const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& descriptors_kp1 =
                frame_kp1_.getDescriptors();//当前帧的描述子
        const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& descriptors_k =
                frame_k_.getDescriptors();//之前帧的描述子匹配

        for (int descriptor_kp1_idx = 0; descriptor_kp1_idx < kNumPointsKp1;
             ++descriptor_kp1_idx) //遍历所有的描述子
        {//储存当前帧的描述子
            descriptors_kp1_wrapped_.emplace_back(
                    &(descriptors_kp1.coeffRef(0, descriptor_kp1_idx)), kDescriptorSizeBytes);
        }

        //储存前一帧的描述子
        for (int descriptor_k_idx = 0; descriptor_k_idx < kNumPointsK;
             ++descriptor_k_idx) {
            descriptors_k_wrapped_.emplace_back(
                    &(descriptors_k.coeffRef(0, descriptor_k_idx)), kDescriptorSizeBytes);
        }

        // Sort keypoints of frame (k+1) from small to large y coordinates.
        for (int i = 0; i < kNumPointsKp1; ++i)//按照特征点的像素坐标的y的大小来排列
        {
            keypoints_kp1_sorted_by_y_.emplace_back(frame_kp1_.getKeypointMeasurement(i), i);
        }

        std::sort(keypoints_kp1_sorted_by_y_.begin(), keypoints_kp1_sorted_by_y_.end(),
                  [](const KeypointData& lhs, const KeypointData& rhs)-> bool {
                      return lhs.measurement(1) < rhs.measurement(1);
                  });

        // Lookup table construction.Lookup table construction.
        // TODO(magehrig):  Sort by y if image height >= image width,
        //                  otherwise sort by x.
        int v = 0;
        for (size_t y = 0u; y < kImageHeight; ++y)
        {
            while (v < kNumPointsKp1 &&
                   y > static_cast<size_t>(keypoints_kp1_sorted_by_y_[v].measurement(1)))
            {
                ++v;
            }
            corner_row_LUT_.push_back(v);//corner_row_LUT的每个维度就对应于图像的每一行。所以x=corner_row_LUT_[i]
            // 的意思就是比图像第i行上方的特征点的数量
        }
        CHECK_EQ(static_cast<int>(corner_row_LUT_.size()), kImageHeight);
    }

//开始用陀螺仪预测模式的匹配
    void GyroTwoFrameMatcher::match()
    {
        initialize();//主要做的就是储存前后两帧的描述子，然后对当前帧的特征点按y的大小进行排序

        if (kNumPointsK == 0 || kNumPointsKp1 == 0) {//没有特征点
            return;
        }

        for (int i = 0; i < kNumPointsK; ++i) //遍历上一帧的特征点
        {
            matchKeypoint(i);//对上一帧的这个关键点进行匹配
        }

        std::vector<bool> is_inferior_keypoint_kp1_matched(
                is_keypoint_kp1_matched_);//用is_keypoint_kp1_matched_去做初始化
        for (size_t i = 0u; i < kMaxNumInferiorIterations; ++i) //去迭代匹配之前没有匹配上的描述子
        {
            if(!matchInferiorMatches(&is_inferior_keypoint_kp1_matched)) return;
        }
    }

    //对前一帧的某一个描述子进行匹配
    void GyroTwoFrameMatcher::matchKeypoint(const int idx_k)
    {
        if (!prediction_success_[idx_k])
        {//如果上一帧没有预测成功，那么直接跳过就ok
            return;
        }

        std::fill(iteration_processed_keypoints_kp1_.begin(),//这个记录的是否已经处理过当前帧的特征点
                  iteration_processed_keypoints_kp1_.end(),
                  false);

        bool found = false;
        bool passed_ratio_test = false;
        int n_processed_corners = 0;//已经处理过的

        KeyPointIterator it_best;
        const static unsigned int kDescriptorSizeBits = 8 * kDescriptorSizeBytes;
        int best_score = static_cast<int>(//就是只要字节的匹配程度大于百分之80,那么就认为可以匹配
                kDescriptorSizeBits * kMatchingThresholdBitsRatioRelaxed);
        unsigned int distance_best = kDescriptorSizeBits + 1;
        unsigned int distance_second_best = kDescriptorSizeBits + 1;
        const common::FeatureDescriptorConstRef& descriptor_k =
                descriptors_k_wrapped_[idx_k];//前一帧的特征点对应的描述子

        Eigen::Vector2d predicted_keypoint_position_kp1 =//这个是用上一帧的特征点去用陀螺仪的旋转信息做预测给出的当前帧这个特征点应该在的位置
                predicted_keypoint_positions_kp1_.block<2, 1>(0, idx_k);
        KeyPointIterator nearest_corners_begin, nearest_corners_end;
        //这个就是确定一下当前帧特征点要搜索的索引，按照行的搜索范围来决定，默认的是上下10行的范围，也就是确定上下的搜索范围
        getKeypointIteratorsInWindow<kSmallSearchDistance>(
                predicted_keypoint_position_kp1, &nearest_corners_begin, &nearest_corners_end);

        //这个是确定左右的搜索范围
        const int bound_left_nearest =
                predicted_keypoint_position_kp1(0) - kSmallSearchDistance;
        const int bound_right_nearest =
                predicted_keypoint_position_kp1(0) + kSmallSearchDistance;

        MatchData current_match_data;//存储匹配信息

        // First search small window.//遍历所有可能的当前帧的特征点，然后更新最高得分，并且把大于百分之80的都加进候选描述子
        for (KeyPointIterator it = nearest_corners_begin; it != nearest_corners_end; ++it)
        {//不在搜索窗口内直接跳过
            if (it->measurement(0) < bound_left_nearest ||
                it->measurement(0) > bound_right_nearest) {
                continue;
            }

            CHECK_LT(it->channel_index, kNumPointsKp1);//检查一下特征点的索引
            CHECK_GE(it->channel_index, 0u);
            const common::FeatureDescriptorConstRef& descriptor_kp1 =
                    descriptors_kp1_wrapped_[it->channel_index];//找到这个候选特征点对应的描述子
            unsigned int distance = common::GetNumBitsDifferent(descriptor_k, descriptor_kp1);//记录汉明距离，也可以说是差异
            int current_score = kDescriptorSizeBits - distance;//它这里定义的得分是指的是相同的字节
            if (current_score > best_score) //更新最好得分
            {
                best_score = current_score;//记录下最高的得分
                distance_second_best = distance_best;//更新下第二好的距离
                distance_best = distance;//更新最好的得分
                it_best = it;//当前最可能匹配的描述子
                found = true;//
            } else if (distance < distance_second_best)
            {
                // The second best distance can also belong
                // to two descriptors that do not qualify as match.
                distance_second_best = distance;
            }
            iteration_processed_keypoints_kp1_[it->channel_index] = true;//设置一下这个特征点已经处理过了
            ++n_processed_corners;
            const double current_matching_score =//这个的意思就是当前帧的描述子和之前帧的描述子之间的相似程度
                    computeMatchingScore(current_score, kDescriptorSizeBits);
            current_match_data.addCandidate(it, current_matching_score);//在当前匹配数据中添加进候选信息
        }

        // If no match in small window, increase window and search again.
        //如果没有发现大于百分之80相似的，就扩大搜索范围，变成20×20去进行搜索，步骤同上
        if (!found)
        {
            const int bound_left_near =
                    predicted_keypoint_position_kp1(0) - kLargeSearchDistance;
            const int bound_right_near =
                    predicted_keypoint_position_kp1(0) + kLargeSearchDistance;

            KeyPointIterator near_corners_begin, near_corners_end;
            getKeypointIteratorsInWindow<kLargeSearchDistance>(
                    predicted_keypoint_position_kp1, &near_corners_begin, &near_corners_end);

            for (KeyPointIterator it = near_corners_begin; it != near_corners_end; ++it) {
                if (iteration_processed_keypoints_kp1_[it->channel_index]) {
                    continue;
                }
                if (it->measurement(0) < bound_left_near ||
                    it->measurement(0) > bound_right_near) {
                    continue;
                }
                CHECK_LT(it->channel_index, kNumPointsKp1);
                CHECK_GE(it->channel_index, 0);
                const common::FeatureDescriptorConstRef& descriptor_kp1 =
                        descriptors_kp1_wrapped_[it->channel_index];
                unsigned int distance =
                        common::GetNumBitsDifferent(descriptor_k, descriptor_kp1);
                int current_score = kDescriptorSizeBits - distance;
                if (current_score > best_score) {
                    best_score = current_score;
                    distance_second_best = distance_best;
                    distance_best = distance;
                    it_best = it;
                    found = true;
                } else if (distance < distance_second_best) {
                    // The second best distance can also belong
                    // to two descriptors that do not qualify as match.
                    distance_second_best = distance;
                }
                ++n_processed_corners;
                const double current_matching_score =
                        computeMatchingScore(current_score, kDescriptorSizeBits);
                current_match_data.addCandidate(it, current_matching_score);//
            }
        }

        //如果找到的话，输入描述子的字节数，最佳距离，第二好的距离
        if (found)
        {//passed_ratio_test是是否通过这个测试
            passed_ratio_test = ratioTest(kDescriptorSizeBits, distance_best,
                                          distance_second_best);
        }

        if (passed_ratio_test) //如果通过了独立性测试，这里都是对最好的匹配进行设置
        {
            CHECK(idx_k_to_attempted_match_data_map_.insert(
                    std::make_pair(idx_k, current_match_data)).second);
            const int best_match_keypoint_idx_kp1 = it_best->channel_index;//当前最佳描述子对应的特征点索引
            //best_score就是0,8×描述子字节数，比如512啊之类的, 就是算最佳的这个占比了百分之多少
            const double matching_score = computeMatchingScore(
                    best_score, kDescriptorSizeBits);
            if (is_keypoint_kp1_matched_[best_match_keypoint_idx_kp1]) //如果这个描述子之前已经被匹配过了
            {
                if (matching_score > kp1_idx_to_matches_iterator_map_//说明这次匹配比之前要好
                [best_match_keypoint_idx_kp1]->getScore())
                {

                    // The current match is better than a previous match associated with the
                    // current keypoint of frame (k+1). Hence, the inferior match is the
                    // previous match associated with the current keypoint of frame (k+1).
                    const int inferior_keypoint_idx_k =//把之前这个描述子匹配到的之前帧的描述子的索引提取出来
                            kp1_idx_to_matches_iterator_map_
                            [best_match_keypoint_idx_kp1]->getKeypointIndexBananaFrame();
                    inferior_match_keypoint_idx_k_.push_back(inferior_keypoint_idx_k);//将差的匹配记录
                    //如果我之前便

                    kp1_idx_to_matches_iterator_map_//重新为当前描述子设置匹配得分
                    [best_match_keypoint_idx_kp1]->setScore(matching_score);
                    kp1_idx_to_matches_iterator_map_//苹果索引就是当前帧的描述子索引
                    [best_match_keypoint_idx_kp1]->setIndexApple(best_match_keypoint_idx_kp1);
                    kp1_idx_to_matches_iterator_map_//香蕉索引就是前一帧的描述子索引
                    [best_match_keypoint_idx_kp1]->setIndexBanana(idx_k);
                } else
                    {
                    // The current match is inferior to a previous match associated with the
                    // current keypoint of frame (k+1).//这个是比之前匹配要差，所以之前帧正在匹配的描述子的id加进来
                    inferior_match_keypoint_idx_k_.push_back(idx_k);
                }
            } else
            {
                is_keypoint_kp1_matched_[best_match_keypoint_idx_kp1] = true;//如果以前没有被匹配到过，就将匹配到的描述子设成true
                matches_kp1_k_->emplace_back(//这里是记录的第一次匹配的信息,记录了前后帧的某个描述子之间的匹配和匹配得分
                        best_match_keypoint_idx_kp1, idx_k, matching_score);

                CHECK(matches_kp1_k_->end() != matches_kp1_k_->begin())
                << "Match vector should not be empty.";
                CHECK(kp1_idx_to_matches_iterator_map_.emplace(
                        best_match_keypoint_idx_kp1, matches_kp1_k_->end() - 1).second);
            }

            statistics::StatsCollector stats_distance_match(
                    "GyroTracker: number of matching bits");
            stats_distance_match.AddSample(best_score);
        }
        statistics::StatsCollector stats_count_processed(
                "GyroTracker: number of computed distances per keypoint");
        stats_count_processed.AddSample(n_processed_corners);
    }

    bool GyroTwoFrameMatcher::matchInferiorMatches(
            std::vector<bool>* is_inferior_keypoint_kp1_matched)
    {
        CHECK_NOTNULL(is_inferior_keypoint_kp1_matched);
        CHECK_EQ(is_inferior_keypoint_kp1_matched->size(), is_keypoint_kp1_matched_.size());

        bool found_inferior_match = false;//是否发现了差一些的匹配

        std::unordered_set<int> erase_inferior_match_keypoint_idx_k;
        for (const int inferior_keypoint_idx_k : inferior_match_keypoint_idx_k_) //遍历所有沦为第二好的前一帧的描述子的匹配
        {
            const MatchData& match_data =//这是前一帧这个描述子对应的所有的可能的匹配
                    idx_k_to_attempted_match_data_map_[inferior_keypoint_idx_k];
            bool found = false;//是否发现
            double best_matching_score = static_cast<double>(kMatchingThresholdBitsRatioStrict);
            //之所以要把匹配分数提高，是因为现在已经没有独立性的判别了
            KeyPointIterator it_best;

            //遍历所有的前一帧的这个描述子的差异
            for (size_t i = 0u; i < match_data.keypoint_match_candidates_kp1.size(); ++i)
            {
                //取出这个描述子和匹配的得分
                const KeyPointIterator& keypoint_kp1 = match_data.keypoint_match_candidates_kp1[i];
                const double matching_score = match_data.match_candidate_matching_scores[i];
                // Make sure that we don't try to match with already matched keypoints
                // of frame (k+1) (also previous inferior matches).
                if (is_keypoint_kp1_matched_[keypoint_kp1->channel_index]) continue;//如果匹配过了就跳过
                if (matching_score > best_matching_score) {//找到得分大于0,85的
                    it_best = keypoint_kp1;
                    best_matching_score = matching_score;
                    found = true;
                }
            }

            if (found)
            {
                found_inferior_match = true;
                const int best_match_keypoint_idx_kp1 = it_best->channel_index;
                if ((*is_inferior_keypoint_kp1_matched)[best_match_keypoint_idx_kp1])//如果匹配过了
                {
                    if (best_matching_score > kp1_idx_to_matches_iterator_map_
                    [best_match_keypoint_idx_kp1]->getScore()) {
                        // The current match is better than a previous match associated with the
                        // current keypoint of frame (k+1). Hence, the revoked match is the
                        // previous match associated with the current keypoint of frame (k+1).
                        ///当前的匹配比以前的匹配更好
                        //
                        ////当前帧的关键点(k+1)。因此，被撤销的匹配是
                        //
                        ////前一个匹配与当前关键帧(k+1)相关联。
                        //这后面实际上和之前匹配都一样了
                        const int revoked_inferior_keypoint_idx_k =
                                kp1_idx_to_matches_iterator_map_
                                [best_match_keypoint_idx_kp1]->getKeypointIndexBananaFrame();
                        // The current keypoint k does not have to be matched anymore
                        // in the next iteration.
                        erase_inferior_match_keypoint_idx_k.insert(inferior_keypoint_idx_k);
                        // The keypoint k that was revoked. That means that it can be matched
                        // again in the next iteration.
                        erase_inferior_match_keypoint_idx_k.erase(revoked_inferior_keypoint_idx_k);

                        kp1_idx_to_matches_iterator_map_
                        [best_match_keypoint_idx_kp1]->setScore(best_matching_score);
                        kp1_idx_to_matches_iterator_map_
                        [best_match_keypoint_idx_kp1]->setIndexApple(best_match_keypoint_idx_kp1);
                        kp1_idx_to_matches_iterator_map_
                        [best_match_keypoint_idx_kp1]->setIndexBanana(inferior_keypoint_idx_k);
                    }
                } else
                {
                    (*is_inferior_keypoint_kp1_matched)[best_match_keypoint_idx_kp1] = true;
                    matches_kp1_k_->emplace_back(
                            best_match_keypoint_idx_kp1, inferior_keypoint_idx_k, best_matching_score);
                    erase_inferior_match_keypoint_idx_k.insert(inferior_keypoint_idx_k);

                    CHECK(matches_kp1_k_->end() != matches_kp1_k_->begin())
                    << "Match vector should not be empty.";
                    CHECK(kp1_idx_to_matches_iterator_map_.emplace(
                            best_match_keypoint_idx_kp1, matches_kp1_k_->end() - 1).second);
                }
            }
        }

        if (erase_inferior_match_keypoint_idx_k.size() > 0u) {
            // Do not iterate again over newly matched keypoints of frame k.
            // Hence, remove the matched keypoints.
            std::vector<int>::iterator iter_erase_from = std::remove_if(
                    inferior_match_keypoint_idx_k_.begin(), inferior_match_keypoint_idx_k_.end(),
                    [&erase_inferior_match_keypoint_idx_k](const int element) -> bool {
                        return erase_inferior_match_keypoint_idx_k.count(element) == 1u;
                    }
            );
            inferior_match_keypoint_idx_k_.erase(
                    iter_erase_from, inferior_match_keypoint_idx_k_.end());
        }

        // Subsequent iterations should not mess with the current matches.
        is_keypoint_kp1_matched_ = *is_inferior_keypoint_kp1_matched;

        return found_inferior_match;
    }


} // namespace aslam
