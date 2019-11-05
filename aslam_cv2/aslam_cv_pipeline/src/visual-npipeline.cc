#include <aslam/pipeline/visual-npipeline.h>

#include <aslam/cameras/camera.h>
#include <aslam/cameras/ncamera.h>
#include <aslam/common/memory.h>
#include <aslam/common/thread-pool.h>
#include <aslam/common/time.h>
#include <aslam/frames/visual-nframe.h>
#include <aslam/pipeline/visual-pipeline.h>
#include <aslam/pipeline/visual-pipeline-null.h>
#include <glog/logging.h>
#include <opencv2/core/core.hpp>

namespace aslam {

VisualNPipeline::VisualNPipeline(
    size_t num_threads,
    const std::vector<std::shared_ptr<VisualPipeline> >& pipelines,
    const std::shared_ptr<NCamera>& input_camera_system,
    const std::shared_ptr<NCamera>& output_camera_system,
    int64_t timestamp_tolerance_ns) :
      pipelines_(pipelines),
      shutdown_(false),
      input_camera_system_(input_camera_system),
      output_camera_system_(output_camera_system),
      timestamp_tolerance_ns_(timestamp_tolerance_ns)  {
  // Defensive programming ninjitsu.
  CHECK_NOTNULL(input_camera_system_.get());
  CHECK_NOTNULL(output_camera_system.get());
  CHECK_GT(input_camera_system_->numCameras(), 0u);
  CHECK_EQ(input_camera_system_->numCameras(),
           output_camera_system_->numCameras());
  CHECK_EQ(input_camera_system_->numCameras(), pipelines.size());
  CHECK_GE(timestamp_tolerance_ns, 0);

  for (size_t i = 0; i < pipelines.size(); ++i) {
    CHECK_NOTNULL(pipelines[i].get());
    // Check that the input cameras actually point to the same object.
    CHECK_EQ(input_camera_system_->getCameraShared(i).get(),
             pipelines[i]->getInputCameraShared().get());
    // Check that the output cameras actually point to the same object.
    CHECK_EQ(output_camera_system_->getCameraShared(i).get(),
             pipelines[i]->getOutputCameraShared().get());
  }
  CHECK_GT(num_threads, 0u);
  thread_pool_.reset(new ThreadPool(num_threads));
}

VisualNPipeline::~VisualNPipeline() {
  shutdown();
}

void VisualNPipeline::shutdown() {
  shutdown_ = true;
  condition_not_empty_.notify_all();
  condition_not_full_.notify_all();
  thread_pool_->stop();
}

//输入:哪一个相机，对应的图片，这个图片对应的时间戳,最大相机队列
bool VisualNPipeline::processImageBlockingIfFull(
    size_t camera_index, const cv::Mat& image, int64_t timestamp,
    size_t max_queue_size) {
  std::unique_lock<std::mutex> lock(mutex_);
  while (!shutdown_) {
    if (completed_.size() >= max_queue_size) {
      condition_not_full_.wait(lock);
      if (completed_.size() >= max_queue_size) {
        continue;
      }
    }
      //输入:哪一个相机，对应的图片，这个图片对应的时间戳,最大相机队列
//把这些消息组成多相机系统存到completed_里
    processImageImpl(camera_index, image, timestamp);
    return true;
  }
  return false;
}

bool VisualNPipeline::processImageNonBlockingDroppingOldestNFrameIfFull(
    size_t camera_index, const cv::Mat &image, int64_t timestamp,
    size_t max_output_queue_size) {
  CHECK_GE(max_output_queue_size, 1u);

  bool oldest_dropped = false;
  std::lock_guard<std::mutex> lock(mutex_);
  if (completed_.size() >= max_output_queue_size) {
    completed_.erase(completed_.begin());
    condition_not_full_.notify_all();
    oldest_dropped = true;
  }
  processImageImpl(camera_index, image, timestamp);
  return oldest_dropped;
}

bool VisualNPipeline::getNextBlocking(std::shared_ptr<VisualNFrame>* nframe)
{
  CHECK_NOTNULL(nframe);

  std::unique_lock<std::mutex> lock(mutex_);
  while (!shutdown_) {
    if (completed_.empty()) {
      condition_not_empty_.wait(lock);
      if (completed_.empty()) {
        continue;
      }
    }
    // Get the oldest frame.
    *nframe = getNextImpl();
    CHECK(*nframe);
    return true;
  }
  return false;
}

void VisualNPipeline::processImage(
    size_t camera_index, const cv::Mat& image, int64_t timestamp) {
  thread_pool_->enqueue(&VisualNPipeline::work, this, camera_index, image,
                        timestamp);
}

size_t VisualNPipeline::getNumFramesComplete() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return completed_.size();
}

std::shared_ptr<VisualNFrame> VisualNPipeline::getNext() {
  std::lock_guard<std::mutex> lock(mutex_);
  return getNextImpl();
}

std::shared_ptr<VisualNFrame> VisualNPipeline::getNextImpl() {
  // Initialize the return value as null
  std::shared_ptr<VisualNFrame> nframe;
  if (completed_.empty()) {
    return nframe;
  }
  // Get the oldest frame.
  auto it_completed = completed_.begin();
  nframe = it_completed->second;
  completed_.erase(it_completed);
  condition_not_full_.notify_all();
  return nframe;
}
//输入:哪一个相机，对应的图片，这个图片对应的时间戳,最大相机队列
//把这些消息组成多相机系统存到completed_里
void VisualNPipeline::processImageImpl(
    size_t camera_index, const cv::Mat& image, int64_t timestamp)
    {
    //都是在用线程池去处理，//这里主要就是接收所有的消息，然后给他们组成多相机系统存到completed_里
  thread_pool_->enqueue(&VisualNPipeline::work, this, camera_index, image,
                        timestamp);
}

std::shared_ptr<VisualNFrame> VisualNPipeline::getLatestAndClear() {
  std::shared_ptr<VisualNFrame> nframe;
  std::lock_guard<std::mutex> lock(mutex_);
  if (completed_.empty()) {
    return nframe;
  }
  auto reverse_it_completed = completed_.rbegin();
  nframe = reverse_it_completed->second;
  const int64_t timestamp_nanoseconds = reverse_it_completed->first;
  completed_.clear();
  condition_not_full_.notify_all();
  // Clear any processing frames older than this one.
  auto it_processing = processing_.begin();
  while (it_processing != processing_.end()
      && it_processing->first <= timestamp_nanoseconds) {
    it_processing = processing_.erase(it_processing);
  }
  return nframe;
}

bool VisualNPipeline::getLatestAndClearBlocking(
    std::shared_ptr<VisualNFrame>* nframe)  {
  CHECK_NOTNULL(nframe);

  std::unique_lock<std::mutex> lock(mutex_);
  while (!shutdown_) {
    if (completed_.empty()) {
      condition_not_empty_.wait(lock);
      if (completed_.empty()) {
        continue;
      }
    }
    TimestampVisualNFrameMap::const_reverse_iterator nframe_iterator =
        completed_.rbegin();
    CHECK(nframe_iterator != completed_.rend());
    *nframe = nframe_iterator->second;
    CHECK(*nframe);
    const int64_t timestamp_nanoseconds = nframe_iterator->first;
    completed_.clear();
    condition_not_full_.notify_all();
    // Clear any processing frames older than this one.
    TimestampVisualNFrameMap::iterator processing_iterator =
        processing_.begin();
    while (processing_iterator != processing_.end() &&
        processing_iterator->first <= timestamp_nanoseconds) {
      processing_iterator = processing_.erase(processing_iterator);
    }
    return true;
  }
  return false;
}

std::shared_ptr<const NCamera> VisualNPipeline::getInputNCameras() const {
  return input_camera_system_;
}

std::shared_ptr<const NCamera> VisualNPipeline::getOutputNCameras() const {
  return output_camera_system_;
}

size_t VisualNPipeline::getNumFramesProcessing() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return processing_.size();
}

//这里主要就是接收所有的消息，然后给他们组成多相机系统存到completed_里
void VisualNPipeline::work(size_t camera_index, const cv::Mat& image,
                           int64_t timestamp_nanoseconds)
{
    CHECK_LE(camera_index, pipelines_.size());//size就是相机的个数
    std::shared_ptr<VisualFrame> frame;
    frame = pipelines_[camera_index]->processImage(image, timestamp_nanoseconds);//输入图片和时间戳,得到VisualFrame里面是帧的结构

    /// Create an iterator into the processing queue.
    std::map<int64_t, std::shared_ptr<VisualNFrame>>::iterator proc_it;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        bool create_new_nframes = false;
        if (processing_.empty())
        {
            create_new_nframes = true;//应该就是刚开始会触发这个
        } else
        {
            // Try to find an existing NFrame in the processing list.
            // Use the timestamp of the frame because there may be a timestamp
            // corrector used in the pipeline.//尝试在处理列表中找到一个现有的NFrame。使用帧的时间戳，因为管道中可能使用了时间戳校正器。
            auto it_processing = processing_.lower_bound(//返回最近的小于等于这帧当前时间戳的某帧
                    frame->getTimestampNanoseconds());
            // Lower bound returns the first element that is not less than the value
            // (i.e. greater than or equal to the value).
            if (it_processing != processing_.begin()) { --it_processing; }//这个是为了找到比当前帧小的么？
            // Now it_processing points to the first element that is less than the
            // value. Check both this value, and the one >=.
            int64_t min_time_diff = std::abs(//这两帧之间的时间间隔是最小的
                    it_processing->first - frame->getTimestampNanoseconds());
            proc_it = it_processing;
            if (++it_processing != processing_.end())
            {
                const int64_t time_diff = std::abs(//向后遍历，总之就是为了找到和当前这帧离的最近的某帧
                        it_processing->first - frame->getTimestampNanoseconds());
                if (time_diff < min_time_diff) {
                    proc_it = it_processing;
                    min_time_diff = time_diff;
                }
            }
            // Now proc_it points to the closest nframes element.//如果最小的这个时间漂移大于了阈值，那么就认为可以创建新的帧
            if (min_time_diff > timestamp_tolerance_ns_) {
                create_new_nframes = true;
            }
        }

        if (create_new_nframes)//需要创建新的nframes系统，刚开始的时候要创建，然后就是根据时间阈值判断是否要加入
        {
            std::shared_ptr<VisualNFrame> nframes(//利用多相机生成NFrame
                    new VisualNFrame(output_camera_system_));
            bool not_replaced;
            std::tie(proc_it, not_replaced) = processing_.insert(//在map中添加这个多相机系统
                    std::make_pair(frame->getTimestampNanoseconds(), nframes)
            );
            CHECK(not_replaced);
        }
        // Now proc_it points to the correct place in the processing_ list and
        // the NFrame has been created if necessary.
        VisualFrame::Ptr existing_frame = proc_it->second->getFrameShared(
                camera_index);//这里应该是不存在这个相机的id的
        if (existing_frame)
        {
            LOG(ERROR) << "Overwriting a frame at index " << camera_index << ":" << std::endl
                       << *existing_frame << std::endl << "with a new frame: "
                       << *frame << std::endl << "because the timestamp was the same.";
        }
        proc_it->second->setFrame(camera_index, frame);//在对应的相机id位置添加帧

        // Find the first index that has N consecutive complete nframes following in chronological
        // ordering.////按时间顺序找到第一个有N个连续完整nframe的索引。
        // E.g. N=3    I I C I C C C C C   (I: incomplete, C: complete)
        //      idx    0 1 2 3 4 5 6 7 8
        //                     # --> first index with N complete = 4
        const size_t kNumMinConsecutiveCompleteThreshold = 2u;
        int delete_upto_including_index = -1;
        if (processing_.size() > kNumMinConsecutiveCompleteThreshold + 1) //如果正在处理的大于了3个，这里是3个多相机 , maplab就会进行删除
        {
            size_t num_consecutive_complete = 0u;
            size_t idx = 0u;
            auto it_processing = processing_.begin();
            while (it_processing != processing_.end())
            {
                bool is_complete = CHECK_NOTNULL(it_processing->second.get())->areAllFramesSet();//检查这个多相机系统里的所有相机的帧是不是都有
                if (is_complete)
                {//完成的化
                    ++num_consecutive_complete;
                } else {//没有完成就置0
                    num_consecutive_complete = 0u;
                }
                if (num_consecutive_complete >= kNumMinConsecutiveCompleteThreshold) {//完成数量大于等于2就可以
                    delete_upto_including_index = static_cast<int>(idx) -//delete_upto_including_index从0开始，或者从1
                                                  static_cast<int>(kNumMinConsecutiveCompleteThreshold);
                    break;
                }
                ++it_processing;
                ++idx;
            }
        }



//现在将所有不完整的nframes删除为delete_upto_includes _index。所有低于此索引的帧可能永远都不会完成，因为多相机中的一个相机掉了一张图像。
        // Now drop all incomplete nframes up to delete_upto_including_index. All frames below this
        // index will probably never complete as one camera in the rig dropped an image.
        if (delete_upto_including_index >= 0)
        {
            int num_nframes_to_delete = delete_upto_including_index + 1;
            auto it_processing = processing_.begin();
            while (it_processing != processing_.end() && num_nframes_to_delete-- > 0)
            {
                it_processing = processing_.erase(it_processing);
            }
            LOG(WARNING) << "Detected frame drop: removing " << delete_upto_including_index + 1
                         << " nframes from the queue.";
        }
        //按时间顺序将所有完成的nframe从processed_队列移动到completed_队列。
        // Move all completed nframes from the processed_ queue to the completed_ queue chronologically.
        auto it_processing = processing_.begin();
        while (it_processing != processing_.end())
        {
            // Check if all images have been received.
            if (it_processing->second->areAllFramesSet())
            {//从头遍历，如果这个多相机系统里所有的相机都存在的话，那么就认为已经完成了
                completed_.insert(*it_processing);
                it_processing = processing_.erase(it_processing);
                condition_not_empty_.notify_all();
            } else {
                // As we are iterating over the map in chronological order we have to abort once an nframe
                // is not yet finished processing to keep chronological ordering in the destination queue.
                break;
            }
        }
    }
}

void VisualNPipeline::waitForAllWorkToComplete() const {
  thread_pool_->waitForEmptyQueue();
}

VisualNPipeline::Ptr VisualNPipeline::createTestVisualNPipeline(
    size_t num_cameras, size_t num_threads, int64_t timestamp_tolerance_ns) {
  NCamera::Ptr ncamera = NCamera::createTestNCamera(num_cameras);
  CHECK_EQ(ncamera->numCameras(), num_cameras);
  const bool kCopyImages = false;
  std::vector<VisualPipeline::Ptr> null_pipelines;
  for (size_t frame_idx = 0; frame_idx < num_cameras; ++frame_idx) {
    CHECK(ncamera->getCameraShared(frame_idx));
    null_pipelines.push_back(
        aligned_shared<NullVisualPipeline>(
            ncamera->getCameraShared(frame_idx), kCopyImages));
  }
  VisualNPipeline::Ptr npipeline = aligned_shared<VisualNPipeline>(
      num_threads, null_pipelines, ncamera, ncamera, timestamp_tolerance_ns);
  return npipeline;
}
}  // namespace aslam
