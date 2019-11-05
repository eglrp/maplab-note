#include <aslam/pipeline/visual-pipeline.h>

#include <aslam/cameras/camera.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/pipeline/undistorter.h>

#include <opencv2/core/core.hpp>

namespace aslam {

VisualPipeline::VisualPipeline(const Camera::ConstPtr& input_camera,
                               const Camera::ConstPtr& output_camera, bool copy_images)
: input_camera_(input_camera), output_camera_(output_camera),//默认不拷贝图像
  copy_images_(copy_images) {
  CHECK(input_camera);
  CHECK(output_camera);
}


VisualPipeline::VisualPipeline(std::unique_ptr<Undistorter>& preprocessing, bool copy_images)
: preprocessing_(std::move(preprocessing)),
  copy_images_(copy_images) {
  CHECK_NOTNULL(preprocessing_.get());
  input_camera_ = preprocessing_->getInputCameraShared();
  output_camera_ = preprocessing_->getOutputCameraShared();
}

std::shared_ptr<VisualFrame> VisualPipeline::processImage(const cv::Mat& raw_image,
                                                          int64_t timestamp) const
{
    CHECK_EQ(input_camera_->imageWidth(), static_cast<size_t>(raw_image.cols));
    CHECK_EQ(input_camera_->imageHeight(), static_cast<size_t>(raw_image.rows));

    // \TODO(PTF) Eventually we can put timestamp correction policies in here.
    std::shared_ptr<VisualFrame> frame(new VisualFrame);//这个就是一个相机中的一个帧的数据结构
    frame->setTimestampNanoseconds(timestamp);
    frame->setRawCameraGeometry(input_camera_);//设置输入相机
    frame->setCameraGeometry(output_camera_);//设置输出相机
    FrameId id;//设置的随机id
    id.randomize();
    frame->setId(id);
    if(copy_images_)
    {
        frame->setRawImage(raw_image.clone());
    } else {
        frame->setRawImage(raw_image);
    }

    cv::Mat image;
    //是否要预处理？
    if(preprocessing_) {
        preprocessing_->processImage(raw_image, &image);
    } else {
        image = raw_image;
    }
    /// Send the image to the derived class for processing
    processFrameImpl(image, frame.get());

    return frame;
}

}  // namespace aslam
