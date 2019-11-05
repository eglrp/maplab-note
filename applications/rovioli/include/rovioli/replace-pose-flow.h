//
// Created by wya on 2019/10/17.
//

#ifndef ROVIOLI_REPLACE_POSE_FLOW_H
#define ROVIOLI_REPLACE_POSE_FLOW_H


#include <aslam/cameras/ncamera.h>
#include <message-flow/message-flow.h>
#include <vio-common/vio-types.h>

#include "rovioli/flow-topics.h"
#include "rovioli/rovio-estimate.h"
#include <iostream>

extern std::unordered_map<double,std::vector<double>> msckf_datas_map;
using namespace std;

namespace rovioli {

    class ReplacePoseFlow {
    public:

        typedef std::function<void(const RovioEstimate::ConstPtr&)>
                ReplacePosePublishFunction;

        void registerReplacePosePublishFunction(
                const ReplacePosePublishFunction& Replace_Pose_Publish_Function) {
            Replace_Pose_Publish_Function_ = Replace_Pose_Publish_Function;
        }

        void attachToMessageFlow(message_flow::MessageFlow* flow)
        {
            CHECK_NOTNULL(flow);
            static constexpr char kSubscriberNodeName[] = "ReplacePoseFlow";

            std::function<void(const RovioEstimate::ConstPtr&)>
                    Replace_Pose_Publish_Function = flow->registerPublisher<message_flow_topics::ROVIO_ESTIMATES>();

            registerReplacePosePublishFunction(Replace_Pose_Publish_Function);


            flow->registerSubscriber<message_flow_topics::SYNCED_NFRAMES_AND_IMU>(
                    kSubscriberNodeName, message_flow::DeliveryOptions(),
                    [this](const vio::SynchronizedNFrameImu::ConstPtr& nframe_imu)
                    {
                        CHECK(nframe_imu);

                        int64_t timestamp_nframe_ns = nframe_imu->nframe->getMinTimestampNanoseconds();

                        double timestamp_nframe_s = aslam::time::nanoSecondsToSeconds(timestamp_nframe_ns);

                        if(msckf_datas_map[timestamp_nframe_s])
                        {
                            cout<<"发现精确时间"<<endl;
                            RovioEstimate::Ptr pose_update = aligned_shared<RovioEstimate>();

                            pose_update->timestamp_s = timestamp_nframe_s;

                            vio::ViNodeState cur_vinode;

                            vio::ViNodeState msckf_vi_node;
                            std::vector<double> msckfdata  = msckf_datas_map[exact_time];

                            aslam::Position3D msckf_position(msckfdata[0],msckfdata[1],msckfdata[2]);


                            aslam::Quaternion msckf_q_A_B(msckfdata[6],msckfdata[3],msckfdata[4],msckfdata[5]);

                            aslam::Transformation msckf_T_M_I(msckf_q_A_B, msckf_position);

                            msckf_vi_node.set_T_M_I(msckf_T_M_I);

                            Eigen::Vector3d msckf_v_M_I(msckfdata[7],msckfdata[8],msckfdata[9]);

                            msckf_vi_node.set_v_M_I(msckf_v_M_I);

                            // Interpolate biases.
                            Eigen::Vector3d msckf_acc_bias(msckfdata[10],msckfdata[11],msckfdata[12]);
                            Eigen::Vector3d msckf_gyro_bias(msckfdata[13],msckfdata[14],msckfdata[15]);

                            msckf_vi_node.setAccBias(msckf_acc_bias);
                            msckf_vi_node.setGyroBias(msckf_gyro_bias);
                            vio_update->vinode = msckf_vi_node;


                        }
                        else if()
                        {
                            cout<<"没有发现精确时间，不publish"<<endl;
                        }



//                        vio_update->timestamp_ns = timestamp_nframe_ns;//使用视觉的时间
//                        vio_update->keyframe_and_imudata = oldest_unmatched_synced_nframe;//最老的这个状态
                        //
                        CHECK(Replace_Pose_Publish_Function_);
//                        Replace_Pose_Publish_Function_(vio_update);
                    });


        }

    private:
        ReplacePosePublishFunction Replace_Pose_Publish_Function_;
    };

}  // namespace rovioli



#endif //ROVIOLI_REPLACE_POSE_FLOW_H
