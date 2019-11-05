#include "rovioli/map-builder-flow.h"

#include <functional>

#include <landmark-triangulation/landmark-triangulation.h>
#include <localization-summary-map/localization-summary-map-creation.h>
#include <localization-summary-map/localization-summary-map.h>
#include <maplab-common/file-logger.h>
#include <maplab-common/map-manager-config.h>
#include <mapping-workflows-plugin/localization-map-creation.h>
#include <vi-map-helpers/vi-map-landmark-quality-evaluation.h>
#include <vi-map-helpers/vi-map-manipulation.h>
#include <vi-map/vi-map-serialization.h>
#include <vi-map/vi-map.h>
#include <visualization/viwls-graph-plotter.h>

#include <map>

DEFINE_double(
    localization_map_keep_landmark_fraction, 0.0,
    "Fraction of landmarks to keep when creating a localization summary map.");
DECLARE_bool(rovioli_visualize_map);

using namespace std;

namespace rovioli {

MapBuilderFlow::MapBuilderFlow(
    const std::shared_ptr<aslam::NCamera>& n_camera, vi_map::Imu::UniquePtr imu,
    const std::string& save_map_folder)
    : map_with_mutex_(aligned_shared<VIMapWithMutex>()),
      mapping_terminated_(false),
      stream_map_builder_(n_camera, std::move(imu), &map_with_mutex_->vi_map)
      {
  if (!save_map_folder.empty()) {
    VLOG(1) << "Set VIMap folder to: " << save_map_folder;
    map_with_mutex_->vi_map.setMapFolder(save_map_folder);
  }
}

void MapBuilderFlow::attachToMessageFlow(message_flow::MessageFlow* flow)
{
    CHECK_NOTNULL(flow);
    static constexpr char kSubscriberNodeName[] = "MapBuilderFlow";
    std::function<void(const VIMapWithMutex::ConstPtr&)> map_publish_function =//建立vi-map接收器
            flow->registerPublisher<message_flow_topics::RAW_VIMAP>();
    CHECK(map_publish_function);
    flow->registerSubscriber<message_flow_topics::VIO_UPDATES>(//订阅VIO_UPDATES
            kSubscriberNodeName, message_flow::DeliveryOptions(),
            [this, map_publish_function](const vio::VioUpdate::ConstPtr& vio_update)
            {
                CHECK(vio_update != nullptr);
                {
                    std::lock_guard<std::mutex> lock(map_with_mutex_->mutex);
                    if (mapping_terminated_) {
                        return;
                    }
                    // WARNING: The tracker is updating the track information in the
                    // current and previous frame; therefore the most recent VisualNFrame
                    // added to the map might still be modified.
                    //跟踪器更新当前帧和上一帧的跟踪信息;因此，添加到地图上的最新VisualNFrame仍然可能被修改,所以不去进行深拷贝
                    //这里会增加新的节点，新的边
                    constexpr bool kDeepCopyNFrame = false;
                    stream_map_builder_.apply(*vio_update, kDeepCopyNFrame);
                }
                map_publish_function(map_with_mutex_);
            });

    std::function<void(const vio::VioUpdate::ConstPtr&)>
            vio_update_builder_publisher =//构建一个vio状态更新发布器
            flow->registerPublisher<message_flow_topics::VIO_UPDATES>();
    vio_update_builder_.registerVioUpdatePublishFunction(
            vio_update_builder_publisher);
    flow->registerSubscriber<message_flow_topics::TRACKED_NFRAMES_AND_IMU>(//订阅TRACKED_NFRAMES_AND_IMU
            kSubscriberNodeName, message_flow::DeliveryOptions(),
            [this](
                    const vio::SynchronizedNFrameImu::ConstPtr& synchronized_nframe_imu) {
                if (mapping_terminated_) {//如果建图已经终止了，就返回
                    return;
                }
                vio_update_builder_.processSynchronizedNFrameImu(//这里是接收视觉信息
                        synchronized_nframe_imu);
            });
    flow->registerSubscriber<message_flow_topics::ROVIO_ESTIMATES>(
            kSubscriberNodeName, message_flow::DeliveryOptions(),
            [this](const RovioEstimate::ConstPtr& rovio_estimate)
            {
                if (mapping_terminated_) {
                    return;
                }
                vio_update_builder_.processRovioEstimate(rovio_estimate);//接收rovio输出的pvq,bias
            });

    //findMatchAndPublish负责同步视觉的rovio发出的状态，都整合到vio::VioUpdate::ptr中，如果时间戳不对齐，要进行插值
    flow->registerSubscriber<message_flow_topics::LOCALIZATION_RESULT>(
            kSubscriberNodeName, message_flow::DeliveryOptions(),
            std::bind(
                    &VioUpdateBuilder::processLocalizationResult, &vio_update_builder_,
                    std::placeholders::_1));
}


//////-------------

    Eigen::Matrix3d YawToR(const double yaw)
    {
        Eigen::Matrix3d ans;

        Eigen::Vector3d eulerAngle(yaw,0,0);

        Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(eulerAngle(2),Eigen::Vector3d::UnitX()));
        Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(eulerAngle(1),Eigen::Vector3d::UnitY()));
        Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(eulerAngle(0),Eigen::Vector3d::UnitZ()));


        ans=yawAngle*pitchAngle*rollAngle;

        return ans;
    }


    typedef Eigen::Matrix<double, 3, 1> Vector3;
    typedef Vector3 Imaginary;


    bool isLessThenEpsilons4thRoot(double x){
        static const double epsilon4thRoot = pow(std::numeric_limits<double>::epsilon(), 1.0/4.0);
        return x < epsilon4thRoot;
    }

    double arcSinXOverX(double x) {
        if(isLessThenEpsilons4thRoot(fabs(x))){
            return double(1.0) + x * x * double(1.0/6.0);
        }
        return asin(x) / x;
    }



    Vector3 log(const Eigen::Quaterniond & q) {
        const Eigen::Matrix<double, 3, 1> a = Vector3(q.x(),q.y(),q.z());
        const double na = a.norm();
        const double eta = q.w();
        double scale;
        if(fabs(eta) < na){ // use eta because it is more precise than na to calculate the scale. No singularities here.
            // check sign of eta so that we can be sure that log(-q) = log(q)
            if (eta >= 0) {
                scale = acos(eta) / na;
            } else {
                scale = -acos(-eta) / na;
            }
        } else {
            /*
             * In this case more precision is in na than in eta so lets use na only to calculate the scale:
             *
             * assume first eta > 0 and 1 > na > 0.
             *               u = asin (na) / na  (this implies u in [1, pi/2], because na i in [0, 1]
             *    sin (u * na) = na
             *  sin^2 (u * na) = na^2
             *  cos^2 (u * na) = 1 - na^2
             *                              (1 = ||q|| = eta^2 + na^2)
             *    cos^2 (u * na) = eta^2
             *                              (eta > 0,  u * na = asin(na) in [0, pi/2] => cos(u * na) >= 0 )
             *      cos (u * na) = eta
             *                              (u * na in [ 0, pi/2] )
             *                 u = acos (eta) / na
             *
             * So the for eta > 0 it is acos(eta) / na == asin(na) / na.
             * From some geometric considerations (mirror the setting at the hyper plane q==0) it follows for eta < 0 that (pi - asin(na)) / na = acos(eta) / na.
             */
            if(eta > 0) {
                // For asin(na)/ na the singularity na == 0 can be removed. We can ask (e.g. Wolfram alpha) for its series expansion at na = 0. And that is done in the following function.
                scale = arcSinXOverX(na);
            } else {
                // the negative is here so that log(-q) == log(q)
                scale = arcSinXOverX(na);
            }
        }
        return a * (double(2.0) * scale);
    }



    Eigen::Quaterniond exp(const Vector3& dx) {
        // Method of implementing this function that is accurate to numerical precision from
        // Grassia, F. S. (1998). Practical parameterization of rotations using the exponential map. journal of graphics, gpu, and game tools, 3(3):29–48.
        double theta = dx.norm();
        // na is 1/theta sin(theta/2)
        double na;
        if(isLessThenEpsilons4thRoot(theta)){
            static const double one_over_48 = 1.0/48.0;
            na = 0.5 + (theta * theta) * one_over_48;
        } else {
            na = sin(theta*0.5) / theta;
        }
        double ct = cos(theta*0.5);
        return Eigen::Quaterniond(ct,dx[0]*na,dx[1]*na,dx[2]*na);
    }

    void interpolateRotation(
            const double t1, const Eigen::Quaterniond & q_A_B1, const double t2,
            const Eigen::Quaterniond& q_A_B2, const double t_interpolated,
            Eigen::Quaterniond & q_A_B_interpolated) {


        q_A_B_interpolated.setIdentity();
        const Eigen::Quaterniond q_B1_B2 = q_A_B1.inverse() * q_A_B2;
        const double theta = (t_interpolated - t1) / static_cast<double>(t2 - t1);
        q_A_B_interpolated = q_A_B1 * exp(theta * log(q_B1_B2));
    }


    void GetAllLidarData(map<double,Eigen::Quaterniond> &lidar_yaw_data,map<double,Eigen::Vector2d>& lidar_p)
    {
        lidar_yaw_data.clear();
        std::string filename = "/home/wya/10.31/robot_pose.txt";
        std::ifstream foutC(filename);
        if(!foutC.is_open())
        {
            std::cout<<" 无法读取"<<std::endl;

        }

        //顺序time,px,py,pz,qx,qy,qz,qw,vx,vy,vz,ax,ay,az,gx,gy,gz;

        Eigen::Quaterniond last_q;

        double last_time;

        bool isfirst = true;

        string line;

        long double time_stamp,time_stamp_pvq;
        string stmp,stmp_pvq;
        double odom_x,odom_y,odom_yaw;
        double pvq_x,pvq_y,pvq_yaw,pvq_q1,pvq_q2,pvq_q3,pvq_q4;
        double laser_x,laser_y,laser_yaw;
        double begin_time;

        vector<double> vtime_stamp;
        vector<double> vodom_x,vodom_y,vodom_yaw;
        vector<double> vlaser_x,vlaser_y,vlaser_yaw;
        vector<double> verror_x,verror_y,verror_xy,verror_yaw;
        vector<double> myvio_x,myvio_y,myvio_yaw;
        vector<double> total_pvq_x,total_pvq_y,total_pvq_yaw;

        double last_yaw;
        double odom_1,odom_2,odom_3,robot_1,robot_2,robot_3;



        while(getline(foutC,line))
        {
            istringstream ss(line);

            //解析数据
            ss >> time_stamp;

            // odom :
            ss >> stmp;
            ss >> stmp; sscanf(stmp.c_str(),"%lf,",&odom_x);
            ss >> stmp; sscanf(stmp.c_str(),"%lf,",&odom_y);
            ss >> stmp; sscanf(stmp.c_str(),"%lf,",&odom_yaw);

            // robot :
            ss >> stmp;
            ss >> stmp; sscanf(stmp.c_str(),"%lf,",&laser_x);
            ss >> stmp; sscanf(stmp.c_str(),"%lf,",&laser_y);
            ss >> stmp; sscanf(stmp.c_str(),"%lf.",&laser_yaw);

            vtime_stamp.push_back(time_stamp);

            vlaser_x.push_back(laser_x);

            vlaser_y.push_back(laser_y);
            vlaser_yaw.push_back(laser_yaw);

            if(time_stamp < 0.1)
            {

            }
            else
            {

                lidar_yaw_data.insert(make_pair(time_stamp,Eigen::Quaterniond(YawToR(laser_yaw))));
                lidar_p.insert(make_pair(time_stamp,Eigen::Vector2d(laser_x,laser_y)));
            }

        }
    }



    template <typename Time, typename InterpolateType>
    void linerarInterpolation(
            const Time t1, const InterpolateType& x1, const Time t2,
            const InterpolateType& x2, const Time t_interpolated,
            InterpolateType & x_interpolated) {
        assert(t1 < t2);
        assert(t1 <= t_interpolated);
        assert(t_interpolated <= t2);
        x_interpolated = x1 + (x2 - x1) / (t2 - t1) * (t_interpolated - t1);
    }



//////-------------

//保存并优化
void MapBuilderFlow::saveMapAndOptionallyOptimize(
    const std::string& path, const bool overwrite_existing_map,
    const bool process_to_localization_map)
{
    CHECK(!path.empty());
    CHECK(map_with_mutex_);

    std::lock_guard<std::mutex> lock(map_with_mutex_->mutex);
    mapping_terminated_ = true;//建图终止

    // Early exit if the map is empty.
    if (map_with_mutex_->vi_map.numVertices() < 3u)
    {
        LOG(WARNING) << "Map is empty; nothing will be saved.";
        return;
    }

    visualization::ViwlsGraphRvizPlotter::UniquePtr plotter;
    if (FLAGS_rovioli_visualize_map)
    {
        plotter = aligned_unique<visualization::ViwlsGraphRvizPlotter>();
    }







    {
        VLOG(1) << "Initializing landmarks of created map.";
        // There should only be one mission in the map.
        vi_map::MissionIdList mission_ids;//得到当前所有的任务id
        map_with_mutex_->vi_map.getAllMissionIds(&mission_ids);
        CHECK_EQ(mission_ids.size(), 1u);
        const vi_map::MissionId& id_of_first_mission = mission_ids.front();//第一个任务

        vi_map_helpers::VIMapManipulation manipulation(&map_with_mutex_->vi_map);//初始化VIMapManipulation
        manipulation.initializeLandmarksFromUnusedFeatureTracksOfMission(//重新三角化之前的预备准备
                id_of_first_mission);

        landmark_triangulation::retriangulateLandmarksOfMission(//重新三角化
                id_of_first_mission, &map_with_mutex_->vi_map);


        /////导入Lidar的pose

        map<double,Eigen::Quaterniond> lidar_q;
        map<double,Eigen::Vector2d> lidar_p;

        GetAllLidarData(lidar_q,lidar_p);

        Eigen::Matrix4d Tl_middle;

        Tl_middle = Eigen::Matrix4d::Identity();

        Tl_middle(0,3) = -0.723;

        Eigen::Matrix4d Tlc;

        Tlc<< -2.0274126863009417e-02, 4.0243969776265615e-01,9.1522196730882954e-01, -3.2843535777714782e-02,
                -9.9890691107931207e-01, -4.6716899311359507e-02,-1.5856597111128033e-03, 5.4736881512303937e-02,
                4.2118200079416433e-02, -9.1425369618253893e-01,4.0294693973107193e-01, 7.5861307451623006e-02,
                0., 0., 0., 1;

        pose_graph::VertexIdList relevant_vertex_ids;
        map_with_mutex_->vi_map.getAllVertexIdsInMissionAlongGraph(id_of_first_mission, &relevant_vertex_ids);//通过图，得到这个任务的所有的节点

        for (int i = 0; i <relevant_vertex_ids.size() ; ++i)
        {
            const pose_graph::VertexId& observer_id = relevant_vertex_ids[i];//得到当前观测点对应的节点id
            CHECK(map_with_mutex_->vi_map.hasVertex(observer_id))
            << "Observer " << observer_id << " of store landmark ";

            vi_map::Vertex& cur_Vertex = map_with_mutex_->vi_map.getVertex(observer_id);

            Eigen::Matrix4d Tcb = cur_Vertex.getNCameras()->get_T_C_B(0).getTransformationMatrix();

            double time = cur_Vertex.getMinTimestampNanoseconds() * kNanosecondsToSeconds;


            Eigen::Quaterniond inter_lidarq;
            Eigen::Vector2d inter_lidarp;

            if(lidar_q.count(time))
            {
                inter_lidarq = lidar_q.find(time)->second;
                inter_lidarp = lidar_p.find(time)->second;
            }
            else
            {

                auto first_lidar_Q = lidar_q.lower_bound(time);

                double time_after_q = first_lidar_Q->first;
                Eigen::Quaterniond q_after = first_lidar_Q->second;


                first_lidar_Q--;
                double time_before_q = first_lidar_Q->first;
                Eigen::Quaterniond q_befor = first_lidar_Q->second;

                interpolateRotation(time_before_q, q_befor, time_after_q, q_after, time, inter_lidarq);


                auto first_lidar_p = lidar_p.lower_bound(time);
                double time_after_p = first_lidar_p->first;
                Eigen::Vector2d p_after =  first_lidar_p->second;

                first_lidar_p--;
                double time_before_p = first_lidar_p->first;
                Eigen::Vector2d p_befor =  first_lidar_p->second;


                linerarInterpolation(time_before_p, p_befor, time_after_p, p_after, time, inter_lidarp);
            }

            Eigen::Quaterniond Qw_middle = inter_lidarq;
            Eigen::Vector3d tw_middle{inter_lidarp[0],inter_lidarp[1],0};

//        std::cout<<"tw_middle"<<tw_middle.transpose()<<std::endl;

            Eigen::Matrix4d Tw_middle = Eigen::Matrix4d::Identity();
            Tw_middle.block<3, 3>(0, 0) = Qw_middle.toRotationMatrix();
            Tw_middle.block<3, 1>(0, 3) = tw_middle;

            Eigen::Matrix4d Twl_c = Tw_middle * Tl_middle.inverse() * Tlc;//这个只是雷达坐标系下的视觉位姿

            Eigen::Matrix4d Twl_b = Twl_c * Tcb;//这个只是雷达坐标系下的视觉位姿


            map_with_mutex_->vi_map.setVertex_T_M_I(observer_id,Eigen::Quaterniond(Twl_b.block<3,3>(0,0)),Eigen::Vector3d{Twl_b.block<3,1>(0,3)});

            //得到节点id中哪一帧看到的这个地图点，

        }

    }






    backend::SaveConfig save_config;
    save_config.overwrite_existing_files = overwrite_existing_map;

    //输入map，保存的路径，保存设置
    vi_map::serialization::saveMapToFolder(
            path, save_config, &map_with_mutex_->vi_map);
    LOG(INFO) << "Raw VI-map saved to: " << path;

    if (process_to_localization_map) {
        LOG(INFO) << "Map is being processed into a localization map... "
                  << "please wait.";

        map_sparsification::KeyframingHeuristicsOptions keyframe_options =
                map_sparsification::KeyframingHeuristicsOptions::initializeFromGFlags();
        constexpr bool kInitializeLandmarks = false;
        mapping_workflows_plugin::processVIMapToLocalizationMap(
                kInitializeLandmarks, keyframe_options, &map_with_mutex_->vi_map,
                plotter.get());

        // Create localization summary map that has pre-projected descriptors.
        std::unique_ptr<summary_map::LocalizationSummaryMap> localization_map(
                new summary_map::LocalizationSummaryMap);
        vi_map_helpers::evaluateLandmarkQuality(&map_with_mutex_->vi_map);

        if (FLAGS_localization_map_keep_landmark_fraction > 0.0) {
            summary_map::createLocalizationSummaryMapForSummarizedLandmarks(
                    map_with_mutex_->vi_map,
                    FLAGS_localization_map_keep_landmark_fraction,
                    localization_map.get());
        } else {
            summary_map::createLocalizationSummaryMapForWellConstrainedLandmarks(
                    map_with_mutex_->vi_map, localization_map.get());
        }

        std::string localization_map_path = path + "_localization";
        localization_map->saveToFolder(localization_map_path, save_config);
        LOG(INFO) << "Localization summary map saved to: " << localization_map_path;
    }
}
}  // namespace rovioli
