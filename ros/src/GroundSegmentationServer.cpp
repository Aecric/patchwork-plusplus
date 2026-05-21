#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

// Patchwork++-ROS
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "GroundSegmentationServer.hpp"
#include "Utils.hpp"

namespace patchworkpp_ros {

using utils::EigenToPointCloud2;
using utils::GetTimestamps;
using utils::PointCloud2ToEigen;

// ---------------------------------------------------------------------------
// Parameter loaders
// ---------------------------------------------------------------------------

patchwork::Params GroundSegmentationServer::loadPlusplusParamsFromROS() {
  patchwork::Params params;

  params.sensor_height = declare_parameter<double>("sensor_height", params.sensor_height);
  params.num_iter      = declare_parameter<int>("num_iter", params.num_iter);
  params.num_lpr       = declare_parameter<int>("num_lpr", params.num_lpr);
  params.num_min_pts   = declare_parameter<int>("num_min_pts", params.num_min_pts);
  params.th_seeds      = declare_parameter<double>("th_seeds", params.th_seeds);

  params.th_dist    = declare_parameter<double>("th_dist", params.th_dist);
  params.th_seeds_v = declare_parameter<double>("th_seeds_v", params.th_seeds_v);
  params.th_dist_v  = declare_parameter<double>("th_dist_v", params.th_dist_v);

  params.max_range       = declare_parameter<double>("max_range", params.max_range);
  params.min_range       = declare_parameter<double>("min_range", params.min_range);
  params.uprightness_thr = declare_parameter<double>("uprightness_thr", params.uprightness_thr);

  params.verbose = get_parameter<bool>("verbose", params.verbose);

  // ToDo. Support intensity
  params.enable_RNR = false;

  return params;
}

patchwork::PatchworkParams GroundSegmentationServer::loadClassicParamsFromROS() {
  patchwork::PatchworkParams params;

  params.sensor_height = declare_parameter<double>("sensor_height", params.sensor_height);
  params.max_range     = declare_parameter<double>("max_range", params.max_range);
  params.min_range     = declare_parameter<double>("min_range", params.min_range);

  params.num_iter    = declare_parameter<int>("num_iter", params.num_iter);
  params.num_lpr     = declare_parameter<int>("num_lpr", params.num_lpr);
  params.num_min_pts = declare_parameter<int>("num_min_pts", params.num_min_pts);
  params.th_seeds    = declare_parameter<double>("th_seeds", params.th_seeds);
  params.th_dist     = declare_parameter<double>("th_dist", params.th_dist);

  params.uprightness_thr = declare_parameter<double>("uprightness_thr", params.uprightness_thr);

  params.adaptive_seed_selection_margin = declare_parameter<double>(
      "adaptive_seed_selection_margin", params.adaptive_seed_selection_margin);

  params.using_global_thr = declare_parameter<bool>("using_global_thr", params.using_global_thr);
  params.global_elevation_thr =
      declare_parameter<double>("global_elevation_thr", params.global_elevation_thr);

  params.ATAT_ON        = declare_parameter<bool>("ATAT_ON", params.ATAT_ON);
  params.max_h_for_ATAT = declare_parameter<double>("max_h_for_ATAT", params.max_h_for_ATAT);
  params.num_sectors_for_ATAT =
      declare_parameter<int>("num_sectors_for_ATAT", params.num_sectors_for_ATAT);
  params.noise_bound = declare_parameter<double>("noise_bound", params.noise_bound);

  params.verbose = declare_parameter<bool>("verbose", params.verbose);

  return params;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

GroundSegmentationServer::GroundSegmentationServer(const rclcpp::NodeOptions &options)
    : rclcpp::Node("patchworkpp_node", options) {
  base_frame_      = declare_parameter<std::string>("base_frame", base_frame_);
  tf_timeout_sec_  = declare_parameter<double>("tf_timeout_sec", tf_timeout_sec_);

  tf_buffer_   = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  const std::string algorithm = declare_parameter<std::string>("algorithm", "patchworkpp");

  // The plus-plus params are also reused as the seed for the ceiling-pass
  // params below, so always load them — even if the ground pass uses classic.
  patchwork::Params plusplus_params = loadPlusplusParamsFromROS();

  if (algorithm == "patchwork") {
    patchwork::PatchworkParams classic_params = loadClassicParamsFromROS();
    impl_ = std::make_unique<patchwork::PatchWork>(classic_params);
    RCLCPP_INFO(get_logger(), "Algorithm: patchwork (classic)");
  } else {
    impl_ = std::make_unique<patchwork::PatchWorkpp>(plusplus_params);
    RCLCPP_INFO(get_logger(), "Algorithm: patchworkpp (default)");
  }

  // Ceiling pass: a second Patchwork++ instance run on the flipped upper
  // half-space. The flip turns the ceiling into a virtual "ground", so the
  // same Patchwork++ ground-fit logic can extract it.
  enable_ceiling_ = declare_parameter<bool>("enable_ceiling", enable_ceiling_);
  ceiling_z_min_  = declare_parameter<double>("ceiling_z_min", ceiling_z_min_);
  if (enable_ceiling_) {
    patchwork::Params ceiling_params = plusplus_params;
    // The virtual sensor in the flipped frame sits below the ceiling at the
    // same distance the real sensor sits above it. Default ~1.0 m, override
    // via `ceiling_sensor_height`.
    ceiling_params.sensor_height =
        declare_parameter<double>("ceiling_sensor_height", 1.0);
    // RNR is tuned for ground noise below the sensor; disable in the flipped
    // pass since the geometry is inverted.
    ceiling_params.enable_RNR = false;
    ceiling_impl_ = std::make_unique<patchwork::PatchWorkpp>(ceiling_params);
    RCLCPP_INFO(get_logger(),
                "Ceiling pass enabled: z_min=%.2f, sensor_height(flipped)=%.2f",
                ceiling_z_min_,
                ceiling_params.sensor_height);
  }

  // Initialize subscribers
  pointcloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "pointcloud_topic",
      rclcpp::SensorDataQoS(),
      std::bind(&GroundSegmentationServer::EstimateGround, this, std::placeholders::_1));

  /*
   * We use the following QoS setting for reliable ground segmentation.
   * If you want to run Patchwork++ in real-time and real-world operation,
   * please change the QoS setting
   */
  //  rclcpp::QoS qos((rclcpp::SystemDefaultsQoS().keep_last(1).durability_volatile()));
  rclcpp::QoS qos(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default));
  qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  qos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);

  cloud_publisher_  = create_publisher<sensor_msgs::msg::PointCloud2>("/patchworkpp/cloud", qos);
  ground_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/patchworkpp/ground", qos);
  nonground_publisher_ =
      create_publisher<sensor_msgs::msg::PointCloud2>("/patchworkpp/nonground", qos);
  if (enable_ceiling_) {
    ceiling_publisher_ =
        create_publisher<sensor_msgs::msg::PointCloud2>("/patchworkpp/ceiling", qos);
    nonceiling_publisher_ =
        create_publisher<sensor_msgs::msg::PointCloud2>("/patchworkpp/nonceiling", qos);
  }

  // walls = points classified as neither ground nor ceiling.
  walls_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/patchworkpp/walls", qos);

  enable_scan_          = declare_parameter<bool>("enable_scan", enable_scan_);
  scan_angle_min_       = declare_parameter<double>("scan_angle_min", scan_angle_min_);
  scan_angle_max_       = declare_parameter<double>("scan_angle_max", scan_angle_max_);
  scan_angle_increment_ = declare_parameter<double>("scan_angle_increment", scan_angle_increment_);
  scan_range_min_       = declare_parameter<double>("scan_range_min", scan_range_min_);
  scan_range_max_       = declare_parameter<double>("scan_range_max", scan_range_max_);
  scan_z_min_           = declare_parameter<double>("scan_z_min", scan_z_min_);
  scan_z_max_           = declare_parameter<double>("scan_z_max", scan_z_max_);
  if (enable_scan_) {
    scan_publisher_ = create_publisher<sensor_msgs::msg::LaserScan>("/patchworkpp/scan", qos);
    RCLCPP_INFO(get_logger(),
                "LaserScan enabled: angle=[%.2f, %.2f] step=%.4f rad, z=[%.2f, %.2f]",
                scan_angle_min_,
                scan_angle_max_,
                scan_angle_increment_,
                scan_z_min_,
                scan_z_max_);
  }

  RCLCPP_INFO(this->get_logger(), "Patchwork++ ROS 2 node initialized");
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

void GroundSegmentationServer::EstimateGround(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg) {
  Eigen::MatrixXf cloud = patchworkpp_ros::utils::PointCloud2ToEigenMat(msg);

  // Patchwork++ assumes z-up. Tilt-mounted lidars (e.g. Mid360 立装) need the
  // cloud transformed into a level frame first. We honour the incoming
  // frame_id and walk TF to base_frame_; if they already match the lookup
  // returns identity. All downstream outputs are published in base_frame_.
  const std::string src_frame = patchworkpp_ros::utils::FixFrameId(msg->header.frame_id);
  if (src_frame != base_frame_) {
    geometry_msgs::msg::TransformStamped tf_msg;
    try {
      tf_msg = tf_buffer_->lookupTransform(base_frame_,
                                           src_frame,
                                           msg->header.stamp,
                                           tf2::durationFromSec(tf_timeout_sec_));
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN_THROTTLE(get_logger(),
                           *get_clock(),
                           2000,
                           "TF lookup %s -> %s failed: %s — skipping frame",
                           src_frame.c_str(),
                           base_frame_.c_str(),
                           ex.what());
      return;
    }

    Eigen::Quaternionf q(static_cast<float>(tf_msg.transform.rotation.w),
                         static_cast<float>(tf_msg.transform.rotation.x),
                         static_cast<float>(tf_msg.transform.rotation.y),
                         static_cast<float>(tf_msg.transform.rotation.z));
    const Eigen::Matrix3f R = q.toRotationMatrix();
    const Eigen::RowVector3f t(static_cast<float>(tf_msg.transform.translation.x),
                               static_cast<float>(tf_msg.transform.translation.y),
                               static_cast<float>(tf_msg.transform.translation.z));
    // Each row is a point; p' = p * R^T + t.
    cloud = (cloud * R.transpose()).rowwise() + t;
  }

  // Build a header in base_frame_ for every output.
  std_msgs::msg::Header out_header = msg->header;
  out_header.frame_id              = base_frame_;

  // Estimate ground
  std::visit([&](auto &impl) { impl->estimateGround(cloud); }, impl_);
  cloud_publisher_->publish(patchworkpp_ros::utils::EigenMatToPointCloud2(cloud, out_header));

  // Get ground / nonground (both as point clouds and as indices into `cloud`).
  Eigen::MatrixX3f ground    = std::visit([](auto &impl) { return impl->getGround(); }, impl_);
  Eigen::MatrixX3f nonground = std::visit([](auto &impl) { return impl->getNonground(); }, impl_);
  std::vector<int> ground_idx = std::visit(
      [](auto &impl) -> std::vector<int> {
        auto raw = impl->getGroundIndices();
        std::vector<int> out(static_cast<size_t>(raw.size()));
        for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<int>(raw[i]);
        return out;
      },
      impl_);
  double time_taken = std::visit([](auto &impl) { return impl->getTimeTaken(); }, impl_);
  (void)time_taken;  // available for debug logging if needed
  PublishClouds(ground, nonground, out_header);

  // Build wall mask: start as all-true, knock out ground, then ceiling.
  const Eigen::Index N = cloud.rows();
  std::vector<uint8_t> is_wall(static_cast<size_t>(N), 1);
  for (int idx : ground_idx) {
    if (idx >= 0 && idx < static_cast<int>(N)) is_wall[idx] = 0;
  }

  std::vector<int> ceiling_full_idx;
  if (enable_ceiling_ && ceiling_impl_) {
    EstimateCeiling(cloud, out_header, ceiling_full_idx);
    for (int idx : ceiling_full_idx) {
      if (idx >= 0 && idx < static_cast<int>(N)) is_wall[idx] = 0;
    }
  }

  // Compose the walls cloud.
  Eigen::Index n_walls = 0;
  for (Eigen::Index i = 0; i < N; ++i) {
    if (is_wall[static_cast<size_t>(i)]) ++n_walls;
  }
  Eigen::MatrixX3f walls(n_walls, 3);
  Eigen::Index w = 0;
  for (Eigen::Index i = 0; i < N; ++i) {
    if (is_wall[static_cast<size_t>(i)]) {
      walls.row(w++) = cloud.row(i).head<3>();
    }
  }
  walls_publisher_->publish(patchworkpp_ros::utils::EigenMatToPointCloud2(walls, out_header));

  if (enable_scan_ && scan_publisher_) {
    PublishWallsScan(walls, out_header);
  }
}

void GroundSegmentationServer::EstimateCeiling(const Eigen::MatrixXf &cloud,
                                               const std_msgs::msg::Header &header,
                                               std::vector<int> &ceiling_full_idx) {
  ceiling_full_idx.clear();

  // Select upper half-space rows: z >= ceiling_z_min_, then flip z so the
  // ceiling becomes a virtual ground for Patchwork++.
  const float z_cut       = static_cast<float>(ceiling_z_min_);
  const Eigen::Index n_in = cloud.rows();

  // First pass: count.
  Eigen::Index n_upper = 0;
  for (Eigen::Index i = 0; i < n_in; ++i) {
    if (cloud(i, 2) >= z_cut) ++n_upper;
  }

  if (n_upper == 0) {
    PublishCeilingClouds(Eigen::MatrixX3f(0, 3), Eigen::MatrixX3f(0, 3), header);
    return;
  }

  // Build upper subset and the mapping subset_idx -> full_idx.
  Eigen::MatrixXf upper(n_upper, cloud.cols());
  std::vector<int> upper_to_full(static_cast<size_t>(n_upper));
  Eigen::Index k = 0;
  for (Eigen::Index i = 0; i < n_in; ++i) {
    if (cloud(i, 2) >= z_cut) {
      upper.row(k)                            = cloud.row(i);
      upper(k, 2)                             = -upper(k, 2);  // flip z
      upper_to_full[static_cast<size_t>(k)]   = static_cast<int>(i);
      ++k;
    }
  }

  ceiling_impl_->estimateGround(upper);
  Eigen::MatrixX3f ceiling_flipped    = ceiling_impl_->getGround();
  Eigen::MatrixX3f nonceiling_flipped = ceiling_impl_->getNonground();
  Eigen::VectorXi ceiling_subset_idx  = ceiling_impl_->getGroundIndices();

  // Map ceiling indices in the upper subset back to indices in the full cloud.
  ceiling_full_idx.reserve(static_cast<size_t>(ceiling_subset_idx.size()));
  for (Eigen::Index i = 0; i < ceiling_subset_idx.size(); ++i) {
    const int sub = ceiling_subset_idx[i];
    if (sub >= 0 && sub < static_cast<int>(upper_to_full.size())) {
      ceiling_full_idx.push_back(upper_to_full[static_cast<size_t>(sub)]);
    }
  }

  // Flip z back to the original sensor frame before publishing.
  ceiling_flipped.col(2)    = -ceiling_flipped.col(2);
  nonceiling_flipped.col(2) = -nonceiling_flipped.col(2);

  PublishCeilingClouds(ceiling_flipped, nonceiling_flipped, header);
}

void GroundSegmentationServer::PublishWallsScan(const Eigen::MatrixX3f &walls,
                                                const std_msgs::msg::Header &header) {
  sensor_msgs::msg::LaserScan scan;
  scan.header           = header;
  scan.header.frame_id  = base_frame_;
  scan.angle_min        = static_cast<float>(scan_angle_min_);
  scan.angle_max        = static_cast<float>(scan_angle_max_);
  scan.angle_increment  = static_cast<float>(scan_angle_increment_);
  scan.time_increment   = 0.0f;
  scan.scan_time        = 0.0f;
  scan.range_min        = static_cast<float>(scan_range_min_);
  scan.range_max        = static_cast<float>(scan_range_max_);

  const double span = scan_angle_max_ - scan_angle_min_;
  const size_t n_bins =
      span > 0.0 && scan_angle_increment_ > 0.0
          ? static_cast<size_t>(std::ceil(span / scan_angle_increment_))
          : 0;
  if (n_bins == 0) return;

  const float inf = std::numeric_limits<float>::infinity();
  scan.ranges.assign(n_bins, inf);

  const float z_lo = static_cast<float>(scan_z_min_);
  const float z_hi = static_cast<float>(scan_z_max_);
  const float r_lo = static_cast<float>(scan_range_min_);
  const float r_hi = static_cast<float>(scan_range_max_);

  for (Eigen::Index i = 0; i < walls.rows(); ++i) {
    const float x = walls(i, 0);
    const float y = walls(i, 1);
    const float z = walls(i, 2);
    if (z < z_lo || z > z_hi) continue;
    const float r = std::hypot(x, y);
    if (r < r_lo || r > r_hi) continue;
    const float a = std::atan2(y, x);
    if (a < scan.angle_min || a >= scan.angle_max) continue;
    const size_t bin = static_cast<size_t>((a - scan.angle_min) / scan.angle_increment);
    if (bin < n_bins && r < scan.ranges[bin]) scan.ranges[bin] = r;
  }

  scan_publisher_->publish(scan);
}

void GroundSegmentationServer::PublishClouds(const Eigen::MatrixX3f &est_ground,
                                             const Eigen::MatrixX3f &est_nonground,
                                             const std_msgs::msg::Header header_msg) {
  std_msgs::msg::Header header = header_msg;
  header.frame_id              = base_frame_;
  ground_publisher_->publish(
      std::move(patchworkpp_ros::utils::EigenMatToPointCloud2(est_ground, header)));
  nonground_publisher_->publish(
      std::move(patchworkpp_ros::utils::EigenMatToPointCloud2(est_nonground, header)));
}

void GroundSegmentationServer::PublishCeilingClouds(const Eigen::MatrixX3f &est_ceiling,
                                                    const Eigen::MatrixX3f &est_nonceiling,
                                                    const std_msgs::msg::Header header_msg) {
  std_msgs::msg::Header header = header_msg;
  header.frame_id              = base_frame_;
  ceiling_publisher_->publish(
      std::move(patchworkpp_ros::utils::EigenMatToPointCloud2(est_ceiling, header)));
  nonceiling_publisher_->publish(
      std::move(patchworkpp_ros::utils::EigenMatToPointCloud2(est_nonceiling, header)));
}
}  // namespace patchworkpp_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(patchworkpp_ros::GroundSegmentationServer)
