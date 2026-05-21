// Patchwork++ and Patchwork classic
#include "patchwork/patchwork.h"
#include "patchwork/patchworkpp.h"

// Standard library
#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace patchworkpp_ros {

class GroundSegmentationServer : public rclcpp::Node {
 public:
  /// GroundSegmentationServer constructor
  GroundSegmentationServer() = delete;
  explicit GroundSegmentationServer(const rclcpp::NodeOptions &options);

 private:
  /// Register new frame
  void EstimateGround(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);

  /// Run a second Patchwork++ pass on the upper half-space (z >= ceiling_z_min)
  /// with z flipped, treating the ceiling as a "ground" plane.
  /// Writes the indices (into `cloud`) classified as ceiling into `ceiling_full_idx`.
  void EstimateCeiling(const Eigen::MatrixXf &cloud,
                       const std_msgs::msg::Header &header,
                       std::vector<int> &ceiling_full_idx);

  /// Project the wall point cloud (rows of `walls` are points in base_frame_)
  /// to a 2D LaserScan around the base_frame_ origin.
  void PublishWallsScan(const Eigen::MatrixX3f &walls, const std_msgs::msg::Header &header);

  /// Stream the ground / non-ground point clouds for visualization
  void PublishClouds(const Eigen::MatrixX3f &est_ground,
                     const Eigen::MatrixX3f &est_nonground,
                     const std_msgs::msg::Header header_msg);

  /// Stream the ceiling / non-ceiling point clouds for visualization
  void PublishCeilingClouds(const Eigen::MatrixX3f &est_ceiling,
                            const Eigen::MatrixX3f &est_nonceiling,
                            const std_msgs::msg::Header header_msg);

  /// Parameter loaders — only the selected algorithm's loader is called
  patchwork::Params loadPlusplusParamsFromROS();
  patchwork::PatchworkParams loadClassicParamsFromROS();

 private:
  /// Data subscribers.
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;

  /// Data publishers.
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr nonground_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ceiling_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr nonceiling_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr walls_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_publisher_;

  /// Algorithm implementation (patchworkpp or patchwork classic)
  using ImplVariant =
      std::variant<std::unique_ptr<patchwork::PatchWorkpp>, std::unique_ptr<patchwork::PatchWork>>;
  ImplVariant impl_;

  /// Secondary Patchwork++ instance for ceiling detection on the flipped upper half-space.
  std::unique_ptr<patchwork::PatchWorkpp> ceiling_impl_;

  std::string base_frame_{"base_link"};

  /// Ceiling-pass configuration.
  bool enable_ceiling_{true};
  double ceiling_z_min_{0.3};  // cut plane: include points with z >= this value (base_frame_)

  /// LaserScan projection (operates on the walls cloud, base_frame_, z-up).
  bool enable_scan_{true};
  double scan_angle_min_{-M_PI};
  double scan_angle_max_{M_PI};
  double scan_angle_increment_{M_PI / 360.0};  // 0.5°
  double scan_range_min_{0.1};
  double scan_range_max_{80.0};
  double scan_z_min_{0.2};
  double scan_z_max_{2.0};

  /// TF for transforming the input cloud from its native sensor frame to base_frame_.
  /// Patchwork++ requires z-up; for tilt-mounted lidars (e.g. Mid360 立装) running
  /// segmentation in the raw sensor frame produces wrong results.
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  double tf_timeout_sec_{0.1};
};

}  // namespace patchworkpp_ros
