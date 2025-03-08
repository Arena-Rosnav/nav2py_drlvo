/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Template Controller for nav2py
 */

 #include <algorithm>
 #include <string>
 #include <memory>
 
 #include "nav2_core/exceptions.hpp"
 #include "nav2_util/node_utils.hpp"
 #include "nav2py_drl_vo_controller/template_controller.hpp"
 #include "nav2_util/geometry_utils.hpp"
 #include "ament_index_cpp/get_package_share_directory.hpp"
 
 using nav2_util::declare_parameter_if_not_declared;
 
 namespace nav2py_drl_vo_controller
 {
 
 void TemplateController::configure(
   const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
   std::string name, const std::shared_ptr<tf2_ros::Buffer> tf,
   const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
 {
   node_ = parent;
 
   auto node = node_.lock();
 
   costmap_ros_ = costmap_ros;
   tf_ = tf;
   plugin_name_ = name;
   logger_ = node->get_logger();
   clock_ = node->get_clock();
 
   // Set transform tolerance parameter
   double transform_tolerance;
   nav2_util::declare_parameter_if_not_declared(
     node, plugin_name_ + ".transform_tolerance", rclcpp::ParameterValue(0.1));
   node->get_parameter(plugin_name_ + ".transform_tolerance", transform_tolerance);
   transform_tolerance_ = rclcpp::Duration::from_seconds(transform_tolerance);
 
   // Initialize nav2py
   std::string nav2py_script = ament_index_cpp::get_package_share_directory("nav2py_drl_vo_controller") + "/../../lib/nav2py_drl_vo_controller/nav2py_run" ;
   nav2py_bootstrap(nav2py_script +
     " --host 127.0.0.1" +
     " --port 0");
 
   // Create publisher for global plan
   global_pub_ = node->create_publisher<nav_msgs::msg::Path>("received_global_plan", 1);
   
   RCLCPP_INFO(
     logger_,
     "Configured controller: %s of type nav2py_drl_vo_controller::TemplateController",
     plugin_name_.c_str());
 }
 
 void TemplateController::sendData(
   const geometry_msgs::msg::PoseStamped & pose,
   const geometry_msgs::msg::Twist & velocity)
 {
   static int frame_count = 0;
   frame_count++;
   
   // Add frame delimiter for logs
   std::string frame_delimiter(50, '=');
   RCLCPP_INFO(logger_, "\n%s", frame_delimiter.c_str());
   RCLCPP_INFO(logger_, "===== SENDING FRAME %d =====", frame_count);
   
   // Create a structured data message
   std::stringstream ss;
   ss << "frame_info:\n";
   ss << "  id: " << frame_count << "\n";
   ss << "  timestamp: " << clock_->now().nanoseconds() << "\n";
   
   // Add robot pose
   ss << "robot_pose:\n";
   ss << "  position:\n";
   ss << "    x: " << pose.pose.position.x << "\n";
   ss << "    y: " << pose.pose.position.y << "\n";
   ss << "    z: " << pose.pose.position.z << "\n";
   ss << "  orientation:\n";
   ss << "    x: " << pose.pose.orientation.x << "\n";
   ss << "    y: " << pose.pose.orientation.y << "\n";
   ss << "    z: " << pose.pose.orientation.z << "\n";
   ss << "    w: " << pose.pose.orientation.w << "\n";
   
   // Add robot velocity
   ss << "robot_velocity:\n";
   ss << "  linear:\n";
   ss << "    x: " << velocity.linear.x << "\n";
   ss << "    y: " << velocity.linear.y << "\n";
   ss << "    z: " << velocity.linear.z << "\n";
   ss << "  angular:\n";
   ss << "    x: " << velocity.angular.x << "\n";
   ss << "    y: " << velocity.angular.y << "\n";
   ss << "    z: " << velocity.angular.z << "\n";
   
   // Send the data
   nav2py_send("data", {ss.str()});
   
   RCLCPP_INFO(logger_, "Data sent to Python side");
 }
 
 void TemplateController::cleanup()
 {
   RCLCPP_INFO(
     logger_,
     "Cleaning up controller: %s",
     plugin_name_.c_str());
   nav2py_cleanup();
   global_pub_.reset();
 }
 
 void TemplateController::activate()
 {
   RCLCPP_INFO(
     logger_,
     "Activating controller: %s",
     plugin_name_.c_str());
   global_pub_->on_activate();
 }
 
 void TemplateController::deactivate()
 {
   RCLCPP_INFO(
     logger_,
     "Deactivating controller: %s",
     plugin_name_.c_str());
   global_pub_->on_deactivate();
 }
 
 void TemplateController::setSpeedLimit(const double& speed_limit, const bool& percentage)
 {
   // Empty implementation for interface compatibility
   (void) speed_limit;
   (void) percentage;
 }
 
 geometry_msgs::msg::TwistStamped TemplateController::computeVelocityCommands(
   const geometry_msgs::msg::PoseStamped & pose,
   const geometry_msgs::msg::Twist & velocity,
   nav2_core::GoalChecker * goal_checker)
 {
   (void)goal_checker;
   
   // Simple implementation that just sends data to Python and waits for response
   try {
     sendData(pose, velocity);
   } catch (const std::exception& e) {
     RCLCPP_ERROR(
       logger_,
       "Error sending data: %s", e.what());
   }
   
   geometry_msgs::msg::TwistStamped cmd_vel;
   cmd_vel.header.frame_id = pose.header.frame_id;
   cmd_vel.header.stamp = clock_->now();
   
   try {
     RCLCPP_INFO(logger_, "Waiting for velocity command from Python...");
     cmd_vel.twist = wait_for_cmd_vel();
     
     RCLCPP_INFO(
       logger_, 
       "Received velocity command: linear_x=%.2f, angular_z=%.2f", 
       cmd_vel.twist.linear.x, cmd_vel.twist.angular.z);
   } catch (const std::exception& e) {
     RCLCPP_ERROR(
       logger_,
       "Error receiving velocity command: %s", e.what());
     
     // Default to stop if there's an error
     cmd_vel.twist.linear.x = 0.0;
     cmd_vel.twist.angular.z = 0.0;
   }
 
   return cmd_vel;
 }
 
 void TemplateController::setPlan(const nav_msgs::msg::Path & path)
 {
   global_plan_ = path;
   global_pub_->publish(path);
   
   try {
     std::string path_yaml = nav_msgs::msg::to_yaml(path, true);
     nav2py_send("path", {path_yaml});
     RCLCPP_INFO(logger_, "Sent path data to Python controller");
   } catch (const std::exception& e) {
     RCLCPP_ERROR(
       logger_,
       "Error sending path: %s", e.what());
   }
 }
 
 }  // namespace nav2py_drl_vo_controller
 
 // Register this controller as a nav2_core plugin
 PLUGINLIB_EXPORT_CLASS(nav2py_drl_vo_controller::TemplateController, nav2_core::Controller)