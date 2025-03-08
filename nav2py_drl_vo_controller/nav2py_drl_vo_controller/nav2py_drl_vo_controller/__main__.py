import yaml
import nav2py
import nav2py.interfaces
import rclpy
from rclpy.logging import get_logger
class nav2py_drl_vo_controller(nav2py.interfaces.nav2py_costmap_controller):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_callback('data', self._data_callback)
        self._register_callback('path', self._path_callback)
        
        self.logger = get_logger('nav2py_drl_vo_controller')
        self.frame_count = 0
        self.path = None
        
        self.logger.info("nav2py_drl_vo_controller initialized")
        
    def _path_callback(self, path_):
        """
        Process path data from C++ controller
        """
        try:
            if isinstance(path_, list) and len(path_) > 0:
                data_str = path_[0]
                if isinstance(data_str, bytes):
                    data_str = data_str.decode()
                
                self.path = yaml.safe_load(data_str)
                self.logger.info("Received path data")
                
                # You could extract the goal position from the path if needed
                if self.path and 'poses' in self.path and len(self.path['poses']) > 0:
                    last_pose = self.path['poses'][-1]['pose']
                    goal_x = last_pose['position']['x']
                    goal_y = last_pose['position']['y']
                    self.logger.info(f"Goal position: x={goal_x:.2f}, y={goal_y:.2f}")

        except Exception as e:
            import traceback
            self.logger.error(f"Error processing path data: {e}")
            self.logger.error(traceback.format_exc())

    def _data_callback(self, data):
        """
        Process data from C++ controller
        """
        try:
            self.frame_count += 1
            
            # Simple frame delimiter for logging
            frame_delimiter = "=" * 50
            self.logger.info(f"\n{frame_delimiter}")
            self.logger.info(f"PROCESSING FRAME {self.frame_count}")
            
            # Parse the incoming data
            if isinstance(data, list) and len(data) > 0:
                data_str = data[0]
                if isinstance(data_str, bytes):
                    data_str = data_str.decode()
                
                parsed_data = yaml.safe_load(data_str)
                self.logger.info("Data decoded successfully")
            else:
                if isinstance(data, bytes):
                    parsed_data = yaml.safe_load(data.decode())
                    self.logger.info("Data decoded from bytes")
                else:
                    self.logger.error(f"Unexpected data type: {type(data)}")
                    # Send a stop command if we can't parse the data
                    self._send_cmd_vel(0.0, 0.0)
                    return
            
            # Extract frame info
            frame_info = parsed_data.get('frame_info', {})
            frame_id = frame_info.get('id', 0)
            timestamp = frame_info.get('timestamp', 0)
            self.logger.info(f"Frame ID: {frame_id}, Timestamp: {timestamp}")
            
            # Extract robot pose
            robot_pose = parsed_data.get('robot_pose', {})
            position = robot_pose.get('position', {})
            x = position.get('x', 0.0)
            y = position.get('y', 0.0)
            self.logger.info(f"Robot position: x={x:.2f}, y={y:.2f}")
            # Extract velocity if available
            velocity = parsed_data.get('robot_velocity', {})
            linear_x = velocity.get('linear', {}).get('x', 0.0)
            angular_z = velocity.get('angular', {}).get('z', 0.0)
            self.logger.info(f"Current velocity: linear_x={linear_x:.2f}, angular_z={angular_z:.2f}")
            
            goal_pose = self.path['poses'][-1]['pose']
            # Determine control commands (this would be your control logic)
            # Here we just implement a simple demo
            
            # For demo: Just return the same linear velocity but rotate based on frame count
            # In a real controller, you would compute velocities based on your algorithm
            # demo_linear_x = 0.1  # Always move forward a bit
            # demo_angular_z = 0.1 * (self.frame_count % 10 - 5)  # Oscillate between -0.5 and 0.5
            from drl_vo_controller.controllerv2 import computeVelocityCommands
            path_data = self.path['poses']
            
            demo_linear_x,demo_angular_z = computeVelocityCommands(robot_pose,goal_pose,path_data)

            self.logger.info(f"Sending control commands: linear_x={demo_linear_x:.2f}, angular_z={demo_angular_z:.2f}")
                        
            # Send velocity commands back to the C++ controller
            self._send_cmd_vel(demo_linear_x, demo_angular_z)
            
            # Add closing delimiter
            self.logger.info(f"FRAME {self.frame_count} COMPLETED")
            self.logger.info(f"{frame_delimiter}")
                
        except Exception as e:
            import traceback
            self.logger.error(f"Error processing data: {e}")
            self.logger.error(traceback.format_exc())
            
            # Send a safe stop command in case of error
            self._send_cmd_vel(0.0, 0.0)
    
if __name__ == "__main__":
    nav2py.main(nav2py_drl_vo_controller)