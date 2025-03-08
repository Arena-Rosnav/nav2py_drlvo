#!/usr/bin/env python3

from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import LaserScan
import numpy as np
import numpy.matlib
from stable_baselines3 import PPO
import rclpy
from rclpy.node import Node
import rclpy.logging
import math 

import sys
from . import custom_cnn_full
import numpy as np
if not hasattr(np, 'float'):
    np.float = float

sys.modules['custom_cnn_full'] = custom_cnn_full
logger = rclpy.logging.get_logger('controller_server')

import tf_transformations 
model = PPO.load('/home/kien/colcon_ws/src/nav2_pyif/drl_vo_controller/model/drl_vo.zip', device='cpu')
goal_pose = PoseStamped()       
position_all = []             

#Store last 10 scans 
scan_buffer = []         

def find_closest_point(path, x, seg=-1):
    """
    Given a global path (as a list of dicts with a 'pose' key) and a point x (np.array([x, y])),
    find the closest point on the path.
    """
    pt_min = np.array([np.nan, np.nan])
    dist_min = np.inf
    seg_min = -1

    if seg == -1:
        for i in range(len(path) - 1):
            pt, dist, s = find_closest_point(path, x, i)
            if dist < dist_min:
                pt_min = pt
                dist_min = dist
                seg_min = s
        return pt_min, dist_min, seg_min
    else:
        p_start = np.array([path[seg]['pose']['position']['x'], path[seg]['pose']['position']['y']])
        p_end   = np.array([path[seg+1]['pose']['position']['x'], path[seg+1]['pose']['position']['y']])
        v = p_end - p_start
        length_seg = np.linalg.norm(v)
        if length_seg == 0:
            return p_start, np.linalg.norm(x - p_start), seg
        v = v / length_seg
        dist_projected = np.dot(x - p_start, v)
        if dist_projected < 0.:
            pt_min = p_start
        elif dist_projected > length_seg:
            pt_min = p_end
        else:
            pt_min = p_start + dist_projected * v
        dist_min = np.linalg.norm(pt_min - x)
        return pt_min, dist_min, seg


def pure_pursuit_subgoal(path, pose, lookahead=2.0):
    """
    Compute the subgoal (to be used as cnn_goal) given:
      - path: a global plan as a list of dictionaries (each with a 'pose' key)
      - pose: the current robot pose (as a dictionary with 'position' and 'orientation' keys)
      - lookahead: the lookahead distance.
    Returns:
      - goal_local: a 2-element numpy array representing the subgoal coordinates in the robot's local frame.
    """
    # Extract the robot's position and orientation from the pose
    x = np.array([pose["position"]["x"], pose["position"]["y"]])
    quat = [pose["orientation"]["x"], pose["orientation"]["y"],
            pose["orientation"]["z"], pose["orientation"]["w"]]
    (_, _, theta) = tf_transformations.euler_from_quaternion(quat)

    # Find the closest point on the path to the robot's current position
    pt, dist, seg = find_closest_point(path, x)

    # Compute the goal point on the path based on lookahead distance
    if dist > lookahead:
        goal = pt
    else:
        seg_max = len(path) - 2
        p_end = np.array([path[seg+1]['pose']['position']['x'],
                          path[seg+1]['pose']['position']['y']])
        dist_end = np.linalg.norm(x - p_end)
        current_seg = seg
        while dist_end < lookahead and current_seg < seg_max:
            current_seg += 1
            p_end = np.array([path[current_seg+1]['pose']['position']['x'],
                              path[current_seg+1]['pose']['position']['y']])
            dist_end = np.linalg.norm(x - p_end)
        if dist_end < lookahead:
            # Use the very end of the path if the lookahead circle contains the end
            goal = np.array([path[seg_max+1]['pose']['position']['x'],
                             path[seg_max+1]['pose']['position']['y']])
        else:
            # Find intersection along the segment
            pt_seg, _, seg_used = find_closest_point(path, x, current_seg)
            p_start = np.array([path[seg_used]['pose']['position']['x'],
                                path[seg_used]['pose']['position']['y']])
            p_end = np.array([path[seg_used+1]['pose']['position']['x'],
                              path[seg_used+1]['pose']['position']['y']])
            v = p_end - p_start
            length_seg = np.linalg.norm(v)
            if length_seg != 0:
                v = v / length_seg
                dist_projected = np.dot(x - pt_seg, v)
                cross_norm = np.linalg.norm(np.cross(x - pt_seg, v))
                # Calculate how far along the segment the intersection occurs
                goal = pt_seg + (np.sqrt(lookahead**2 - cross_norm**2) + dist_projected) * v
            else:
                goal = pt_seg

    # Transform the computed goal into the robot's local frame
    map_T_robot = np.array([[np.cos(theta), -np.sin(theta), x[0]],
                              [np.sin(theta),  np.cos(theta), x[1]],
                              [0,              0,             1]])
    inv_map_T_robot = np.linalg.inv(map_T_robot)
    goal_hom = np.array([goal[0], goal[1], 1])
    goal_local = np.matmul(inv_map_T_robot, goal_hom)
    return goal_local[0:2]


def upsample_scan(scan, target_length = 720):
    original_length = len(scan)
    x_old = np.linspace(0, 1, original_length)
    x_new = np.linspace(0, 1, target_length)
    upsampled_scan = np.interp(x_new, x_old, scan)
    return upsampled_scan

def scan_callback(msg):
    global scan_buffer
    scan_arr = np.array(msg.ranges, dtype=np.float32)
    if len(scan_arr) == 360:
        scan_arr = upsample_scan(scan_arr, target_length=720)
    scan_buffer.append(scan_arr)
    if len(scan_buffer) > 10:
        scan_buffer.pop(0)
    rclpy.logging.get_logger('controller_server').info(f"Lidar scan updated. Buffer size: {len(scan_buffer)}")

class LidarSubscriber(Node):
    def __init__(self):
        super().__init__('lidar_subscriber')
        self.create_subscription(
            LaserScan,
            '/scan',  
            scan_callback,
            10
        )

def computeVelocityCommands(pose, goal_pose, global_path_data):
    """
    Compute velocity commands using PPO inference.
    The observation is composed of:
      - A pedestrian map (80x80x2), set as zeros.
      - A lidar scan (80x80) processed from the /scan data.
      - A goal position (2x1) extracted from the global goal.
    The PPO model then outputs an action which is mapped to command velocities.
    
    Parameters:
      pose: The current robot pose (as a dict with 'position' and 'orientation')
      goal_pose: The goal pose (as a dict with a 'pose' key containing position and orientation)
      global_path_data: A list of dictionaries, each containing a 'pose' key.
      
    Returns:
      A tuple (linear_x, angular_z) with the computed velocities.
    """
    logger = rclpy.logging.get_logger('controller_server')
    global model, scan_buffer

    ped_pos = np.zeros((80, 80, 2), dtype=np.float32)
    
    # Process lidar data
    if len(scan_buffer) < 10:
        # Not enough scans collected; use a default (zeros) 10x720 array.
        raw_scan = np.zeros((10, 720), dtype=np.float32)
        logger.warn("Insufficient lidar scans in buffer; using zeros for lidar data.")
    else:
        raw_scan = np.stack(scan_buffer, axis=0)
    
    # Compute subgoal based on global path if available
    if global_path_data is not None:
        subgoal = pure_pursuit_subgoal(global_path_data, pose, lookahead=2.0)
    else:
        subgoal = np.array([goal_pose["pose"]["position"]["x"], 
                            goal_pose["pose"]["position"]["y"]], dtype=np.float32)
    
    scan_avg = np.zeros((20, 80), dtype=np.float32)
    for n in range(10):
        scan_tmp = raw_scan[n, :]
        for i in range(80):
            segment = scan_tmp[i*9:(i+1)*9]
            scan_avg[2*n, i] = np.min(segment)
            scan_avg[2*n+1, i] = np.mean(segment)

    scan_avg = scan_avg.reshape(1600)
    scan_avg_map = np.matlib.repmat(scan_avg, 1, 4)
    scan_norm = np.array(scan_avg_map).reshape(6400)
    s_min = 0
    s_max = 30
    scan_norm = 2 * (scan_norm - s_min) / (s_max - s_min) - 1

    # Normalize goal (MaxAbsScaler)
    g_min = -2.0
    g_max = 2.0
    goal_norm = 2 * (subgoal.reshape(2, 1) - g_min) / (g_max - g_min) - 1

    # Normalize pedestrian map
    v_min = -2 
    v_max = 2 
    ped_pos = 2 * (ped_pos - v_min) / (v_max - v_min) - 1

    observation = np.concatenate((ped_pos.flatten(), scan_norm.flatten(), goal_norm.flatten()))
    
    # Use the PPO model to predict an action.
    action, _ = model.predict(observation)
    
    # Map the action output to command velocities.
    vx_min = 0
    vx_max = 0.5
    vz_min = -2
    vz_max = 2
    
    linear_x = (action[0] + 1) * (vx_max - vx_min) / 2 + vx_min
    angular_z = (action[1] + 1) * (vz_max - vz_min) / 2 + vz_min
    
    return linear_x, angular_z



def handleGlobalPlan(global_path):
    position_x = []
    position_y = []
    i=0
    while(i <= len(global_path.poses)-1):
        position_x.append(global_path.poses[i].pose.position.x)
        position_y.append(global_path.poses[i].pose.position.y)
        i=i+1
    position_all = [list(double) for double in zip(position_x,position_y)]
    
    return position_all

def setPath(global_plan):
    global goal_pose , global_path_data
    goal_pose = global_plan.poses[-1]
    global_path_data = global_plan 
    global position_all
    position_all = handleGlobalPlan(global_plan)
    return


def setSpeedLimit(speed_limit, is_percentage):
    return

rclpy.init(args=None)
lidar_subscriber = LidarSubscriber()
import threading
sub_thread = threading.Thread(target=rclpy.spin, args=(lidar_subscriber,), daemon=True)
sub_thread.start()