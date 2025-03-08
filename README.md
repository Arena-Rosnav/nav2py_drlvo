# Nav2_pyif_drl_vo

This repository integrates the ROS 2 Navigation Stack (Nav2) with a Deep Reinforcement Learning-based Velocity Obstacle (DRL-VO) control policy, enabling autonomous navigation through environments with static and dynamic obstacles.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Tonmanhkien/Nav2_pyif_drl_vo.git
   cd Nav2_pyif_drl_vo

2. **Dowload PPO model**:

   ```bash
   wget https://github.com/TempleRAIL/drl_vo_nav/blob/drl_vo/drl_vo/src/model/drl_vo.zip -O drl_vo.zip

  Then add it into model folder inside drl_controller folder
  
3. **Install Necessary Libraries**:

   
   Ensure you have the required Python libraries installed:
   
   ```bash
   pip install stable-baselines3 numpy==1.26

4. **Run**:
   ```bash
   cd /Nav2_pyif_drl_vo
   . colcon_build
   ros2 launch nav2_bringup tb3_simulation_launch.py params_file:=<>/src/param.yaml headless:=True
   
