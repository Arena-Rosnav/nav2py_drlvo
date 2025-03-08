# nav2py_template

This project provides a Python-based controller interface for interacting with `nav2` in ROS2.

## 📌 Installation

### **1. Install ROS2 (Humble) and Navigation 2**
- [ROS2 Installation Guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
- [Nav2 Installation Guide](https://docs.nav2.org/getting_started/index.html)

### **2. Clone this repository into your workspace**
```bash
mkdir -p ~/nav2_ws/src
cd ~/nav2_ws/src
git clone git@github.com:zenghjian/nav2py_template.git
```

### **3. Build the workspace**
```bash
cd ~/nav2_ws
source /opt/ros/humble/setup.bash
colcon build
```

### **4. Source the workspace**
```bash
source ~/nav2_ws/install/setup.bash
```

### **5. Run the Nav2 demo launch**
```bash
export TURTLEBOT3_MODEL=waffle
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/humble/share/turtlebot3_gazebo/models
ros2 launch nav2_bringup tb3_simulation_launch.py headless:=False params_file:=$(pwd)/src/nav2py_template/my_nav2_params.yaml
```

---

## 🔧 Development

### **Code Structure**
```
nav2py_template/
│── src/
│   ├── nav2py_template/
│   │   ├── nav2py_template_controller/
│   │   │   ├── __init__.py
│   │   │   ├── __main__.py  # Main entry point, calls the planner
│   │   │   ├── planner.py   # Custom planner implementation
│   │   ├── template_controller.cpp  # C++ controller
│   │   ├── template_controller.hpp  # C++ controller header file
│── README.md
│── setup.py
```

### **Development Guide**
1. **Add your Python code in the `src/nav2py_template/nav2py_template_controller/` directory.**
   - Entry file: `__main__.py`
   - Implement `planner.py` according to your navigation requirements.

2. **If additional input data is needed**, modify the following files:
   - `template_controller.cpp`
   - `template_controller.hpp`
   - `__main__.py`

   **The current example only supports transmitting:**
   - Robot position
   - Velocity
   - Goal


## 📧 Contact
For any inquiries, please contact [Huajian Zeng](mailto:zenghuajian97@gmail.com).