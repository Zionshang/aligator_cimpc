import numpy as np
import pinocchio as pin
import pandas as pd
import time
from pinocchio.visualize import MeshcatVisualizer

# csv_path = "/home/zishang/cpp_workspace/aligator_cimpc/build/offline_test.csv"
csv_path = "/home/zishang/cpp_workspace/aligator_cimpc/build/offline_inv_sim_x.csv"

# 读取URDF文件和创建机器人模型
urdf_path = "/home/zishang/cpp_workspace/aligator_cimpc/robot/mini_cheetah/urdf/mini_cheetah_ground_mesh.urdf"
# urdf_path = "/home/zishang/cpp_workspace/aligator_cimpc/robot/galileo_v1d6_description/urdf/galileo_v1d6.urdf"
model = pin.buildModelFromUrdf(urdf_path)
visual_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.VISUAL)
collision_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.COLLISION)

# 设置可视化器
viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(loadModel=True)
viz.viewer.open()

# 读取轨迹数据
try:
    trajectory_data = pd.read_csv(csv_path, header=None)
    x_trajectory = trajectory_data.values
    q_trajectory = x_trajectory[:, : model.nq]

    while True:
        for q in q_trajectory:
            viz.display(q)
            time.sleep(0.06)  # 可调整显示速度
        time.sleep(1)  # 可调整显示速度
        
except FileNotFoundError:
    print("找不到轨迹文件")
except Exception as e:
    print(f"发生错误：{str(e)}")

