import matplotlib.pyplot as plt
import csv

def read_csv(file_path):
    """读取CSV文件并返回数据列表"""
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append([float(value) for value in row])
    return data

def plot_joint_positions(desired_positions, actual_positions):
    """绘制每个关节位置的对比图"""
    num_joints = len(desired_positions[0])
    time_steps = range(len(desired_positions))
    
    for joint_idx in range(3):
        desired = [row[joint_idx] for row in desired_positions]
        actual = [row[joint_idx] for row in actual_positions]
        
        plt.figure()
        plt.plot(time_steps, desired, label='Desired Position')
        plt.plot(time_steps, actual, label='Actual Position')
        plt.title(f'Joint {joint_idx + 1} Position')
        plt.xlabel('Time Step')
        plt.ylabel('Position')
        plt.legend()
        plt.grid()
    
    plt.show()  # 在所有图绘制完成后一次性显示

# 替换为实际的CSV文件路径
desired_csv_path = "/home/zishang/cpp_workspace/aligator_cimpc/build/webots_sim_qd.csv"
actual_csv_path = "/home/zishang/cpp_workspace/aligator_cimpc/build/webots_sim_q.csv"

desired_positions = read_csv(desired_csv_path)
actual_positions = read_csv(actual_csv_path)

plot_joint_positions(desired_positions, actual_positions)
