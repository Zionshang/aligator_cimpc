import numpy as np
import matplotlib.pyplot as plt

def plot_cost(*csv_files):
    # 创建绘图窗口
    plt.figure()
    for csv_file in csv_files:
        # 读取 CSV 文件
        cost_data = np.loadtxt(csv_file, delimiter=",")
        
        # 如果数据是多维的，选择第一列或展平
        if len(cost_data.shape) > 1 and cost_data.shape[1] > 1:
            cost_values = cost_data[:, 0]  # 默认取第一列
        else:
            cost_values = cost_data
        
        # 绘制曲线
        plt.plot(cost_values, label=f"Cost from {csv_file}")

    # 添加标题和标签
    plt.title("Optimization Cost Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Cost Value")
    plt.legend()
    plt.grid(True)



cost_csv_1 = "/home/zishang/cpp_workspace/aligator_cimpc/build/offline_inv_sim_cost.csv"
cost_csv_2 = "/home/zishang/cpp_workspace/aligator_cimpc/build/offline_fwd_sim_cost.csv"

plot_cost(cost_csv_1, cost_csv_2)
plt.show()