import numpy as np
import matplotlib.pyplot as plt


def plot_contact_forces(csv_file):
    # 读取 CSV 文件
    data = np.loadtxt(csv_file, delimiter=",")

    # Z 向力（列 2, 5, 8, 11）
    fz_0 = data[:, 2]
    fz_1 = data[:, 5]
    fz_2 = data[:, 8]
    fz_3 = data[:, 11]

    # X 向力（列 0, 3, 6, 9）
    fx_0 = data[:, 0]
    fx_1 = data[:, 3]
    fx_2 = data[:, 6]
    fx_3 = data[:, 9]
    fx_total = fx_0 + fx_1 + fx_2 + fx_3

    # 第一个窗口：Z 方向力（子图形式）
    plt.figure(figsize=(10, 8))
    plt.subplot(4, 1, 1)
    plt.plot(fz_0, label="Leg 0 Z-force")
    plt.title("Leg 0 Z-direction Contact Force")
    plt.xlabel("Time Step")
    plt.ylabel("Z Force (N)")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(fz_1, label="Leg 1 Z-force")
    plt.title("Leg 1 Z-direction Contact Force")
    plt.xlabel("Time Step")
    plt.ylabel("Z Force (N)")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(fz_2, label="Leg 2 Z-force")
    plt.title("Leg 2 Z-direction Contact Force")
    plt.xlabel("Time Step")
    plt.ylabel("Z Force (N)")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(fz_3, label="Leg 3 Z-force")
    plt.title("Leg 3 Z-direction Contact Force")
    plt.xlabel("Time Step")
    plt.ylabel("Z Force (N)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # 第二个窗口：X 方向合力
    plt.figure()
    plt.plot(fx_total, label="Total X-force")
    plt.title("Total X-direction Force")
    plt.xlabel("Time Step")
    plt.ylabel("X Force (N)")
    plt.legend()
    plt.grid(True)


# 调用方式：
contact_forces_csv = (
    "/home/zishang/cpp_workspace/aligator_cimpc/build/offine_inv_sim_contact_forces.csv"
)

plot_contact_forces(contact_forces_csv)
plt.show()