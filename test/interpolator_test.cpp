#include "interpolator.hpp"
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <iostream>

using namespace Eigen;

int main()
{
    std::cout << std::fixed << std::setprecision(2);

    std::string urdf_filename = "/home/zishang/cpp_workspace/aligator_cimpc/robot/mini_cheetah/urdf/mini_cheetah_ground_mesh.urdf";

    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf_filename, model);
    pinocchio::Data data(model);
    model.lowerPositionLimit.head<3>().fill(-1.);
    model.upperPositionLimit.head<3>().fill(1);
    std::srand(static_cast<unsigned int>(std::time(nullptr))); // 设置当前时间为随机种子

    // 创建插值器
    Interpolator interpolator(model);

    // 假设有4个配置点（3段轨迹）
    std::vector<VectorXd> qs(4, VectorXd::Zero(model.nq));
    qs[0] = pinocchio::randomConfiguration(model);
    qs[1] = pinocchio::randomConfiguration(model);
    qs[2] = pinocchio::randomConfiguration(model);
    qs[3] = pinocchio::randomConfiguration(model);

    // 对应每段轨迹的时间间隔
    std::vector<double> timesteps = {1.0, 2.0, 3.0}; // 不同步长

    // 线性数据（3维向量）
    std::vector<VectorXd> vs(4, VectorXd::Zero(model.nv));
    vs[0] = Eigen::VectorXd::Random(model.nv);
    vs[1] = Eigen::VectorXd::Random(model.nv);
    vs[2] = Eigen::VectorXd::Random(model.nv);
    vs[3] = Eigen::VectorXd::Random(model.nv);
    std::cout << "vs[0]: " << vs[0].transpose() << std::endl;
    std::cout << "vs[1]: " << vs[1].transpose() << std::endl;

    // 状态数据（构型 + 速度）
    std::vector<VectorXd> xs(4, VectorXd::Zero(model.nq + model.nv));
    for (size_t i = 0; i < 4; ++i)
    {
        xs[i].head(model.nq) = qs[i];
        xs[i].tail(model.nv) = vs[i];
    }

    // 测试 delay 在中间不同位置
    std::vector<double> test_delays = {0, 1.0, 3.0, 6.0};
    int i = 0;
    for (const auto &delay : test_delays)
    {
        VectorXd q_interp(model.nq), v_interp(model.nv), x_interp(model.nq + model.nv);

        interpolator.interpolateConfiguration(delay, timesteps, qs, q_interp);
        interpolator.interpolateLinear(delay, timesteps, vs, v_interp);
        interpolator.interpolateState(delay, timesteps, xs, x_interp);

        if (q_interp.isApprox(qs[i], 1e-6))
            std::cout << "interpolateConfiguration is correct" << std::endl;
        else
            std::cout << "interpolateConfiguration is wrong" << std::endl;

        if (v_interp.isApprox(vs[i], 1e-6))
            std::cout << "interpolateLinear is correct" << std::endl;
        else
            std::cout << "interpolateLinear is wrong" << std::endl;
        if (x_interp.isApprox(xs[i], 1e-6))

            std::cout << "interpolateState is correct" << std::endl;
        else
            std::cout << "interpolateState is wrong" << std::endl;
        i++;
    }

    VectorXd v_interp(model.nv);
    interpolator.interpolateLinear(timesteps[0] + timesteps[1] / 2, timesteps, vs, v_interp);
    if (v_interp.isApprox((vs[1] + vs[2]) / 2, 1e-6))
        std::cout << "interpolateLinear is correct" << std::endl;
    else
        std::cout << "interpolateLinear is wrong" << std::endl;
    return 0;
}
