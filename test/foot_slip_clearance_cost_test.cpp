#include "foot_slip_clearance_cost.hpp"
#include <pinocchio/parsers/urdf.hpp>

int main()
{
    std::string urdf_filename = "/home/zishang/cpp_workspace/aligator_cimpc/robot/mini_cheetah/urdf/mini_cheetah_ground.urdf";

    //////////// 创建模型 //////////
    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf_filename, model);
    int nq = model.nq;
    int nv = model.nv;
    int nu = model.nv - 6;
    model.lowerPositionLimit.head<3>().fill(-1.);
    model.upperPositionLimit.head<3>().fill(-0.5);
    MultibodyPhaseSpace space(model);

    //////////// 创建动力学模型 //////////
    FootSlipClearanceCost fsc_cost(space, nu, 1.0, -30.0);

    ////////// 创建测试变量 //////////
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    Eigen::VectorXd q = pinocchio::randomConfiguration(model);
    Eigen::VectorXd v = Eigen::VectorXd::Random(model.nv);
    Eigen::VectorXd u = Eigen::VectorXd::Random(nu);
    Eigen::VectorXd x(nq + nv);
    x.head(nq) = q;
    x.tail(nv) = v;

    ////////// cppad 测试 //////////
    auto fsc_cost_data = fsc_cost.createData();
    fsc_cost.evaluate(x, u, *fsc_cost_data);
    FootSlipClearanceCostData &data = static_cast<FootSlipClearanceCostData &>(*fsc_cost_data);
    Eigen::VectorXd x2(nq + nv + nv);
    x2 << x, Eigen::VectorXd::Zero(nv);
    Eigen::VectorXd ad_cost = data.ad_cost_fun_.Forward(0, x2);

    std::cout << fsc_cost_data->value_ << std::endl;
    std::cout << ad_cost[0] << std::endl;

    return 0;
}
