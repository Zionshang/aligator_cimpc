#include "contact_fwd_dynamics.hpp"
#include <pinocchio/parsers/urdf.hpp>

#include <aligator/modelling/dynamics/multibody-free-fwd.hpp>

using MultibodyFreeFwdDynamics = aligator::dynamics::MultibodyFreeFwdDynamicsTpl<double>;
using MultibodyFreeFwdData = aligator::dynamics::MultibodyFreeFwdDataTpl<double>;

/*当没有发生碰撞时，ContactFwdDynamics类的计算结果应该跟
aligator类自带的MultibodyFreeFwdDynamics一样*/
int main()
{
    std::string urdf_filename = "/home/zishang/cpp_workspace/aligator_cimpc/robot/mini_cheetah/urdf/mini_cheetah_ground.urdf";

    //////////// 创建模型 //////////
    Model model;
    pinocchio::urdf::buildModel(urdf_filename, model);
    Data data(model);
    int nq = model.nq;
    int nv = model.nv;
    model.lowerPositionLimit.head<3>().fill(-1.);
    model.upperPositionLimit.head<3>().fill(-0.5);

    MultibodyPhaseSpace space(model);
    int nu = model.nv - 6;
    MatrixXd actuation = MatrixXd::Zero(model.nv, nu);
    actuation.bottomRows(nu).setIdentity();

    //////////// 创建动力学模型 //////////
    ContactFwdDynamics cont_dynamics(space, actuation);
    ContactFwdDynamicsData cont_dyn_data(cont_dynamics);

    MultibodyFreeFwdDynamics free_dynamics(space, actuation);
    MultibodyFreeFwdData free_dyn_data(&free_dynamics);

    ////////// 创建测试变量 //////////
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    Eigen::VectorXd q = pinocchio::randomConfiguration(model);
    Eigen::VectorXd v = Eigen::VectorXd::Random(model.nv);
    Eigen::VectorXd u = Eigen::VectorXd::Random(nu);
    Eigen::VectorXd x(nq + nv);
    x.head(nq) = q;
    x.tail(nv) = v;

    cont_dynamics.forward(x, u, cont_dyn_data);
    cont_dynamics.dForward(x, u, cont_dyn_data);
    free_dynamics.forward(x, u, free_dyn_data);
    free_dynamics.dForward(x, u, free_dyn_data);

    if (cont_dyn_data.xdot_.isApprox(free_dyn_data.xdot_, 1e-6))
        std::cout << "xdot is correct" << std::endl;
    else
        std::cout << "xdot is wrong" << std::endl;

    if (cont_dyn_data.Jx_.isApprox(free_dyn_data.Jx_, 1e-6))
        std::cout << "Jx is correct" << std::endl;
    else
        std::cout << "Jx is wrong" << std::endl;

    if (cont_dyn_data.Ju_.isApprox(free_dyn_data.Ju_, 1e-6))
        std::cout << "Ju is correct" << std::endl;
    else
        std::cout << "Ju is wrong" << std::endl;

    return 0;
}
