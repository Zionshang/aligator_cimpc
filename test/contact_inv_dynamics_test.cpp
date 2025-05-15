#include "contact_fwd_dynamics.hpp"
#include "contact_inv_dynamics_residual.hpp"

#include <pinocchio/parsers/urdf.hpp>
#include <aligator/modelling/dynamics/multibody-free-fwd.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <aligator/modelling/autodiff/cost-finite-difference.hpp>
#include <aligator/modelling/costs/quad-residual-cost.hpp>

using MultibodyFreeFwdDynamics = aligator::dynamics::MultibodyFreeFwdDynamicsTpl<double>;
using MultibodyFreeFwdData = aligator::dynamics::MultibodyFreeFwdDataTpl<double>;
using CostFiniteDifferenceHelper = aligator::autodiff::CostFiniteDifferenceHelper<double>;
using QuadraticResidualCost = aligator::QuadraticResidualCostTpl<double>;

int main()
{
    std::string urdf_filename = "/home/zishang/cpp_workspace/aligator_cimpc/robot/mini_cheetah/urdf/mini_cheetah_ground.urdf";

    //////////// 创建模型 //////////
    Model model;
    pinocchio::urdf::buildModel(urdf_filename, model);
    Data data(model);
    int nq = model.nq;
    int nv = model.nv;
    model.lowerPositionLimit.head<3>().fill(1);
    model.upperPositionLimit.head<3>().fill(2);

    MultibodyPhaseSpace space(model);
    int nu = model.nv - 6;
    MatrixXd actuation = MatrixXd::Zero(model.nv, nu);
    actuation.bottomRows(nu).setIdentity();

    ////////// 创建正逆动力学 //////////
    ContactParams<double> contact_params;
    ContactFwdDynamics cont_fwd_dynamics(space, actuation, contact_params);
    ContactFwdDynamicsData cont_fwd_dyn_data(cont_fwd_dynamics);

    ContactInvDynamicsResidual cant_inv_dynamics(space.ndx(), model, actuation, contact_params);
    ContactInvDynamicsResidualData cont_inv_dyn_data(cant_inv_dynamics);

    ////////// 创建测试变量 //////////
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    Eigen::VectorXd q = pinocchio::randomConfiguration(model);
    Eigen::VectorXd v = Eigen::VectorXd::Random(model.nv);
    Eigen::VectorXd tau = Eigen::VectorXd::Random(nu);
    Eigen::VectorXd x(nq + nv);
    x.head(nq) = q;
    x.tail(nv) = v;

    ////////// 正动力学计算 //////////
    Eigen::VectorXd u_fwd = tau;
    cont_fwd_dynamics.forward(x, u_fwd, cont_fwd_dyn_data);
    Eigen::VectorXd a = cont_fwd_dyn_data.xdot_.tail(nv);

    ////////// 逆动力学计算 //////////
    Eigen::VectorXd u_inv(nv + nu); // a + tau
    u_inv.head(nv) = a;
    u_inv.tail(nu) = tau;
    cant_inv_dynamics.evaluate(x, u_inv, cont_inv_dyn_data);

    std::cout << "cont_inv_dyn_data.value_: " << cont_inv_dyn_data.value_.transpose() << std::endl;

    ////////// 逆动力学雅可比计算 //////////
    cant_inv_dynamics.computeJacobians(x, u_inv, cont_inv_dyn_data);
    std::cout << "Jx: \n"
              << cont_inv_dyn_data.Jx_ << std::endl;
    std::cout << "Ju: \n"
              << cont_inv_dyn_data.Ju_ << std::endl;

    return 0;
}
