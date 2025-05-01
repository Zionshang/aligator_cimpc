#include "contact_force.hpp"
#include <pinocchio/autodiff/cppad.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>

int main(int, char **)
{
    std::string urdf_filename = "/home/zishang/cpp_workspace/aligator_cimpc/robot/mini_cheetah/urdf/mini_cheetah_ground.urdf";

    //////////// 创建模型 //////////
    Model model;
    pinocchio::urdf::buildModel(urdf_filename, model);
    Data data(model);
    int nq = model.nq;
    int nv = model.nv;
    model.lowerPositionLimit.head<3>().fill(1.);
    model.upperPositionLimit.head<3>().fill(2.);

    ////////// 创建 CppAD 函数 //////////
    using CppAD::AD;
    using ADVectorX = Eigen::VectorX<AD<double>>;

    pinocchio::ModelTpl<AD<double>> ad_model = model.cast<AD<double>>();
    pinocchio::DataTpl<AD<double>> ad_data(ad_model);

    ADVectorX ad_X(nq + nv + nv - 6 + nv); // q, v, tau, dq
    ad_X.setZero();
    aligned_vector<pinocchio::ForceTpl<AD<double>>> ad_f_ext(ad_model.njoints,
                                                             pinocchio::ForceTpl<AD<double>>::Zero());
    CppAD::Independent(ad_X);
    Eigen::VectorX<AD<double>> ad_Y(nv);
    // forwardDynamics<AD<double>>(ad_model, ad_data,
    //                             ad_X.head(nq), ad_X.segment(nq, nv), ad_X.tail(nv - 6),
    //                             contact_param, ad_f_ext, ad_Y);
    forwardDynamics<AD<double>>(ad_model, ad_data,
                                ad_X.head(nq), ad_X.segment(nq, nv), ad_X.tail(nv - 6),
                                contact_param, ad_Y);
    CppAD::ADFun<double> fun(ad_X, ad_Y);

    ////////// 生成测试向量 q, v, tau //////////
    Eigen::VectorXd q(model.nq);
    q = pinocchio::randomConfiguration(model);
    Eigen::VectorXd v(Eigen::VectorXd::Random(nv));
    Eigen::VectorXd u(Eigen::VectorXd::Random(nv - 6));
    Eigen::VectorXd x = Eigen::VectorXd::Zero(nq + nv + nv - 6);
    x << q, v, u;

    ////////// 使用 CppAD 求正动力学 //////////
    Eigen::VectorXd a_generated = fun.Forward(0, x);
    std::cout << "a_generated:\n"
              << a_generated.transpose() << std::endl;

    ////////// 使用 pinocchio 求正动力学//////////
    Eigen::VectorXd tau(model.nv);
    tau.setZero();
    tau.tail(12) = u;
    pinocchio::aba(model, data, q, v, tau);
    Eigen::VectorXd a_analytical = data.ddq;
    std::cout << "a_analytical:\n"
              << a_analytical.transpose() << std::endl;

    ////////// 比较两者的结果 //////////
    if (a_generated.isApprox(a_analytical, 1e-6))
        std::cout << "a is correct" << std::endl;
    else
        std::cout << "a is wrong" << std::endl;
}