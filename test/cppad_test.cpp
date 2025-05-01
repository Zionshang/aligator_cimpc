#include "contact_force.hpp"
#include <pinocchio/autodiff/cppad.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/aba-derivatives.hpp>

using CppAD::AD;
using ADVectorX = Eigen::VectorX<AD<double>>;

int main(int, char **)
{
    std::string urdf_filename = "/home/zishang/cpp_workspace/aligator_cimpc/robot/mini_cheetah/urdf/mini_cheetah_ground.urdf";

    //////////// 创建模型 //////////
    Model model;
    pinocchio::urdf::buildModel(urdf_filename, model);
    Data data(model);
    int nq = model.nq;
    int nv = model.nv;
    model.lowerPositionLimit.head<3>().fill(-1.);
    model.upperPositionLimit.head<3>().fill(0.5);

    ////////// 创建 CppAD 相关变量 //////////
    pinocchio::ModelTpl<AD<double>> ad_model = model.cast<AD<double>>();
    pinocchio::DataTpl<AD<double>> ad_data(ad_model);

    ADVectorX ad_X(nq + nv + nv + nv); // q, v, tau, dq
    ad_X.setZero();
    CppAD::Independent(ad_X);
    Eigen::VectorX<AD<double>> ad_Y(nv);

    //////////// 创建 CppAD 的正动力学函数 //////////
    ADVectorX ad_q_plus = pinocchio::integrate(ad_model, ad_X.head(nq), ad_X.tail(nv));
    pinocchio::forwardKinematics(ad_model, ad_data, ad_q_plus, ad_X.segment(nq, nv));
    pinocchio::updateFramePlacements(ad_model, ad_data);
    aligned_vector<pinocchio::ForceTpl<AD<double>>> ad_f_ext(ad_model.njoints,
                                                             pinocchio::ForceTpl<AD<double>>::Zero());
    CalcContactForceContributionAD(ad_model, ad_data, ad_f_ext);
    ad_Y = pinocchio::aba(ad_model, ad_data, ad_q_plus, ad_X.segment(nq, nv), ad_X.segment(nq + nv, nv), ad_f_ext);
    CppAD::ADFun<double> ad_fun(ad_X, ad_Y);

    ////////// 生成测试向量 q, v, tau //////////
    std::srand(static_cast<unsigned int>(std::time(nullptr))); // 设置当前时间为随机种子
    Eigen::VectorXd q = pinocchio::randomConfiguration(model);
    Eigen::VectorXd v = Eigen::VectorXd::Random(nv);
    Eigen::VectorXd tau = Eigen::VectorXd::Random(nv);

    //////////// 正常计算正动力学 //////////
    pinocchio::forwardKinematics(model, data, q, v);
    pinocchio::updateFramePlacements(model, data);
    aligned_vector<pinocchio::Force> f_ext(model.njoints, pinocchio::Force::Zero());
    CalcContactForceContribution(model, data, f_ext);
    Eigen::VectorXd a = pinocchio::aba(model, data, q, v, tau, f_ext);

    //////////// CppAD 计算正动力学 //////////
    Eigen::VectorXd dq = Eigen::VectorXd::Zero(nv);
    Eigen::VectorXd X(nq + nv + nv + nv);
    X << q, v, tau, dq;                       // 按顺序拼接
    Eigen::VectorXd Y = ad_fun.Forward(0, X); // 0 表示零阶导数，也就是直接计算函数值

    //////////// 输出正动力学结果 //////////
    std::cout << "cppad_Y = " << Y.transpose() << std::endl;
    std::cout << "pinocchio_a = " << a.transpose() << std::endl;

    //////////// CppAD 计算雅可比矩阵 //////////
    Eigen::VectorXd da_dx_vec = ad_fun.Jacobian(X);
    Eigen::MatrixXd da_dx = da_dx_vec.reshaped(ad_X.size(), ad_Y.size()).transpose();
    Eigen::MatrixXd ad_da_dq = da_dx.rightCols(nv);
    Eigen::MatrixXd ad_da_dv = da_dx.middleCols(nq, nv);
    Eigen::MatrixXd ad_da_dtau = da_dx.middleCols(nq + nv, nv);

    //////////// pinocchio 计算解析雅可比矩阵 //////////
    pinocchio::computeABADerivatives(model, data, q, v, tau);
    Eigen::MatrixXd analytical_da_dq = data.ddq_dq;
    Eigen::MatrixXd analytical_da_dv = data.ddq_dv;
    Eigen::MatrixXd analytical_da_dtau = data.Minv;

    //////////// 输出雅可比矩阵 //////////
    // pinocchio只能计算没有外力与状态无关时的雅可比矩阵，因此验证时，必须保证接触力为0
    if (ad_da_dq.isApprox(analytical_da_dq, 1e-6))
        std::cout << "da_dq is correct" << std::endl;
    else
        std::cout << "da_dq is wrong" << std::endl;

    if (ad_da_dv.isApprox(analytical_da_dv, 1e-6))
        std::cout << "da_dv is correct" << std::endl;
    else
        std::cout << "da_dv is wrong" << std::endl;

    if (ad_da_dtau.isApprox(analytical_da_dtau, 1e-6))
        std::cout << "da_dtau is correct" << std::endl;
    else
        std::cout << "da_dtau is wrong" << std::endl;
}