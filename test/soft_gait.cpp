#include "gait_phase_alignment.hpp"
#include <pinocchio/parsers/urdf.hpp>
#include <proxsuite-nlp/modelling/spaces/multibody.hpp>
#include <pinocchio/autodiff/cppad.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <aligator/modelling/costs/relaxed-log-barrier.hpp>

using MultibodyPhaseSpace = proxsuite::nlp::MultibodyPhaseSpace<double>;
using RelaxedLogBarrierCost = aligator::RelaxedLogBarrierCostTpl<double>;

int main()
{

    std::string urdf_filename = "/home/zishang/cpp_workspace/aligator_cimpc/robot/mini_cheetah/urdf/mini_cheetah_ground_mesh.urdf";

    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf_filename, model);
    model.lowerPositionLimit.head<3>().fill(0.);
    model.upperPositionLimit.head<3>().fill(1.);

    MultibodyPhaseSpace space(model);
    const int nu = model.nv - 6;
    const int nq = model.nq;
    const int nv = model.nv;

    std::vector<double> phi = {0.0, 0.0, M_PI, M_PI}; // 步态偏移量
    double T = 1.0;                                   // 步态周期为1秒
    GaitAlignmentResidual gait_phase_alignment(model, space.ndx(), nu, phi, T);
    auto gait_phase_alignment_data = gait_phase_alignment.createData();
    Eigen::VectorXd x = Eigen::VectorXd::Zero(nq + nv);
    x.head(nq) << 0.0, 0.0, 0.29,
        0.0, 0.0, 0.0, 1.0,
        0.0, -0.8, 1.6,
        0.0, -0.8, 1.6,
        0.0, -0.8, 1.6,
        0.0, -0.8, 1.6;
    Eigen::VectorXd u = Eigen::VectorXd::Zero(nu);

    for (double t = 0.0; t <= 2.0; t += 0.1)
    {
        gait_phase_alignment.updateTime(t);
        gait_phase_alignment.evaluate(x, u, *gait_phase_alignment_data);
        std::cout << "Time: " << t << ", Gait Phase Alignment Values: "
                  << gait_phase_alignment_data->value_.transpose() << std::endl;
    }

    ////////////////////// 利用cppad验证导数是否正确 //////////////////////
    using CppAD::AD;
    using ADVectorX = Eigen::VectorX<AD<double>>;
    int num_feet = 4;

    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    double t = static_cast<double>(std::rand()) / RAND_MAX * T; // 随机生成 [0, T] 范围内的 t

    CppAD::ADFun<double> ad_fun_f;

    pinocchio::ModelTpl<AD<double>> ad_model = model.cast<AD<double>>();
    pinocchio::DataTpl<AD<double>> ad_data(ad_model);

    ADVectorX ad_X(nq + nv); // q, dq
    ad_X.setZero();
    CppAD::Independent(ad_X);
    ADVectorX ad_f(num_feet);

    ADVectorX ad_q_plus = pinocchio::integrate(ad_model, ad_X.head(nq), ad_X.tail(nv));
    pinocchio::forwardKinematics(ad_model, ad_data, ad_q_plus);
    pinocchio::updateFramePlacements(ad_model, ad_data);
    for (size_t i = 0; i < num_feet; i++)
    {
        // 计算 signed distance
        AD<double> signed_distance = ad_data.oMf[gait_phase_alignment.foot_frame_ids_[i]].translation()(2) -
                                     AD<double>(gait_phase_alignment.foot_radius_);

        // 计算 gait phase alignment
        AD<double> slope = gait_phase_alignment.slope_;
        AD<double> offset = gait_phase_alignment.offset_;
        AD<double> d_lower = gait_phase_alignment.d_lower_;

        ad_f(i) = calcGaitAlignment(signed_distance, AD<double>(t), AD<double>(T), AD<double>(phi[i]), slope, offset) - d_lower;
    }
    ad_fun_f.Dependent(ad_X, ad_f);

    //////////////////////// 验证导数 //////////////////////
    Eigen::VectorXd q = pinocchio::randomConfiguration(model);
    x.head(nq) = q;

    Eigen::VectorXd X(nq + nv);
    X << q, Eigen::VectorXd::Zero(nv);
    Eigen::VectorXd df_dx_vec = ad_fun_f.Jacobian(X);
    Eigen::MatrixXd df_dx = df_dx_vec.reshaped(X.size(), num_feet).transpose();
    Eigen::MatrixXd df_dq = df_dx.rightCols(nv);

    gait_phase_alignment.updateTime(t);
    gait_phase_alignment.evaluate(x, u, *gait_phase_alignment_data);
    gait_phase_alignment.computeJacobians(x, u, *gait_phase_alignment_data);
    GaitAlignmentResidualData &d = static_cast<GaitAlignmentResidualData &>(*gait_phase_alignment_data);

    if (df_dq.isApprox(gait_phase_alignment_data->Jx_.leftCols(nv), 1e-6))
        std::cout << "Jacobians is correct" << std::endl;
    else
        std::cout << "Jacobians is wrong" << std::endl;

    ////////////////////// 测试加入了障碍函数的cost //////////////////////
    double weight = 1;
    double threshold = 1e-2;
    RelaxedLogBarrierCost log_barrier_cost(space, gait_phase_alignment, weight, threshold);
    auto log_barrier_cost_data = log_barrier_cost.createData();
    for (double t = 0.0; t <= 2.0; t += 0.1)
    {
        GaitAlignmentResidual *gar = log_barrier_cost.getResidual<GaitAlignmentResidual>();
        gar->updateTime(t);
        log_barrier_cost.evaluate(x, u, *log_barrier_cost_data);
        std::cout << "Time: " << t << ", Relaxed Log Barrier Cost: "
                  << log_barrier_cost_data->value_ << std::endl;
    }
    return 0;
}