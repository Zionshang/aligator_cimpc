#include "contact_inv_dynamics_residual2.hpp"

#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>

ContactInvDynamicsResidual2::ContactInvDynamicsResidual2(int ndx, const Model &model, MatrixXd actuation, double timestep,
                                                         ContactParams<double> contact_params)
    : model_(model),
      StageFunction(ndx, model_.nv + actuation.cols(), model_.nv),
      timestep_(timestep), actuation_matrix_(actuation), contact_params_(contact_params)
{
    const int nv = model_.nv;
    if (nv != actuation.rows())
    {
        ALIGATOR_DOMAIN_ERROR(
            fmt::format("actuation matrix should have number of rows = pinocchio "
                        "model nv ({} and {}).",
                        actuation.rows(), nv));
    }
}

void ContactInvDynamicsResidual2::evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                                           StageFunctionData &data) const
{
    ContactInvDynamicsResidualData2 &d = static_cast<ContactInvDynamicsResidualData2 &>(data);

    const int nq = model_.nq;
    const int nv = model_.nv;
    const int nu = actuation_matrix_.cols();
    const auto &q = x.head(nq);
    const auto &v = x.segment(nq, nv);
    const auto &a = u.head(nv);
    const auto &tau = actuation_matrix_ * u.tail(nu);

    // 使用半隐式欧拉法计算下一时刻的状态
    VectorXd v_next = v + a * timestep_;                                   // 先更新速度
    VectorXd q_next = pinocchio::integrate(model_, q, v_next * timestep_); // 再更新位置

    // 计算接触力
    pinocchio::forwardKinematics(model_, d.data_, q_next, v_next);
    pinocchio::updateFramePlacements(model_, d.data_);
    aligned_vector<pinocchio::Force> f_ext(model_.njoints, pinocchio::Force::Zero()); // todo: 删除临时变量
    CalcContactForceContribution(model_, d.data_, f_ext, contact_params_, d.contact_forces_);

    // 逆动力学计算
    d.tau_ = pinocchio::rnea(model_, d.data_, q_next, v_next, a, f_ext);

    data.value_ = d.tau_ - tau;
}

void ContactInvDynamicsResidual2::computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                                                   StageFunctionData &data) const
{
    ContactInvDynamicsResidualData2 &d = static_cast<ContactInvDynamicsResidualData2 &>(data);
    const int nq = model_.nq;
    const int nv = model_.nv;
    const int nu = actuation_matrix_.cols();

    Eigen::VectorXd X(nq + nv + nv + nv); // q, v, a, dq
    X << x, u.head(nv), Eigen::VectorXd::Zero(nv);

    Eigen::VectorXd dtau_dx_vec = d.ad_inv_dynamics_.Jacobian(X);
    Eigen::MatrixXd dtau_dx = dtau_dx_vec.reshaped(X.size(), nv).transpose();
    Eigen::MatrixXd dtau_dq = dtau_dx.rightCols(nv);
    Eigen::MatrixXd dtau_dv = dtau_dx.middleCols(nq, nv);
    Eigen::MatrixXd dtau_da = dtau_dx.middleCols(nq + nv, nv);

    d.Jx_.leftCols(nv) = dtau_dq;
    d.Jx_.rightCols(nv) = dtau_dv;
    d.Ju_.leftCols(nv) = dtau_da;
}

std::shared_ptr<StageFunctionData> ContactInvDynamicsResidual2::createData() const
{
    return std::make_shared<ContactInvDynamicsResidualData2>(*this);
}

ContactInvDynamicsResidualData2::ContactInvDynamicsResidualData2(const ContactInvDynamicsResidual2 &resdl)
    : StageFunctionData(resdl),
      tau_(resdl.model_.nv),
      data_(resdl.model_),
      dtau_dtaua_(resdl.actuation_matrix_),
      contact_forces_(4, Vector3d::Zero())
{
    int nu = resdl.actuation_matrix_.cols();
    tau_.setZero();
    Ju_.rightCols(nu) = -resdl.actuation_matrix_;

    //////////// 创建 CppAD 的正动力学函数 //////////
    using CppAD::AD;
    using ADVectorX = Eigen::VectorX<AD<double>>;

    pinocchio::ModelTpl<AD<double>> ad_model = resdl.model_.cast<AD<double>>();
    pinocchio::DataTpl<AD<double>> ad_data(ad_model);
    int nq = ad_model.nq;
    int nv = ad_model.nv;

    ADVectorX ad_X(nq + nv + nv + nv); // q, v, a, dq
    ad_X.setZero();
    CppAD::Independent(ad_X);
    ADVectorX ad_Y(nv);

    ADVectorX ad_q = ad_X.head(nq);
    ADVectorX ad_v = ad_X.segment(nq, nv);
    ADVectorX ad_a = ad_X.segment(nq + nv, nv);
    ADVectorX ad_dq = ad_X.tail(nv);

    ADVectorX ad_q_plus = pinocchio::integrate(ad_model, ad_q, ad_dq);

    ADVectorX ad_v_next = ad_v + ad_a * resdl.timestep_;                                        // 先更新速度
    ADVectorX ad_q_next = pinocchio::integrate(model_, ad_q_plus, ad_v_next * resdl.timestep_); // 再更新位置

    pinocchio::forwardKinematics(ad_model, ad_data, ad_q_next, ad_v_next);
    pinocchio::updateFramePlacements(ad_model, ad_data);
    aligned_vector<pinocchio::ForceTpl<AD<double>>> ad_f_ext(ad_model.njoints,
                                                             pinocchio::ForceTpl<AD<double>>::Zero());
    ContactParams<CppAD::AD<double>> ad_contact_params(resdl.contact_params_);
    CalcContactForceContributionAD(ad_model, ad_data, ad_f_ext, ad_contact_params);
    ad_Y = pinocchio::rnea(ad_model, ad_data, ad_q_next, ad_v_next, ad_a, ad_f_ext);
    ad_inv_dynamics_.Dependent(ad_X, ad_Y);
}
