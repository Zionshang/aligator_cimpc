#include "contact_inv_dynamics_residual.hpp"

#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>

ContactInvDynamicsResidual::ContactInvDynamicsResidual(int ndx, const Model &model, MatrixXd actuation,
                                                       ContactParams<double> contact_params)
    : StageFunction(ndx, model.nv + actuation.cols(), model.nv),
      model_(model), actuation_matrix_(actuation), contact_params_(contact_params)
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

void ContactInvDynamicsResidual::evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                                          StageFunctionData &data) const
{
    ContactInvDynamicsResidualData &d = static_cast<ContactInvDynamicsResidualData &>(data);

    const int nq = model_.nq;
    const int nv = model_.nv;
    const int nu = actuation_matrix_.cols();
    const auto &q = x.head(nq);
    const auto &v = x.segment(nq, nv);
    const auto &a = u.head(nv);
    const auto &tau = actuation_matrix_ * u.tail(nu);

    // 计算接触力
    pinocchio::forwardKinematics(model_, d.data_, q, v);
    pinocchio::updateFramePlacements(model_, d.data_);
    aligned_vector<pinocchio::Force> f_ext(model_.njoints, pinocchio::Force::Zero()); // todo: 删除临时变量
    CalcContactForceContribution(model_, d.data_, f_ext, contact_params_, d.contact_forces_);

    // 逆动力学计算
    d.tau_ = pinocchio::rnea(model_, d.data_, q, v, a, f_ext);

    data.value_ = d.tau_ - tau;
}

void ContactInvDynamicsResidual::computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                                                  StageFunctionData &data) const
{
    ContactInvDynamicsResidualData &d = static_cast<ContactInvDynamicsResidualData &>(data);
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

std::shared_ptr<StageFunctionData> ContactInvDynamicsResidual::createData() const
{
    return std::make_shared<ContactInvDynamicsResidualData>(*this);
}

ContactInvDynamicsResidualData::ContactInvDynamicsResidualData(const ContactInvDynamicsResidual &resdl)
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
    pinocchio::forwardKinematics(ad_model, ad_data, ad_q_plus, ad_v);
    pinocchio::updateFramePlacements(ad_model, ad_data);
    aligned_vector<pinocchio::ForceTpl<AD<double>>> ad_f_ext(ad_model.njoints,
                                                             pinocchio::ForceTpl<AD<double>>::Zero());
    ContactParams<CppAD::AD<double>> ad_contact_params(resdl.contact_params_);
    CalcContactForceContributionAD(ad_model, ad_data, ad_f_ext, ad_contact_params);
    ad_Y = pinocchio::rnea(ad_model, ad_data, ad_q_plus, ad_v, ad_a, ad_f_ext);
    ad_inv_dynamics_.Dependent(ad_X, ad_Y);
}
