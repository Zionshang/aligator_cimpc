#include "contact_fwd_dynamics.hpp"

ContactFwdDynamics::ContactFwdDynamics(const MultibodyPhaseSpace &space, const MatrixXd &actuation)
    : ODEAbstract(space, (int)actuation.cols()),
      space_(space), actuation_matrix_(actuation)
{
    const int nv = space.getModel().nv;
    if (nv != actuation.rows())
    {
        ALIGATOR_DOMAIN_ERROR(
            fmt::format("actuation matrix should have number of rows = pinocchio "
                        "model nv ({} and {}).",
                        actuation.rows(), nv));
    }
}

void ContactFwdDynamics::forward(const ConstVectorRef &x, const ConstVectorRef &u,
                                 ContinuousDynamicsData &data) const
{
    ContactFwdDynamicsData &d = static_cast<ContactFwdDynamicsData &>(data);
    const Model &model = space_.getModel();
    d.tau_.noalias() = actuation_matrix_ * u;

    const int nq = model.nq;
    const int nv = model.nv;
    const auto &q = x.head(nq);
    const auto &v = x.segment(nq, nv);

    pinocchio::forwardKinematics(model, d.data_, q, v);
    pinocchio::updateFramePlacements(model, d.data_);
    aligned_vector<pinocchio::Force> f_ext(model.njoints, pinocchio::Force::Zero()); // todo: 删除临时变量
    // CalcContactForceContribution(model, d.data_, f_ext);
    CalcContactForceContribution(model, d.data_, f_ext, d.contact_forces_);

    data.xdot_.head(nv) = v;
    data.xdot_.segment(nv, nv) = pinocchio::aba(model, d.data_, q, v, d.tau_, f_ext);
}

void ContactFwdDynamics::dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                                  ContinuousDynamicsData &data) const
{
    ContactFwdDynamicsData &d = static_cast<ContactFwdDynamicsData &>(data);

    const Model &model = space_.getModel();
    const int nq = model.nq;
    const int nv = model.nv;

    Eigen::VectorXd X(nq + nv + nv + nv);
    X << x, d.tau_, Eigen::VectorXd::Zero(nv);

    Eigen::VectorXd da_dx_vec = d.ad_fwd_dynamics_.Jacobian(X);
    Eigen::MatrixXd da_dx = da_dx_vec.reshaped(X.size(), nv).transpose();
    Eigen::MatrixXd da_dq = da_dx.rightCols(nv);
    Eigen::MatrixXd da_dv = da_dx.middleCols(nq, nv);
    Eigen::MatrixXd da_dtau = da_dx.middleCols(nq + nv, nv);

    d.Jx_.bottomLeftCorner(nv, nv) = da_dq;
    d.Jx_.bottomRightCorner(nv, nv) = da_dv;
    d.Ju_.bottomRows(nv) = da_dtau * d.dtau_du_;
}

std::shared_ptr<ContinuousDynamicsData> ContactFwdDynamics::createData() const
{
    return std::make_shared<ContactFwdDynamicsData>(*this);
}

ContactFwdDynamicsData::ContactFwdDynamicsData(const ContactFwdDynamics &dynamics)
    : ContinuousDynamicsData(dynamics.ndx(), dynamics.nu()),
      tau_(dynamics.space_.getModel().nv),
      dtau_du_(dynamics.actuation_matrix_)
{
    tau_.setZero();
    const Model &model = dynamics.space_.getModel();
    data_ = Data(model);
    Jx_.topRightCorner(model.nv, model.nv).setIdentity();

    //////////// 创建 CppAD 的正动力学函数 //////////
    using CppAD::AD;
    using ADVectorX = Eigen::VectorX<AD<double>>;

    pinocchio::ModelTpl<AD<double>> ad_model = model.cast<AD<double>>();
    pinocchio::DataTpl<AD<double>> ad_data(ad_model);
    int nq = ad_model.nq;
    int nv = ad_model.nv;

    ADVectorX ad_X(nq + nv + nv + nv); // q, v, tau, dq
    ad_X.setZero();
    CppAD::Independent(ad_X);
    ADVectorX ad_Y(nv);

    ADVectorX ad_q_plus = pinocchio::integrate(ad_model, ad_X.head(nq), ad_X.tail(nv));
    pinocchio::forwardKinematics(ad_model, ad_data, ad_q_plus, ad_X.segment(nq, nv));
    pinocchio::updateFramePlacements(ad_model, ad_data);
    aligned_vector<pinocchio::ForceTpl<AD<double>>> ad_f_ext(ad_model.njoints,
                                                             pinocchio::ForceTpl<AD<double>>::Zero());
    CalcContactForceContributionAD(ad_model, ad_data, ad_f_ext);
    ad_Y = pinocchio::aba(ad_model, ad_data, ad_q_plus, ad_X.segment(nq, nv), ad_X.segment(nq + nv, nv), ad_f_ext);
    ad_fwd_dynamics_.Dependent(ad_X, ad_Y);

    contact_forces_.assign(4, Vector3d::Zero());
}
