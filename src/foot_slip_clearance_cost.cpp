#include "foot_slip_clearance_cost.hpp"

FootSlipClearanceCost::FootSlipClearanceCost(MultibodyPhaseSpace space,
                                             int nu, double cf, double c1)
    : CostAbstract(space, nu), space_(space), cf_(cf), c1_(c1) {}

void FootSlipClearanceCost::evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                                     CostData &data) const
{
    FootSlipClearanceCostData &d = static_cast<FootSlipClearanceCostData &>(data);

    const int num_feet = 4;
    double cost = 0.0;

    const auto &model = space_.getModel();
    pinocchio::forwardKinematics(model, d.data_, x.head(model.nq), x.tail(model.nv));
    pinocchio::updateFramePlacements(model, d.data_);

    for (int k = 0; k < num_feet; ++k)
    {
        // 脚的离地高度
        double phi = d.data_.oMf[foot_frame_ids_[k]].translation()(foot_height_idx_);

        // 脚的切向速度
        Vector3d v = pinocchio::getFrameVelocity(model, d.data_, foot_frame_ids_[k], pinocchio::LOCAL_WORLD_ALIGNED).linear();
        Vector3d v_t = v - v.dot(nhat_) * nhat_;

        double s = 1.0 / (1.0 + std::exp(-c1_ * phi)); // Sigmoid
        cost += s * v_t.squaredNorm();
    }

    data.value_ = cf_ * cost;
}

void FootSlipClearanceCost::computeGradients(const ConstVectorRef &x, const ConstVectorRef &u,
                                             CostData &data) const
{
}

void FootSlipClearanceCost::computeHessians(const ConstVectorRef &x, const ConstVectorRef &u,
                                            CostData &data) const
{
}

std::shared_ptr<CostData> FootSlipClearanceCost::createData() const
{
    return std::make_shared<FootSlipClearanceCostData>(*this);
}

FootSlipClearanceCostData::FootSlipClearanceCostData(const FootSlipClearanceCost &cost)
    : CostData(cost)
{
    data_ = pinocchio::Data(cost.space_.getModel());

    //////////// 创建 CppAD 的cost函数 //////////
    using CppAD::AD;
    using ADVectorX = Eigen::VectorX<AD<double>>;

    const auto &model = cost.space_.getModel();
    pinocchio::ModelTpl<AD<double>> ad_model = model.cast<AD<double>>();
    pinocchio::DataTpl<AD<double>> ad_data(ad_model);
    int nq = ad_model.nq;
    int nv = ad_model.nv;
    ADVectorX ad_X(nq + nv + nv); // q, v, dq
    ad_X.setZero();
    CppAD::Independent(ad_X);
    ADVectorX ad_Y(1);

    const int num_feet = 4;
    AD<double> ad_cost = AD<double>(0.0);

    ADVectorX ad_q = ad_X.head(nq);
    ADVectorX ad_v = ad_X.segment(nq, nv);
    ADVectorX ad_dq = ad_X.tail(nv);

    ADVectorX ad_q_plus = pinocchio::integrate(ad_model, ad_q, ad_dq);
    pinocchio::forwardKinematics(ad_model, ad_data, ad_q_plus, ad_v);
    pinocchio::updateFramePlacements(ad_model, ad_data);

    ADVectorX ad_nhat = cost.nhat_.cast<AD<double>>();
    for (int k = 0; k < num_feet; ++k)
    {
        // 脚的离地高度
        AD<double> phi = ad_data.oMf[cost.foot_frame_ids_[k]].translation()(cost.foot_height_idx_);

        // 脚的切向速度
        ADVectorX v = pinocchio::getFrameVelocity(ad_model, ad_data, cost.foot_frame_ids_[k], pinocchio::LOCAL_WORLD_ALIGNED).linear();
        ADVectorX v_t = v - v.dot(ad_nhat) * ad_nhat;

        AD<double> s = 1.0 / (1.0 + CppAD::exp(-cost.c1_ * phi)); // Sigmoid
        ad_cost += s * v_t.squaredNorm();
    }

    ad_Y << cost.cf_ * ad_cost;
    ad_cost_fun_.Dependent(ad_X, ad_Y);
}
