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
}
