#include "gait_phase_alignment.hpp"
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>

GaitAlignmentResidual::GaitAlignmentResidual(const pinocchio::Model model, int ndx, int nu,
                                             std::vector<double> phi, double T)
    : StageFunction(ndx, nu, phi.size()),
      model_(model), phi_(phi), T_(T), t_(0.0) {}

void GaitAlignmentResidual::evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                                     StageFunctionData &data) const
{
    GaitAlignmentResidualData &d = static_cast<GaitAlignmentResidualData &>(data);

    const int nq = model_.nq;
    const auto &q = x.head(nq);
    const int num_feet = foot_frame_ids_.size();

    pinocchio::forwardKinematics(model_, d.data_, q);
    pinocchio::updateFramePlacements(model_, d.data_);
    for (size_t i = 0; i < num_feet; i++)
    {
        // 计算 signed distance
        double signed_distance = d.data_.oMf[foot_frame_ids_[i]].translation()(2) - foot_radius_;

        // 计算 gait phase alignment
        d.value_(i) = calcGaitAlignment(signed_distance, t_, T_, phi_[i], slope_, offset_) - d_lower_;
    }
}

void GaitAlignmentResidual::computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                                             StageFunctionData &data) const
{
    GaitAlignmentResidualData &d = static_cast<GaitAlignmentResidualData &>(data);

    const int nv = model_.nv;
    const int num_feet = foot_frame_ids_.size();

    for (size_t i = 0; i < num_feet; i++)
    {
        // 计算 signed distance 关于 q 的导数
        pinocchio::computeJointJacobians(model_, d.data_);
        pinocchio::getFrameJacobian(model_, d.data_, foot_frame_ids_[i],
                                    pinocchio::LOCAL_WORLD_ALIGNED, d.J_feet[i]);
        d.ds_dq.row(i) = d.J_feet[i].row(2);

        // 计算 gait phase alignment 关于 signed distance 的导数
        double signed_distance = d.data_.oMf[foot_frame_ids_[i]].translation()(2) - foot_radius_;
        d.df_ds(i, i) = calcGaitAlignmentDerivative(signed_distance,
                                                    t_, T_, phi_[i], slope_, offset_);
    }

    d.Jx_.leftCols(nv) = d.df_ds * d.ds_dq;
}

std::shared_ptr<StageFunctionData> GaitAlignmentResidual::createData() const
{
    return std::make_shared<GaitAlignmentResidualData>(*this);
}

GaitAlignmentResidualData::GaitAlignmentResidualData(const GaitAlignmentResidual &function)
    : StageFunctionData(function)
{
    const int nv = function.model_.nv;
    const int num_feet = function.foot_frame_ids_.size();

    data_ = pinocchio::Data(function.model_);
    J_feet.assign(num_feet, Eigen::MatrixXd::Zero(6, nv));
    ds_dq.setZero(num_feet, nv);
    df_ds.setZero(num_feet, num_feet);
}
