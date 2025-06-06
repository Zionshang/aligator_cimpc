#pragma once

#include <aligator/core/function-abstract.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

using StageFunction = aligator::StageFunctionTpl<double>;
using StageFunctionData = aligator::StageFunctionDataTpl<double>;

template <typename Scalar>
Scalar calcGaitAlignment(Scalar signed_distance, Scalar t,
                         Scalar T, Scalar phi,
                         Scalar slope, Scalar offset)
{
    using std::pow;
    using std::sin;
    using std::tanh;

    // contact indicator
    Scalar c = 0.5 * (1 - tanh(slope * (signed_distance - offset)));

    // phase function
    Scalar g = sin(2 * M_PI * t / T + phi);

    // violation cost
    Scalar f = (Scalar(2) * Scalar(c) - Scalar(1)) * g;

    return f;
}

template <typename Scalar>
Scalar calcGaitAlignmentDerivative(Scalar signed_distance, Scalar t,
                                   Scalar T, Scalar phi,
                                   Scalar slope, Scalar offset)
{
    using std::pow;
    using std::sin;
    using std::tanh;

    // contact indicator derivative
    Scalar tmp = tanh(slope * (signed_distance - offset));
    Scalar dc_ds = -0.5 * slope * (1 - tmp * tmp);

    // phase function
    Scalar g = sin(2 * M_PI * t / T + phi);

    // derivative of violation cost w.r.t signed_distance
    Scalar df_ds = Scalar(2) * dc_ds * g;

    return df_ds;
}

struct GaitAlignmentResidual: StageFunction
{
    GaitAlignmentResidual(const pinocchio::Model model, int ndx, int nu,
                          std::vector<double> phi, double T);

    void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                  StageFunctionData &data) const;

    void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                          StageFunctionData &data) const;

    void updateTime(double t) { t_ = t; }

    std::shared_ptr<StageFunctionData> createData() const;

    pinocchio::Model model_;  // 步态偏移量
    std::vector<double> phi_; // 步态偏移量
    double T_;                // 步态周期
    double t_;                // 当前时间

    // todo: 作为参数传递
    double d_lower_ = -0.6; // gait 违反程度的最低可接受门槛
    double slope_ = 10.0;
    double offset_ = 0.1;
    std::vector<pinocchio::FrameIndex> foot_frame_ids_{11, 19, 27, 35};
    double foot_radius_ = 0.0175;
};

struct GaitAlignmentResidualData : StageFunctionData
{
    GaitAlignmentResidualData(const GaitAlignmentResidual &function);

    pinocchio::Data data_;
    std::vector<Eigen::Matrix<double, 6, Eigen::Dynamic>> J_feet;
    Eigen::MatrixXd ds_dq; // signed distance 关于 q 的导数
    Eigen::MatrixXd df_ds; // gait phase alignment 关于 signed distance 的导数
};