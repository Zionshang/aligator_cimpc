#pragma once
#include <aligator/core/cost-abstract.hpp>
#include <proxsuite-nlp/modelling/spaces/multibody.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>

using CostAbstract = aligator::CostAbstractTpl<double>;
using MultibodyPhaseSpace = proxsuite::nlp::MultibodyPhaseSpace<double>;
using CostData = aligator::CostDataAbstractTpl<double>;

using Eigen::Vector3d;

struct FootSlipClearanceCostData;

struct FootSlipClearanceCost : CostAbstract
{

    FootSlipClearanceCost(MultibodyPhaseSpace space,
                          int nu, double cf, double c1);

    void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                  CostData &data) const override;

    void computeGradients(const ConstVectorRef &x, const ConstVectorRef &u,
                          CostData &data) const override;

    void computeHessians(const ConstVectorRef &x, const ConstVectorRef &u,
                         CostData &data) const override;

    std::shared_ptr<CostData> createData() const override;

    MultibodyPhaseSpace space_;

    int foot_height_idx_ = 2;           // 可根据实际数据布局设置
    Vector3d nhat_ = Vector3d(0, 0, 1); // 脚和地面之间的法向量

    double cf_ = 1.0; // 总成本系数
    double c1_ = -30; // Sigmoid 函数陡峭程度（负值）

    std::vector<int> foot_frame_ids_{11, 19, 27, 35};
};

struct FootSlipClearanceCostData : CostData
{
    FootSlipClearanceCostData(const FootSlipClearanceCost &cost);

    pinocchio::Data data_;
};