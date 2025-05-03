#pragma once
#include <aligator/modelling/dynamics/ode-abstract.hpp>
#include <proxsuite-nlp/modelling/spaces/multibody.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/autodiff/cppad.hpp>

#include "contact_force.hpp"

using ODEAbstract = aligator::dynamics::ODEAbstractTpl<double>;
using ContinuousDynamicsData = aligator::dynamics::ContinuousDynamicsDataTpl<double>;
using MultibodyPhaseSpace = proxsuite::nlp::MultibodyPhaseSpace<double>;

using Eigen::MatrixXd;
using Eigen::VectorXd;

struct CompliantContactFwdData;

struct ContactFwdDynamics : ODEAbstract
{
    ContactFwdDynamics(const MultibodyPhaseSpace &space, const MatrixXd &actuation);

    const MultibodyPhaseSpace &space() const { return space_; }

    void forward(const ConstVectorRef &x, const ConstVectorRef &u,
                 ContinuousDynamicsData &data) const override;

    void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                  ContinuousDynamicsData &data) const override;

    // 必须得有，否则会报错
    std::shared_ptr<ContinuousDynamicsData> createData() const;

    MultibodyPhaseSpace space_; // 存储着pinocchio模型
    MatrixXd actuation_matrix_;
};

struct ContactFwdDynamicsData : ContinuousDynamicsData
{
    ContactFwdDynamicsData(const ContactFwdDynamics &dynamics);

    VectorXd tau_;
    Data data_;
    MatrixXd dtau_du_;
    CppAD::ADFun<double> ad_fwd_dynamics_;
    std::vector<Vector3d> contact_forces_; // 仅仅用于记录数据便于debug
};
