#pragma once
#include <Eigen/Dense>
#include "contact_fwd_dynamics.hpp"

using Eigen::Vector4d;
class ContactAssessment
{
public:
    ContactAssessment(const ContactFwdDynamics &dynamics)
        : dynamics_(dynamics), dyn_data_(dynamics)
    {
        contact_state_.assign(dyn_data_.contact_forces_.size(), 0);
    }

    void update(VectorXd x)
    {
        dynamics_.forward(x, VectorXd::Zero(dynamics_.space().getModel().nv), dyn_data_);
        for (size_t i = 0; i < dyn_data_.contact_forces_.size(); i++)
        {
            if (dyn_data_.contact_forces_[i].norm() > 0.1)
                contact_state_[i] = 1;
            else
                contact_state_[i] = 0;
        }
    }

    const std::vector<bool> &contact_state() const { return contact_state_; }
    const std::vector<Vector3d> &contact_forces() const { return dyn_data_.contact_forces_; }

private:
    ContactFwdDynamics dynamics_;
    ContactFwdDynamicsData dyn_data_; // 用于打印当前地面接触力
    std::vector<bool> contact_state_;
};