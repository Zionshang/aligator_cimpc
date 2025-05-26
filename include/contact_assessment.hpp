#pragma once
#include <Eigen/Dense>
#include "contact_force.hpp"

using Eigen::Vector4d;
class ContactAssessment
{
public:
    ContactAssessment(const pinocchio::Model &model, ContactParams<double> contact_params)
        : model_(model), data_(model), contact_params_(contact_params)
    {
        contact_state_.assign(4, 0);
        contact_forces_.assign(4, Vector3d::Zero());
    }

    void update(const VectorXd &q, const VectorXd &v)
    {
        pinocchio::forwardKinematics(model_, data_, q, v);
        pinocchio::updateFramePlacements(model_, data_);
        aligned_vector<pinocchio::Force> f_ext(model_.njoints, pinocchio::Force::Zero()); // todo: 删除临时变量
        CalcContactForceContribution(model_, data_, f_ext, contact_params_, contact_forces_);

        for (size_t i = 0; i < contact_forces_.size(); i++)
        {
            if (contact_forces_[i].norm() > 0.5)
                contact_state_[i] = 1;
            else
                contact_state_[i] = 0;
        }
        f_ext_ = f_ext;
    }

    const std::vector<bool> &contact_state() const { return contact_state_; }
    const std::vector<Vector3d> &contact_forces() const { return contact_forces_; }
    const aligned_vector<pinocchio::Force> &f_ext() const { return f_ext_; }

private:
    pinocchio::Model model_;
    pinocchio::Data data_;
    ContactParams<double> contact_params_;
    std::vector<bool> contact_state_;
    std::vector<Vector3d> contact_forces_;
    aligned_vector<pinocchio::Force> f_ext_; // todo: 删除临时变量
};