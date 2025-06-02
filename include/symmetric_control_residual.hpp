#pragma once

#include <aligator/core/function-abstract.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/autodiff/cppad.hpp>

using StageFunction = aligator::StageFunctionTpl<double>;
using StageFunctionData = aligator::StageFunctionDataTpl<double>;

struct SymmetricControlResidual : StageFunction
{
    SymmetricControlResidual(int ndx, int nu);

    void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                  StageFunctionData &data) const;

    void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                          StageFunctionData &data) const;

    // TODO: 重命名
    Eigen::MatrixXd D_; // 选择矩阵
    Eigen::MatrixXd C_; // 选择矩阵
};