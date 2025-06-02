#include "symmetric_control_residual.hpp"

SymmetricControlResidual::SymmetricControlResidual(int ndx, int nu)
    : StageFunction(ndx, nu, 4)
{
    D_ = Eigen::MatrixXd::Zero(2, 3);
    D_.rightCols(2).setIdentity();

    Eigen::MatrixXd zero_mat = Eigen::MatrixXd::Zero(2, 3);
    C_ = Eigen::MatrixXd::Zero(4, nu);
    C_ << D_, zero_mat, zero_mat, -D_,
        zero_mat, D_, -D_, zero_mat;
}

void SymmetricControlResidual::evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                                        StageFunctionData &data) const
{
    const Eigen::VectorXd &tau = u;

    data.value_ = C_ * tau;
}

void SymmetricControlResidual::computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                                                StageFunctionData &data) const
{
    int nu = u.size();
    data.Ju_.rightCols(nu) = C_; // todo: 提前赋值
}