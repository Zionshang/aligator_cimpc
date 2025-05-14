#include "contact_inv_dynamics_residual.hpp"

#include <pinocchio/algorithm/rnea.hpp>

ContactInvDynamicsResidual::ContactInvDynamicsResidual(int ndx, const Model &model, MatrixXd actuation,
                                                       ContactParams<double> contact_params)
    : StageFunction(ndx, actuation.cols(), model.nv),
      model_(model), actuation_matrix_(actuation), contact_params_(contact_params)
{
    const int nv = model_.nv;
    if (nv != actuation.rows())
    {
        ALIGATOR_DOMAIN_ERROR(
            fmt::format("actuation matrix should have number of rows = pinocchio "
                        "model nv ({} and {}).",
                        actuation.rows(), nv));
    }
}

void ContactInvDynamicsResidual::evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                                          StageFunctionData &data)
{
    ContactInvDynamicsResidualData &d = static_cast<ContactInvDynamicsResidualData &>(data);

    const int nq = model.nq;
    const int nv = model.nv;
    const auto &q = x.head(nq);
    const auto &v = x.segment(nq, nv);
    const auto &a = u;

    // 计算接触力
    pinocchio::forwardKinematics(model_, d.data_, q, v);
    pinocchio::updateFramePlacements(model_, d.data_);
    aligned_vector<pinocchio::Force> f_ext(model_.njoints, pinocchio::Force::Zero()); // todo: 删除临时变量
    CalcContactForceContribution(model_, d.data_, f_ext, contact_params_, d.contact_forces_);

    // 逆动力学计算
    d.tau_ = pinocchio::rnea(model_, d.data_, q, v, a, f_ext);

    data.value_ = d.tau_ 
}

void ContactInvDynamicsResidual::computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                                                  StageFunctionData &data)
{
}

shared_ptr<StageFunctionData> ContactInvDynamicsResidual::createData()
{
}

struct ContactInvDynamicsResidualData : StageFunctionData
{
    ContactInvDynamicsResidualData(const GravityCompensationResidualTpl &resdl);

    VectorXd tau_;
    pinocchio::Data data_;
    MatrixXd dtau_du_;
    CppAD::ADFun<double> ad_inv_dynamics_;
    std::vector<Vector3d> contact_forces_; // 仅仅用于记录数据便于debug
};