#pragma once
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <proxsuite/proxqp/dense/dense.hpp>

using namespace proxsuite;

using ConstVectorRef = Eigen::Ref<const Eigen::VectorXd>;

struct RelaxedWbcSettings
{
    std::vector<pinocchio::FrameIndex> contact_ids; //< Index of contacts
    double mu;                                      //< Friction parameter
    long force_size;                                //< Dimension of contact forces
    double w_acc;                                   //< Weight for acceleration regularization
    double w_force;                                 //< Weight for force regularization
    bool verbose;                                   //< Print solver information
};

class RelaxedWbc
{
public:
    explicit RelaxedWbc(const RelaxedWbcSettings &settings, const pinocchio::Model &model);

    void solveQP(const std::vector<bool> &contact_state,
                 const ConstVectorRef &q,
                 const ConstVectorRef &v,
                 const ConstVectorRef &a,
                 const ConstVectorRef &tau,
                 const ConstVectorRef &forces);

    Eigen::MatrixXd getA()
    {
        return qp_.model.A;
    }
    Eigen::MatrixXd getH()
    {
        return qp_.model.H;
    }
    Eigen::MatrixXd getC()
    {
        return qp_.model.C;
    }
    Eigen::VectorXd getg()
    {
        return qp_.model.g;
    }
    Eigen::VectorXd getb()
    {
        return qp_.model.b;
    }

    // QP results
    Eigen::VectorXd solved_forces_;
    Eigen::VectorXd solved_acc_;
    Eigen::VectorXd solved_torque_;

private:
    RelaxedWbcSettings settings_;
    proxqp::dense::QP<double> qp_;
    pinocchio::Model model_;
    pinocchio::Data data_;

    int force_dim_;
    int nforcein_;
    int nk_;
    int nv_;
    int nq_;

    Eigen::MatrixXd H_;
    Eigen::MatrixXd A_;
    Eigen::MatrixXd C_;
    Eigen::MatrixXd S_;
    Eigen::MatrixXd Cmin_;
    Eigen::VectorXd b_;
    Eigen::VectorXd g_;
    Eigen::VectorXd l_;
    Eigen::VectorXd u_;

    Eigen::MatrixXd Jc_;
    Eigen::VectorXd Jdot_v_;
    Eigen::MatrixXd Jdot_;

    // Internal matrix computation
    void computeMatrices(const std::vector<bool> &contact_state,
                         const ConstVectorRef &v,
                         const ConstVectorRef &a,
                         const ConstVectorRef &tau,
                         const ConstVectorRef &forces);

    void updatePinocchioData(const ConstVectorRef &q, const ConstVectorRef &v);
};
