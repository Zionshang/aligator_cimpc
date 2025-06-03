#include <aligator/core/traj-opt-problem.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <aligator/modelling/costs/sum-of-costs.hpp>
#include <aligator/modelling/costs/quad-state-cost.hpp>
#include <aligator/modelling/dynamics/integrator-semi-euler.hpp>
#include <aligator/solvers/proxddp/solver-proxddp.hpp>
#include <aligator/modelling/autodiff/finite-difference.hpp>
#include <aligator/modelling/autodiff/cost-finite-difference.hpp>
#include <aligator/modelling/dynamics/integrator-euler.hpp>
#include <aligator/modelling/state-error.hpp>
#include <proxsuite-nlp/modelling/constraints/box-constraint.hpp>

#include <fstream>

#include <pinocchio/algorithm/rnea.hpp>
#include "contact_fwd_dynamics.hpp"
#include "foot_slip_clearance_cost.hpp"
#include "logger.hpp"
#include "yaml_loader.hpp"
#include "symmetric_control_residual.hpp"
#include "contact_assessment.hpp"

using aligator::context::TrajOptProblem;
using StageModel = aligator::StageModelTpl<double>;
using IntegratorSemiImplEuler = aligator::dynamics::IntegratorSemiImplEulerTpl<double>;
using IntegratorEuler = aligator::dynamics::IntegratorEulerTpl<double>;
using CostStack = aligator::CostStackTpl<double>;
using QuadraticStateCost = aligator::QuadraticStateCostTpl<double>;
using QuadraticControlCost = aligator::QuadraticControlCostTpl<double>;
using QuadraticResidualCost = aligator::QuadraticResidualCostTpl<double>;
using CostFiniteDifference = aligator::autodiff::CostFiniteDifferenceHelper<double>;
using ControlErrorResidual = aligator::ControlErrorResidualTpl<double>;
using BoxConstraint = proxsuite::nlp::BoxConstraintTpl<double>;

std::string yaml_filename = "/home/zishang/cpp_workspace/aligator_cimpc/config/parameters_walk.yaml";
YamlLoader yaml_loader(yaml_filename);

double CalcCost(const pinocchio::Model &model,
                const std::vector<VectorXd> &q, const std::vector<VectorXd> &v,
                const std::vector<VectorXd> &tau,
                const std::vector<VectorXd> &q_nom, const std::vector<VectorXd> &v_nom)
{
    double cost = 0;

    int nq = model.nq;
    int nv = model.nv;
    int num_steps = q.size() - 1;

    MatrixXd Qq = MatrixXd::Zero(nq, nq);
    MatrixXd Qv = MatrixXd::Zero(nv, nv);
    MatrixXd Qf_q = MatrixXd::Zero(nq, nq);
    MatrixXd Qf_v = MatrixXd::Zero(nv, nv);
    MatrixXd R = MatrixXd::Zero(nv, nv);

    Qq.diagonal() << yaml_loader.w_pos_body, 1,
        yaml_loader.w_pos_leg, yaml_loader.w_pos_leg, yaml_loader.w_pos_leg, yaml_loader.w_pos_leg;
    Qv.diagonal() << yaml_loader.w_vel_body,
        yaml_loader.w_vel_leg, yaml_loader.w_vel_leg, yaml_loader.w_vel_leg, yaml_loader.w_vel_leg;
    Qf_q.diagonal() << yaml_loader.w_pos_body_term, 10,
        yaml_loader.w_pos_leg_term, yaml_loader.w_pos_leg_term, yaml_loader.w_pos_leg_term, yaml_loader.w_pos_leg_term;
    Qf_v.diagonal() << yaml_loader.w_vel_body_term,
        yaml_loader.w_vel_leg_term, yaml_loader.w_vel_leg_term, yaml_loader.w_vel_leg_term, yaml_loader.w_vel_leg_term;
    R.diagonal() << 0, 0, 0, 0, 0, 0,
        yaml_loader.w_u_leg, yaml_loader.w_u_leg, yaml_loader.w_u_leg, yaml_loader.w_u_leg;

    std::cout << "Qq:   " << Qq.diagonal().transpose() << std::endl;
    std::cout << "Qv:   " << Qv.diagonal().transpose() << std::endl;
    std::cout << "Qf_q: " << Qf_q.diagonal().transpose() << std::endl;
    std::cout << "Qf_v: " << Qf_v.diagonal().transpose() << std::endl;
    std::cout << "R:    " << R.diagonal().transpose() << std::endl;

    VectorXd q_err(nq), v_err(nv);

    // Running cost
    for (int t = 0; t < num_steps; ++t)
    {
        q_err = q[t] - q_nom[t];
        v_err = v[t] - v_nom[t];
        cost += double(q_err.transpose() * Qq * q_err);
        cost += double(v_err.transpose() * Qv * v_err);
        cost += double(tau[t].transpose() * R * tau[t]);
    }

    // Terminal cost
    q_err = q[num_steps] - q_nom[num_steps];
    v_err = v[num_steps] - v_nom[num_steps];
    cost += double(q_err.transpose() * Qf_q * q_err);
    cost += double(v_err.transpose() * Qf_v * v_err);
    return cost;
}

std::shared_ptr<TrajOptProblem> createTrajOptProblem(const ContactFwdDynamics &dynamics,
                                                     int nsteps, double timestep,
                                                     const std::vector<VectorXd> &x_ref,
                                                     const std::vector<VectorXd> &u_ref,
                                                     const VectorXd &x0)
{
    const auto space = dynamics.space();
    const int nu = dynamics.nu(); // Number of control inputs
    const int ndx = space.ndx();  // Number of state variables

    // Define stage state weights
    VectorXd w_x_diag(ndx);
    w_x_diag << yaml_loader.w_pos_body,
        yaml_loader.w_pos_leg, yaml_loader.w_pos_leg, yaml_loader.w_pos_leg, yaml_loader.w_pos_leg,
        yaml_loader.w_vel_body,
        yaml_loader.w_vel_leg, yaml_loader.w_vel_leg, yaml_loader.w_vel_leg, yaml_loader.w_vel_leg;
    MatrixXd w_x = w_x_diag.asDiagonal();

    // Define terminal state weights
    w_x_diag << yaml_loader.w_pos_body_term,
        yaml_loader.w_pos_leg_term, yaml_loader.w_pos_leg_term, yaml_loader.w_pos_leg_term, yaml_loader.w_pos_leg_term,
        yaml_loader.w_vel_body_term,
        yaml_loader.w_vel_leg_term, yaml_loader.w_vel_leg_term, yaml_loader.w_vel_leg_term, yaml_loader.w_vel_leg_term;
    MatrixXd w_x_term = w_x_diag.asDiagonal();

    // Define input state weights
    VectorXd w_u_diag(nu);
    w_u_diag << yaml_loader.w_u_leg, yaml_loader.w_u_leg, yaml_loader.w_u_leg, yaml_loader.w_u_leg;
    MatrixXd w_u = w_u_diag.asDiagonal();

    IntegratorSemiImplEuler discrete_dyn = IntegratorSemiImplEuler(dynamics, timestep);

    std::vector<xyz::polymorphic<StageModel>> stage_models;
    FootSlipClearanceCost fscc(space, nu, yaml_loader.w_foot_slip_clearance, -30.0);
    CostFiniteDifference fscc_fini_diff(fscc, 1e-6);
    ControlErrorResidual control_error(space.ndx(), nu);
    VectorXd u_max = space.getModel().effortLimit.tail(nu);
    SymmetricControlResidual symmetric_control_residual(ndx, nu);

    for (size_t i = 0; i < nsteps; i++)
    {
        auto rcost = CostStack(space, nu);

        rcost.addCost("state_cost", QuadraticStateCost(space, nu, x_ref[i], w_x));
        rcost.addCost("control_cost", QuadraticControlCost(space, u_ref[i], w_u));
        // rcost.addCost("foot_slip_clearance_cost", fscc_fini_diff);
        // rcost.addCost("symmetric_control_cost",
        //               QuadraticResidualCost(space, symmetric_control_residual,
        //                                     yaml_loader.w_symmetric_control * MatrixXd::Identity(4, 4)));

        StageModel sm = StageModel(rcost, discrete_dyn);

        sm.addConstraint(control_error, BoxConstraint(-u_max, u_max));
        stage_models.push_back(std::move(sm));
    }

    auto term_cost = CostStack(space, nu);
    term_cost.addCost("term_state_cost", QuadraticStateCost(space, nu, x_ref[nsteps], w_x_term));

    return std::make_shared<TrajOptProblem>(x0, stage_models, term_cost);
}

void updateStateReferences(std::shared_ptr<TrajOptProblem> problem,
                           const std::vector<VectorXd> &x_ref)
{
    for (size_t i = 0; i < problem->numSteps(); i++)
    {
        CostStack *cs = dynamic_cast<CostStack *>(&*problem->stages_[i]->cost_);
        QuadraticStateCost *qsc = cs->getComponent<QuadraticStateCost>("state_cost");
        qsc->setTarget(x_ref[i]);
    }
    CostStack *cs = dynamic_cast<CostStack *>(&*problem->term_cost_);
    QuadraticStateCost *qsc = cs->getComponent<QuadraticStateCost>("term_state_cost");
    qsc->setTarget(x_ref.back());
}

int main(int argc, char const *argv[])
{

    std::string urdf_filename = "/home/zishang/cpp_workspace/aligator_cimpc/robot/mini_cheetah/urdf/mini_cheetah_ground_mesh.urdf";

    Model model;
    pinocchio::urdf::buildModel(urdf_filename, model);
    MultibodyPhaseSpace space(model);
    const int nu = model.nv - 6;
    const int nq = model.nq;
    const int nv = model.nv;
    MatrixXd actuation = MatrixXd::Zero(model.nv, nu);
    actuation.bottomRows(nu).setIdentity();
    ContactParams<double> contact_params = yaml_loader.real_contact_params;
    ContactFwdDynamics dynamics(space, actuation, contact_params);

    int nsteps = yaml_loader.nsteps;
    double timestep = yaml_loader.timestep;

    /************************用idto的cost计算idto的结果**********************/
    std::vector<VectorXd> q_idto = readVectorsFromCsv("/home/zishang/cpp_workspace/aligator_cimpc/log/q.csv");
    std::vector<VectorXd> v_idto = readVectorsFromCsv("/home/zishang/cpp_workspace/aligator_cimpc/log/v.csv");
    std::vector<VectorXd> tau_idto = readVectorsFromCsv("/home/zishang/cpp_workspace/aligator_cimpc/log/tau.csv");
    std::vector<VectorXd> q_nom = readVectorsFromCsv("/home/zishang/cpp_workspace/aligator_cimpc/log/q_nom.csv");
    std::vector<VectorXd> v_nom = readVectorsFromCsv("/home/zishang/cpp_workspace/aligator_cimpc/log/v_nom.csv");
    double cost_idto = CalcCost(model, q_idto, v_idto, tau_idto, q_nom, v_nom);
    std::cout << "用idto的cost计算idto的结果: " << cost_idto << std::endl;

    /************************initial state**********************/
    std::vector<VectorXd> q0_vec = readVectorsFromCsv("/home/zishang/cpp_workspace/aligator_cimpc/log/q0.csv");
    std::vector<VectorXd> v0_vec = readVectorsFromCsv("/home/zishang/cpp_workspace/aligator_cimpc/log/v0.csv");
    VectorXd x0 = VectorXd::Zero(nq + nv);
    x0.head(nq) = q0_vec[0];
    x0.tail(nv) = v0_vec[0];
    x0.segment(3, 4).normalize();

    std::vector<VectorXd> q_guess = readVectorsFromCsv("/home/zishang/cpp_workspace/aligator_cimpc/log/q_guess.csv");
    std::vector<VectorXd> v_guess = readVectorsFromCsv("/home/zishang/cpp_workspace/aligator_cimpc/log/v_guess.csv");
    std::vector<VectorXd> tau_guess = readVectorsFromCsv("/home/zishang/cpp_workspace/aligator_cimpc/log/tau_guess.csv");
    std::vector<VectorXd> x_guess(nsteps + 1, VectorXd::Zero(nq + nv));
    std::vector<VectorXd> u_guess(nsteps, VectorXd::Zero(nu));
    std::vector<VectorXd> x_ref(nsteps + 1, VectorXd::Zero(nq + nv));
    std::vector<VectorXd> u_ref(nsteps, VectorXd::Zero(nu));

    for (int i = 0; i < nsteps + 1; ++i)
    {
        x_guess[i].head(nq) = q_guess[i];
        x_guess[i].tail(nv) = v_guess[i];
        x_ref[i].head(nq) = pinocchio::neutral(model);
        x_ref[i].tail(nv) = v_nom[i];
    }

    for (int i = 0; i < nsteps; ++i)
    {
        u_guess[i] = tau_guess[i].tail(nu);
    }

    /************************create problem**********************/
    auto problem = createTrajOptProblem(dynamics, nsteps, timestep, x_ref, u_ref, x0);
    double tol = 1e-4;
    int max_iters = 250;
    double mu_init = 1e-8;
    aligator::SolverProxDDPTpl<double> solver(tol, mu_init, max_iters, aligator::VerboseLevel::VERBOSE);
    solver.rollout_type_ = aligator::RolloutType::LINEAR;
    solver.force_initial_condition_ = true;
    solver.linear_solver_choice = aligator::LQSolverChoice::PARALLEL;
    solver.setNumThreads(8);
    solver.setup(*problem);
    solver.run(*problem, x_guess, u_guess);

    std::vector<VectorXd> q_aligator, v_aligator, tau_aligator;
    for (const auto &x : solver.results_.xs)
    {
        q_aligator.push_back(x.head(nq));
        v_aligator.push_back(x.tail(nv));
    }
    for (const auto &u : solver.results_.us)
    {
        VectorXd tau = VectorXd::Zero(model.nv);
        tau.tail(nu) = u;
        tau_aligator.push_back(tau);
    }

    double cost_aligator = CalcCost(model, q_aligator, v_aligator, tau_aligator, q_nom, v_nom);
    std::cout << "用idto的cost计算aligator的结果: " << cost_aligator << std::endl;

    std::cout << "q_aligator:" << std::endl;
    for (const auto &q : q_aligator)
    {
        std::cout << q.transpose() << std::endl;
    }

    return 0;
}
