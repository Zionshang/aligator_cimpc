#include <aligator/core/traj-opt-problem.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <aligator/modelling/costs/sum-of-costs.hpp>
#include <aligator/modelling/costs/quad-state-cost.hpp>
#include <aligator/modelling/dynamics/integrator-semi-euler.hpp>
#include <aligator/solvers/proxddp/solver-proxddp.hpp>
#include <aligator/modelling/autodiff/finite-difference.hpp>
#include <aligator/modelling/autodiff/cost-finite-difference.hpp>
#include <aligator/modelling/state-error.hpp>
#include <proxsuite-nlp/modelling/constraints/box-constraint.hpp>
#include <aligator/modelling/costs/quad-residual-cost.hpp>
#include <aligator/modelling/function-xpr-slice.hpp>
#include <aligator/core/stage-data.hpp>

#include <fstream>

#include <pinocchio/algorithm/rnea.hpp>
#include "contact_fwd_dynamics.hpp"
#include "foot_slip_clearance_cost.hpp"
#include "logger.hpp"
#include "yaml_loader.hpp"
#include "contact_inv_dynamics_residual.hpp"
#include "contact_inv_dynamics_residual2.hpp"
#include "kinematics_ode.hpp"
#include "contact_assessment.hpp"

using aligator::context::TrajOptProblem;
using StageModel = aligator::StageModelTpl<double>;
using IntegratorSemiImplEuler = aligator::dynamics::IntegratorSemiImplEulerTpl<double>;
using CostStack = aligator::CostStackTpl<double>;
using QuadraticStateCost = aligator::QuadraticStateCostTpl<double>;
using QuadraticControlCost = aligator::QuadraticControlCostTpl<double>;
using CostFiniteDifference = aligator::autodiff::CostFiniteDifferenceHelper<double>;
using ControlErrorResidual = aligator::ControlErrorResidualTpl<double>;
using BoxConstraint = proxsuite::nlp::BoxConstraintTpl<double>;
using QuadraticResidualCost = aligator::QuadraticResidualCostTpl<double>;
using FunctionSliceXpr = aligator::FunctionSliceXprTpl<double>;
using EqualityConstraint = proxsuite::nlp::EqualityConstraintTpl<double>;

std::string yaml_filename = "/home/zishang/cpp_workspace/aligator_cimpc/config/parameters.yaml";
YamlLoader yaml_loader(yaml_filename);

double CalcCost(const pinocchio::Model &model,
                const std::vector<VectorXd> &q, const std::vector<VectorXd> &v,
                const std::vector<VectorXd> &tau,
                const std::vector<VectorXd> &q_nom, const std::vector<VectorXd> &v_nom,
                const MatrixXd &Qq, const MatrixXd &Qv, const MatrixXd &R,
                const MatrixXd &Qf_q, const MatrixXd &Qf_v)
{
    double cost = 0;

    int nq = model.nq;
    int nv = model.nv;
    int num_steps = q.size() - 1;

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

std::shared_ptr<TrajOptProblem> createTrajOptProblem(const KinematicsODE &kinematics,
                                                     const Model &model,
                                                     int nsteps, double timestep,
                                                     const std::vector<VectorXd> &x_ref,
                                                     const std::vector<VectorXd> &u_ref,
                                                     const VectorXd &x0)
{
    const auto space = kinematics.space();
    const int num_actuated = model.nv - 6; // Number of actuated joints
    const int nu = kinematics.nu();        // Number of control inputs
    const int ndx = space.ndx();           // Number of state variables
    const int nv = model.nv;

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
    w_u_diag << Eigen::VectorXd::Zero(nv), yaml_loader.w_u_leg, yaml_loader.w_u_leg, yaml_loader.w_u_leg, yaml_loader.w_u_leg;
    MatrixXd w_u = w_u_diag.asDiagonal();

    IntegratorSemiImplEuler discrete_dyn = IntegratorSemiImplEuler(kinematics, timestep);
    // IntegratorEuler discrete_dyn = IntegratorEuler(dynamics, timestep);

    std::vector<xyz::polymorphic<StageModel>> stage_models;
    MatrixXd actuation = MatrixXd::Zero(model.nv, num_actuated);
    actuation.bottomRows(num_actuated).setIdentity();
    ContactInvDynamicsResidual contact_inv_dynamics_residual(ndx, model, actuation, yaml_loader.real_contact_params);
    ContactInvDynamicsResidual2 contact_inv_dynamics_residual2(ndx, model, actuation, timestep, yaml_loader.real_contact_params);

    for (size_t i = 0; i < nsteps; i++)
    {
        auto rcost = CostStack(space, nu);
        rcost.addCost("state_cost", QuadraticStateCost(space, nu, x_ref[i], w_x));
        rcost.addCost("control_cost", QuadraticControlCost(space, u_ref[i], w_u));

        StageModel sm = StageModel(rcost, discrete_dyn);

        sm.addConstraint(contact_inv_dynamics_residual2, EqualityConstraint());
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
    qsc->setTarget(x_ref[problem->numSteps()]);
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

    ///////////////////// 构建运动学ODE /////////////////////
    KinematicsODE kinematics(space);

    int nsteps = yaml_loader.nsteps;
    double timestep = yaml_loader.timestep;

    /************************initial state**********************/
    VectorXd x0 = VectorXd::Zero(nq + nv);
    x0.head(nq) << 0.0, 0.0, 0.29,
        0.0, 0.0, 0.0, 1.0,
        0.0, -0.8, 1.6,
        0.0, -0.8, 1.6,
        0.0, -0.8, 1.6,
        0.0, -0.8, 1.6;

    std::vector<VectorXd> x_ref(nsteps + 1, x0);
    VectorXd u_nom = VectorXd::Zero(nv + nu);
    std::vector<VectorXd> u_ref(nsteps, u_nom);

    /************************create problem**********************/
    auto problem = createTrajOptProblem(kinematics, model, nsteps, timestep, x_ref, u_ref, x0);
    double tol = 1e-4;
    int max_iters = 100;
    double mu_init = 1e-8;
    aligator::SolverProxDDPTpl<double> solver(tol, mu_init, max_iters, aligator::VerboseLevel::VERBOSE);
    std::vector<VectorXd> x_guess, u_guess;
    x_guess.assign(nsteps + 1, x0);
    u_guess.assign(nsteps, u_nom);
    solver.rollout_type_ = aligator::RolloutType::LINEAR;
    solver.force_initial_condition_ = true;
    solver.linear_solver_choice = aligator::LQSolverChoice::PARALLEL;
    solver.setNumThreads(8);
    solver.setup(*problem);

    /************************first solve**********************/
    solver.run(*problem, x_guess, u_guess);
    solver.max_iters = yaml_loader.max_iter;

    /************************second solve**********************/
    x0.head(nq) << 0, 0, 0.29538, 0, 0, 0, 1, 7.58493e-07, -0.8, 1.6, -1.03355e-06, -0.8, 1.6, 6.33446e-07, -0.8, 1.6, -9.08758e-07, -0.8, 1.6;
    x0.tail(nv) << 0, 0, 0, 0, 0, 0, 0.000758493, -0.000244311, 0.00318397, -0.00103355, -0.000201817, 0.00241434, 0.000633446, -0.000217522, 0.00290923, -0.000908758, -0.000213618, 0.00214024;
    std::vector<VectorXd> q_nom = readVectorsFromCsv("/home/zishang/cpp_workspace/aligator_cimpc/log/q_nom.csv");
    std::vector<VectorXd> v_nom(q_nom.size(), VectorXd::Zero(nv));
    for (size_t i = 0; i < q_nom.size(); ++i)
    {
        x_ref[i].head(nq) = q_nom[i];
        x_ref[i].tail(nv) = v_nom[i];
    }

    problem->setInitState(x0);
    updateStateReferences(problem, x_ref);

    // 更新warm start
    x_guess = solver.results_.xs;
    u_guess = solver.results_.us;
    x_guess.erase(x_guess.begin());
    x_guess[0] = x0;
    x_guess.push_back(x_guess.back());
    u_guess.erase(u_guess.begin());
    u_guess.push_back(u_guess.back());

    solver.run(*problem, x_guess, u_guess);

    // for (int i = 0; i < x_ref.size(); ++i)
    // {
    //     std::cout << "Step " << i << ": " << x_ref[i].transpose() << std::endl;
    // }

    // for (int i = 0; i < solver.results_.xs.size(); ++i)
    // {
    //     std::cout << "Step " << i << ": " << solver.results_.xs[i].transpose() << std::endl;
    // }

    /// idto 的结果
    std::vector<VectorXd> q = readVectorsFromCsv("/home/zishang/cpp_workspace/aligator_cimpc/log/q.csv");
    std::vector<VectorXd> v = readVectorsFromCsv("/home/zishang/cpp_workspace/aligator_cimpc/log/v.csv");
    std::vector<VectorXd> tau = readVectorsFromCsv("/home/zishang/cpp_workspace/aligator_cimpc/log/tau.csv");

    /// itdo 的权重
    MatrixXd Qq = MatrixXd::Zero(model.nq, model.nq);
    Qq.diagonal() << yaml_loader.w_pos_body, 1,
        yaml_loader.w_pos_leg, yaml_loader.w_pos_leg, yaml_loader.w_pos_leg, yaml_loader.w_pos_leg;
    MatrixXd Qv = MatrixXd::Zero(model.nv, model.nv);
    Qv.diagonal() << yaml_loader.w_vel_body,
        yaml_loader.w_vel_leg, yaml_loader.w_vel_leg, yaml_loader.w_vel_leg, yaml_loader.w_vel_leg;
    MatrixXd R = MatrixXd::Zero(model.nv, model.nv);
    R.diagonal() << 0, 0, 0, 0, 0, 0, // 把浮动基力矩的权重去了，方便对比
        yaml_loader.w_u_leg, yaml_loader.w_u_leg, yaml_loader.w_u_leg, yaml_loader.w_u_leg;
    MatrixXd Qf_q = MatrixXd::Zero(model.nq, model.nq);
    Qf_q.diagonal() << yaml_loader.w_pos_body_term, 1,
        yaml_loader.w_pos_leg_term, yaml_loader.w_pos_leg_term, yaml_loader.w_pos_leg_term, yaml_loader.w_pos_leg_term;
    MatrixXd Qf_v = MatrixXd::Zero(model.nv, model.nv);
    Qf_v.diagonal() << yaml_loader.w_vel_body_term,
        yaml_loader.w_vel_leg_term, yaml_loader.w_vel_leg_term, yaml_loader.w_vel_leg_term, yaml_loader.w_vel_leg_term;

    //// idto 的cost
    double cost1 = CalcCost(model, q, v, tau, q_nom, v_nom,
                            Qq, Qv, R, Qf_q, Qf_v);

    std::cout << "用 idto 的 cost 计算方式计算 idto 的结果: " << cost1 << std::endl;

    //// 用 idto 的cost计算方式计算aligator的结果
    std::vector<VectorXd> q2, v2, tau2;
    for (int i = 0; i < solver.results_.xs.size(); ++i)
    {
        q2.push_back(solver.results_.xs[i].head(model.nq));
        v2.push_back(solver.results_.xs[i].tail(model.nv));
    }
    for (int i = 0; i < solver.results_.us.size(); ++i)
    {
        VectorXd tau(model.nv);
        tau.head(6) = VectorXd::Zero(6); // 前6个关节的力矩为0
        tau.tail(model.nv - 6) = solver.results_.us[i].tail(model.nv - 6);
        tau2.push_back(tau);
    }

    double cost2 = CalcCost(model, q2, v2, tau2, q_nom, v_nom,
                            Qq, Qv, R, Qf_q, Qf_v);
    std::cout << "用 idto 的 cost 计算方式计算 aligator 的结果: " << cost2 << std::endl;

    //// 用 aligator 的cost计算方式计算 aligator 的结果
    aligator::TrajOptDataTpl<double> problem_data(*problem);
    problem->evaluate(solver.results_.xs, solver.results_.us, problem_data, 1);
    std::cout << "用 aligator 的 cost 计算方式计算 aligator 的结果: " << problem_data.cost_ << std::endl;

    // auto cost_data = problem->stages_[0]->cost_->createData();
    // problem->stages_[0]->cost_->evaluate(solver.results_.xs[0], solver.results_.us[0], *cost_data);
    // std::cout << "用 aligator 的cost计算方式计算 aligator 的结果: " << cost_data->value_ << std::endl;
    // // 访问 sub_cost_data
    // auto sum_cost_data = static_cast<aligator::CostStackDataTpl<double> *>(cost_data.get());
    // for (const auto &[key, sub_data] : sum_cost_data->sub_cost_data)
    // {
    //     std::cout << "Key: " << std::get<std::string>(key) << std::endl;
    //     std::cout << "Sub-cost value: " << sub_data->value_ << std::endl;
    // }

    return 0;
}
