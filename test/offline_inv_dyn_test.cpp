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
#include <aligator/modelling/costs/quad-residual-cost.hpp>
#include <aligator/modelling/function-xpr-slice.hpp>

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
#include "symmetric_control_residual.hpp"

using aligator::context::TrajOptProblem;
using StageModel = aligator::StageModelTpl<double>;
using IntegratorSemiImplEuler = aligator::dynamics::IntegratorSemiImplEulerTpl<double>;
using IntegratorEuler = aligator::dynamics::IntegratorEulerTpl<double>;
using CostStack = aligator::CostStackTpl<double>;
using QuadraticStateCost = aligator::QuadraticStateCostTpl<double>;
using QuadraticControlCost = aligator::QuadraticControlCostTpl<double>;
using CostFiniteDifference = aligator::autodiff::CostFiniteDifferenceHelper<double>;
using ControlErrorResidual = aligator::ControlErrorResidualTpl<double>;
using BoxConstraint = proxsuite::nlp::BoxConstraintTpl<double>;
using QuadraticResidualCost = aligator::QuadraticResidualCostTpl<double>;
using FunctionSliceXpr = aligator::FunctionSliceXprTpl<double>;
using EqualityConstraint = proxsuite::nlp::EqualityConstraintTpl<double>;

std::string yaml_filename = "/home/zishang/cpp_workspace/aligator_cimpc/config/parameters_walk.yaml";
YamlLoader yaml_loader(yaml_filename);

VectorXd calcNominalTorque(const Model &model, const VectorXd &q_nom)
{
    int nq = model.nq;
    int nv = model.nv;
    Data data(model);
    pinocchio::rnea(model, data, q_nom, VectorXd::Zero(nv), VectorXd::Zero(nv));
    // return data.tau.tail(nv - 6);
    // return VectorXd::Zero(nv - 6);
    return VectorXd::Zero(nv + nv - 6);
}

void computeFutureStates(const double &dx,
                         const VectorXd &x0,
                         std::vector<VectorXd> &x_ref)
{
    // x_ref.back() = x0;
    // x_ref.back()[0] += dx;

    // for (int i = 0; i < x_ref.size(); ++i)
    // {
    //     double alpha = static_cast<double>(i) / (x_ref.size() - 1);
    //     x_ref[i] = (1.0 - alpha) * x0 + alpha * x_ref.back();
    // }

    for (int i = 0; i < x_ref.size(); ++i)
    {
        x_ref[i](0) = x0(0) + dx;
    }
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
    ContactInvDynamicsResidual2 contact_inv_dynamics_residual(ndx, model, actuation, timestep, yaml_loader.real_contact_params);
    FootSlipClearanceCost fscc(space, nu, yaml_loader.w_foot_slip_clearance, -30.0);
    CostFiniteDifference fscc_fini_diff(fscc, 1e-6);
    SymmetricControlResidual symmetric_control_residual(ndx, nu, num_actuated);

    for (size_t i = 0; i < nsteps; i++)
    {
        auto rcost = CostStack(space, nu);
        rcost.addCost("state_cost", QuadraticStateCost(space, nu, x_ref[i], w_x));
        rcost.addCost("control_cost", QuadraticControlCost(space, u_ref[i], w_u));
        rcost.addCost("foot_slip_clearance_cost", fscc_fini_diff);
        rcost.addCost("symmetric_control_cost",
                      QuadraticResidualCost(space, symmetric_control_residual,
                                            yaml_loader.w_symmetric_control * MatrixXd::Identity(4, 4)));

        StageModel sm = StageModel(rcost, discrete_dyn);

        sm.addConstraint(contact_inv_dynamics_residual, EqualityConstraint());
        stage_models.push_back(std::move(sm));
    }

    auto term_cost = CostStack(space, nu);
    term_cost.addCost("term_state_cost", QuadraticStateCost(space, nu, x_ref.back(), w_x_term));

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

    std::string urdf_filename = EXAMPLE_ROBOT_DATA_MODEL_DIR "/go2_description/urdf/go2.urdf";

    Model model;
    pinocchio::urdf::buildModel(urdf_filename, pinocchio::JointModelFreeFlyer(), model);
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
    x0.head(nq) << 0.0, 0.0, 0.34,
        0.0, 0.0, 0.0, 1.0,
        0.0, 0.785, -1.44,
        0.0, 0.785, -1.44,
        0.0, 0.785, -1.44,
        0.0, 0.785, -1.44;

    /************************reference state**********************/
    double dx = 0;

    std::vector<VectorXd> x_ref(nsteps, x0);
    computeFutureStates(dx, x0, x_ref);
    VectorXd u_nom = calcNominalTorque(model, x0.head(nq));
    std::vector<VectorXd> u_ref(nsteps, u_nom);

    /************************print reference state**********************/
    // std::cout << "Printing x_ref values:" << std::endl;
    // for (size_t i = 0; i < x_ref.size(); ++i)
    // {
    //     std::cout << "x_ref[" << i << "] = "
    //               << x_ref[i].transpose() << std::endl;
    // }
    /************************create problem**********************/
    auto problem = createTrajOptProblem(kinematics, model, nsteps, timestep, x_ref, u_ref, x0);
    double tol = 1e-4;
    int max_iters = 100;
    double mu_init = 1e-8;
    aligator::SolverProxDDPTpl<double> solver(tol, mu_init, max_iters, aligator::VerboseLevel::QUIET);
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
    // saveVectorsToCsv("offline_test.csv", solver.results_.xs);

    x_guess = solver.results_.xs;
    u_guess = solver.results_.us;
    solver.max_iters = yaml_loader.max_iter;

    /************************理想迭代**********************/
    std::vector<VectorXd> x_log, u_log;
    std::vector<double> cost_log;
    VectorXd contact_forces = VectorXd::Zero(12);
    std::vector<VectorXd> contact_forces_log;
    std::cout << std::fixed << std::setprecision(2);
    dx = 0.5;
    std::vector<double> solve_times; // 用于存储每次求解时间
    ContactAssessment contact_assessment(model, contact_params);

    for (size_t i = 0; i < 200; i++)
    {
        // 更新期望状态
        // computeFutureStates(model, vx, x0, timestep, x_ref);
        computeFutureStates(dx, x0, x_ref);
        // std::cout << "x_ref[0] = " << x_ref[0].transpose() << std::endl;
        // std::cout << "x_ref[end] = " << x_ref.back().transpose() << std::endl;
        updateStateReferences(problem, x_ref);

        // 更新当前位置
        problem->setInitState(x0);

        // 求解
        auto start_time = std::chrono::high_resolution_clock::now();
        solver.run(*problem, x_guess, u_guess);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        solve_times.push_back(elapsed.count()); // 记录求解时间

        // 评估当前接触力
        contact_assessment.update(x0.head(nq), x0.tail(nv));

        // 更新位置
        x0 = solver.results_.xs[1];

        // 更新warm start
        x_guess = solver.results_.xs;
        u_guess = solver.results_.us;
        x_guess.erase(x_guess.begin());
        x_guess[0] = x0;
        x_guess.push_back(x_guess.back());
        u_guess.erase(u_guess.begin());
        u_guess.push_back(u_guess.back());

        // 记录数据
        x_log.push_back(x0);
        u_log.push_back(solver.results_.us[0]);
        for (size_t i = 0; i < 4; i++)
        {
            const auto &force = contact_assessment.contact_forces()[i];
            std::cout << force.transpose() << "  ";
            contact_forces.segment(i * 3, 3) = force;
        }
        std::cout << std::endl;
        contact_forces_log.push_back(contact_forces);
        cost_log.push_back(solver.results_.traj_cost_);
    }

    // 计算并输出平均求解时间
    double total_time = std::accumulate(solve_times.begin(), solve_times.end(), 0.0);
    double average_time_ms = (total_time / solve_times.size()) * 1000.0; // Convert to milliseconds
    std::cout << "Average solve time: " << std::fixed << std::setprecision(3) << average_time_ms << " ms" << std::endl;

    saveVectorsToCsv("offline_inv_sim_x.csv", x_log);
    saveVectorsToCsv("offline_inv_sim_u.csv", u_log);
    saveVectorsToCsv("offine_inv_sim_contact_forces.csv", contact_forces_log);
    saveVectorsToCsv("offline_inv_sim_cost.csv", cost_log);

    return 0;
}
