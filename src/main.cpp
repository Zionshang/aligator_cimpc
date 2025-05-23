#include <aligator/core/traj-opt-problem.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <aligator/modelling/costs/sum-of-costs.hpp>
#include <aligator/modelling/costs/quad-state-cost.hpp>
#include <aligator/solvers/proxddp/solver-proxddp.hpp>
#include <aligator/modelling/autodiff/finite-difference.hpp>
#include <aligator/modelling/autodiff/cost-finite-difference.hpp>
#include <aligator/modelling/state-error.hpp>
#include <proxsuite-nlp/modelling/constraints/box-constraint.hpp>
#include <aligator/core/stage-data.hpp>
#include <aligator/modelling/dynamics/integrator-semi-euler.hpp>
#include <aligator/modelling/dynamics/integrator-euler.hpp>
#include <aligator/modelling/dynamics/integrator-midpoint.hpp>
#include <aligator/modelling/dynamics/integrator-rk2.hpp>

#include <fstream>

#include <pinocchio/algorithm/rnea.hpp>
#include "contact_fwd_dynamics.hpp"
#include "foot_slip_clearance_cost.hpp"
#include "logger.hpp"
#include "yaml_loader.hpp"
#include "webots_interface.hpp"
#include "contact_assessment.hpp"
#include "relaxed_wbc.hpp"
#include "interpolator.hpp"
#include "contact_inv_dynamics_residual.hpp"
#include "kinematics_ode.hpp"

#define PD
// #define WBC
// #define FWD_DYNAMICS
#define INV_DYNAMICS

using aligator::context::TrajOptProblem;
using StageModel = aligator::StageModelTpl<double>;
using CostStack = aligator::CostStackTpl<double>;
using QuadraticStateCost = aligator::QuadraticStateCostTpl<double>;
using QuadraticControlCost = aligator::QuadraticControlCostTpl<double>;
using CostFiniteDifference = aligator::autodiff::CostFiniteDifferenceHelper<double>;
using ControlErrorResidual = aligator::ControlErrorResidualTpl<double>;
using BoxConstraint = proxsuite::nlp::BoxConstraintTpl<double>;
using ExplicitIntegratorData = aligator::dynamics::ExplicitIntegratorDataTpl<double>;
using IntegratorSemiImplEuler = aligator::dynamics::IntegratorSemiImplEulerTpl<double>;
using IntegratorEuler = aligator::dynamics::IntegratorEulerTpl<double>;
using IntegratorMidpoint = aligator::dynamics::IntegratorMidpointTpl<double>;
using IntegratorRK2 = aligator::dynamics::IntegratorRK2Tpl<double>;
using EqualityConstraint = proxsuite::nlp::EqualityConstraintTpl<double>;

std::string yaml_filename = "/home/zishang/cpp_workspace/aligator_cimpc/config/parameters.yaml";
YamlLoader yaml_loader(yaml_filename);

VectorXd calcNominalTorque(const Model &model, const VectorXd &q_nom)
{
    int nq = model.nq;
    int nv = model.nv;
    Data data(model);
    pinocchio::rnea(model, data, q_nom, VectorXd::Zero(nv), VectorXd::Zero(nv));
    // return data.tau.tail(nv - 6);
    return VectorXd::Zero(nv - 6);
}

void computeFutureStates(const double &dx,
                         const VectorXd &x0,
                         std::vector<VectorXd> &x_ref)
{
    for (int i = 0; i < x_ref.size(); ++i)
    {
        x_ref[i](0) = x0(0) + dx;
    }
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
    // IntegratorEuler discrete_dyn = IntegratorEuler(dynamics, timestep);
    // IntegratorMidpoint discrete_dyn = IntegratorMidpoint(dynamics, timestep);
    // IntegratorRK2 discrete_dyn = IntegratorRK2(dynamics, timestep);

    std::vector<xyz::polymorphic<StageModel>> stage_models;
    FootSlipClearanceCost fscc(space, nu, yaml_loader.w_foot_slip_clearance, -30.0);
    CostFiniteDifference fscc_fini_diff(fscc, 1e-6);
    ControlErrorResidual control_error(space.ndx(), nu);
    VectorXd u_max = space.getModel().effortLimit.tail(nu);

    for (size_t i = 0; i < nsteps; i++)
    {
        auto rcost = CostStack(space, nu);
        rcost.addCost("state_cost", QuadraticStateCost(space, nu, x_ref[i], w_x * timestep));
        rcost.addCost("control_cost", QuadraticControlCost(space, u_ref[i], w_u * timestep));
        rcost.addCost("foot_slip_clearance_cost", fscc_fini_diff);

        StageModel sm = StageModel(rcost, discrete_dyn);

        sm.addConstraint(control_error, BoxConstraint(-u_max, u_max));
        stage_models.push_back(std::move(sm));
    }

    auto term_cost = CostStack(space, nu);
    term_cost.addCost("term_state_cost", QuadraticStateCost(space, nu, x_ref.back(), w_x_term));

    return std::make_shared<TrajOptProblem>(x0, stage_models, term_cost);
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

    for (size_t i = 0; i < nsteps; i++)
    {
        auto rcost = CostStack(space, nu);
        rcost.addCost("state_cost", QuadraticStateCost(space, nu, x_ref[i], w_x * timestep));
        rcost.addCost("control_cost", QuadraticControlCost(space, u_ref[i], w_u * timestep));

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

std::vector<VectorXd> getAccelerationResult(const aligator::SolverProxDDPTpl<double> &solver, int nv)
{
    std::vector<VectorXd> a;
    for (size_t i = 0; i < solver.workspace_.problem_data.stage_data.size(); i++)
    {
        auto int_data = std::dynamic_pointer_cast<ExplicitIntegratorData>(
            solver.workspace_.problem_data.stage_data[i]->dynamics_data);
        a.push_back(int_data->continuous_data->xdot_.tail(nv));
    }
    return a;
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

    /************************reference state**********************/
    double vx = 0;
    double dx = 0;

    std::vector<VectorXd> x_ref(nsteps, x0);
    // computeFutureStates(model, vx, x0, timestep, x_ref);
    computeFutureStates(dx, x0, x_ref);

    /************************create problem**********************/

#ifdef FWD_DYNAMICS
    // VectorXd u_nom = calcNominalTorque(model, x0.head(nq));
    // std::vector<VectorXd> u_ref(nsteps, u_nom);
    ContactFwdDynamics dynamics(space, actuation, contact_params);
    auto problem = createTrajOptProblem(dynamics, nsteps, timestep, x_ref, u_ref, x0);
#endif

#ifdef INV_DYNAMICS
    VectorXd u_nom = Eigen::VectorXd::Zero(nv + nu);
    std::vector<VectorXd> u_ref(nsteps, u_nom);
    KinematicsODE kinematics(space);
    auto problem = createTrajOptProblem(kinematics, model, nsteps, timestep, x_ref, u_ref, x0);
#endif

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

    x_guess = solver.results_.xs;
    u_guess = solver.results_.us;
    solver.max_iters = yaml_loader.max_iter;

    /************************webots仿真**********************/
    WebotsInterface webots_interface;
    ContactAssessment contact_assessment(model, contact_params);
    RelaxedWbcSettings Rwbc_settings;
    Interpolator interpolator(model);
    Rwbc_settings.contact_ids = std::vector<pinocchio::FrameIndex>{11, 19, 27, 35};
    Rwbc_settings.mu = 1;
    Rwbc_settings.force_size = 3;
    Rwbc_settings.w_acc = 100;
    Rwbc_settings.w_force = 1;
    Rwbc_settings.verbose = false;
    RelaxedWbc relaxed_wbc(Rwbc_settings, model);

    int mpc_cycle = 10;
    int itr = 0;
    const double dt = webots_interface.timestep();
    std::vector<VectorXd> x_log, u_log, qd_log, q_log;
    std::vector<double> cost_log;
    VectorXd contact_forces = VectorXd::Zero(12);
    std::vector<VectorXd> contact_forces_log;
    std::cout << std::fixed << std::setprecision(3);
    dx = 0;
    VectorXd kp(nu), kd(nu);
    kp << yaml_loader.kp_leg, yaml_loader.kp_leg, yaml_loader.kp_leg, yaml_loader.kp_leg;
    kd << yaml_loader.kd_leg, yaml_loader.kd_leg, yaml_loader.kd_leg, yaml_loader.kd_leg;
    while (webots_interface.isRunning())
    {
        // 获取当前状态
        webots_interface.recvState(x0);
        // std::cout << "x0: " << x0.transpose() << std::endl;

        // 更新期望状态
        computeFutureStates(dx, x0, x_ref);
        updateStateReferences(problem, x_ref);

        if (int(itr % mpc_cycle) == 0)
        {
            auto start_time = std::chrono::high_resolution_clock::now();

            // 更新当前位置
            problem->setInitState(x0);

            // 求解
            solver.run(*problem, x_guess, u_guess);

            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> solve_time = end_time - start_time;
            std::cout << "MPC solve time: " << solve_time.count() << " ms" << std::endl;

            itr = 0;
        }

#ifdef WBC
        // 插值
        double delay = itr * dt;
        VectorXd x_interp(nq + nv), a_interp(nv);
        std::vector<VectorXd> a_result = getAccelerationResult(solver, nv);
        interpolator.interpolateState(delay, timestep, solver.results_.xs, x_interp);
        interpolator.interpolateLinear(delay, timestep, a_result, a_interp);

        // 评估接触信息
        contact_assessment.update(x_interp);

        // wbc
        VectorXd q = x0.head(nq), v = x0.tail(nv);
        std::vector<bool> contact_state = contact_assessment.contact_state();
        for (size_t i = 0; i < 4; i++)
            contact_forces.segment(i * 3, 3) = contact_assessment.contact_forces()[i];

        std::cout << "contact state: ";
        for (size_t i = 0; i < 4; i++)
            std::cout << contact_assessment.contact_state()[i] << "  ";
        std::cout << std::endl;

        relaxed_wbc.solveQP(contact_state, q, v, a_interp, VectorXd::Zero(12), contact_forces);

        std::cout << "contact forces: " << contact_forces.transpose() << std::endl;
        std::cout << "solved_forces_: " << relaxed_wbc.solved_forces_.transpose() << std::endl;
        std::cout << "a_interp: " << a_interp.tail(nu).transpose() << std::endl;
        std::cout << "solved_acc_: " << relaxed_wbc.solved_acc_.tail(nu).transpose() << std::endl;
        std::cout << "solved_torque_: " << relaxed_wbc.solved_torque_.transpose() << std::endl;

        // 发送力矩
        webots_interface.sendCmd(relaxed_wbc.solved_torque_);
#endif

#ifdef PD
        double delay = itr * dt;
        std::cout << "delay: " << delay << std::endl;
        VectorXd x_interp(nq + nv), u_interp(nu);
        interpolator.interpolateState(delay, timestep, solver.results_.xs, x_interp);
        interpolator.interpolateLinear(delay, timestep, solver.results_.us, u_interp);

#ifdef FWD_DYNAMICS
        std::cout << "=== result state ===\n";
        for (size_t i = 0; i < solver.results_.xs.size(); ++i)
        {
            std::cout << "xs[" << i << "]: " << solver.results_.xs[i].head(nq).transpose().format(Eigen::IOFormat(3, 0, ", ", ", ", "", "", "[", "]")) << std::endl;
        }
        std::cout << "=== result tau ===\n";
        for (size_t i = 0; i < solver.results_.us.size(); ++i)
        {
            std::cout << "us[" << i << "]: " << solver.results_.us[i].transpose().format(Eigen::IOFormat(3, 0, ", ", ", ", "", "", "[", "]")) << std::endl;
        }
        std::cout << "=== End of us vector ===\n";
        // 评估接触信息
        // contact_assessment.update(x_interp);

        VectorXd qd = x_interp.segment(7, nu), vd = x_interp.segment(nq + 6, nu);
        VectorXd q = x0.segment(7, nu), v = x0.segment(nq + 6, nu);
        VectorXd tau = tau_interp + kp.cwiseProduct(qd - q) + kd.cwiseProduct(vd - v);
#endif

#ifdef INV_DYNAMICS
        std::cout << "=== result state ===\n";
        for (size_t i = 0; i < solver.results_.xs.size() / 2; ++i)
        {
            std::cout << "xs[" << i << "]: " << solver.results_.xs[i].segment(7, 3).transpose().format(Eigen::IOFormat(3, 0, ", ", ", ", "", "", "[", "]"))
                      << solver.results_.xs[i].segment(nq + 6, 3).transpose().format(Eigen::IOFormat(3, 0, ", ", ", ", "", "", "[", "]")) << std::endl;
        }
        std::cout << "=== result tau ===\n";
        for (size_t i = 0; i < solver.results_.us.size() / 2; ++i)
        {
            std::cout << "us[" << i << "]: " << solver.results_.us[i].tail(nu).transpose().format(Eigen::IOFormat(3, 0, ", ", ", ", "", "", "[", "]")) << std::endl;
        }
        std::cout << "=== End of us vector ===\n";

        VectorXd qd = x_interp.segment(7, nu), vd = x_interp.segment(nq + 6, nu);
        VectorXd q = x0.segment(7, nu), v = x0.segment(nq + 6, nu);
        VectorXd tau = u_interp.tail(nu) + kp.cwiseProduct(qd - q) + kd.cwiseProduct(vd - v);
#endif
        std::cout << "qd: " << qd.transpose() << std::endl;
        std::cout << "q: " << q.transpose() << std::endl;
        std::cout << "vd: " << vd.transpose() << std::endl;
        std::cout << "v: " << v.transpose() << std::endl;
        std::cout << "tau_interp: " << u_interp.tail(nu).transpose() << std::endl;
        std::cout << "tau: " << tau.transpose() << std::endl;
        webots_interface.sendCmd(tau);
#endif

        // 评估接触信息
        contact_assessment.update(x_interp.head(nq), x_interp.tail(nv));

        // 更新warm start
        x_guess = solver.results_.xs;
        u_guess = solver.results_.us;
        x_guess.erase(x_guess.begin());
        x_guess[0] = x0;
        x_guess.push_back(x_guess.back());
        u_guess.erase(u_guess.begin());
        u_guess.push_back(u_guess.back());

        itr++;

        // 记录数据
        // x_log.push_back(x0);
        // u_log.push_back(solver.results_.us[0]);
        std::cout << "contact forces: ";
        for (size_t i = 0; i < 4; i++)
            std::cout << contact_assessment.contact_forces()[i].transpose() << "  ";
        std::cout << std::endl;
        // std::cout << "contact state: ";
        // for (size_t i = 0; i < 4; i++)
        //     std::cout << contact_assessment.contact_state()[i] << "  ";
        // std::cout << std::endl;
        // contact_forces_log.push_back(contact_forces);
        // cost_log.push_back(solver.results_.traj_cost_);
        qd_log.push_back(qd);
        q_log.push_back(q);
    }
    // saveVectorsToCsv("idea_sim_x.csv", x_log);
    // saveVectorsToCsv("idea_sim_u.csv", u_log);
    // saveVectorsToCsv("idea_sim_contact_forces.csv", contact_forces_log);
    // saveVectorsToCsv("idea_sim_cost.csv", cost_log);
    saveVectorsToCsv("webots_sim_qd.csv", qd_log);
    saveVectorsToCsv("webots_sim_q.csv", q_log);

    return 0;
}
