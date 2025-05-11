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
#include <aligator/core/stage-data.hpp>

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
using ExplicitIntegratorData = aligator::dynamics::ExplicitIntegratorDataTpl<double>;

std::string yaml_filename = "/home/zishang/cpp_workspace/aligator_cimpc/config/parameters.yaml";
YamlLoader yaml_loader(yaml_filename);

VectorXd calcNominalTorque(const Model &model, const VectorXd &q_nom)
{
    int nq = model.nq;
    int nv = model.nv;
    Data data(model);
    pinocchio::rnea(model, data, q_nom, VectorXd::Zero(nv), VectorXd::Zero(nv));
    return data.tau.tail(nv - 6);
    // return VectorXd::Zero(nv - 6);
}

void computeFutureStates(const Model &model,
                         const double &vx,
                         const VectorXd &x0,
                         double dt,
                         std::vector<VectorXd> &x_ref)
{
    Data data(model);

    int nq = model.nq;
    int nv = model.nv;

    VectorXd q = x0.head(nq);
    VectorXd v = x0.tail(nv);

    x_ref[0] = x0;

    for (int i = 1; i < x_ref.size(); ++i)
    {
        x_ref[i](0) = x_ref[i - 1](0) + vx * dt;
        x_ref[i](nq) = vx;
    }
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

    std::vector<xyz::polymorphic<StageModel>> stage_models;
    FootSlipClearanceCost fscc(space, nu, yaml_loader.w_foot_slip_clearance, -30.0);
    CostFiniteDifference fscc_fini_diff(fscc, 1e-6);
    ControlErrorResidual control_error(space.ndx(), nu);
    VectorXd u_max = space.getModel().effortLimit.tail(nu);

    for (size_t i = 0; i < nsteps; i++)
    {
        auto rcost = CostStack(space, nu);
        rcost.addCost("state_cost", QuadraticStateCost(space, nu, x_ref[i], w_x));
        rcost.addCost("control_cost", QuadraticControlCost(space, u_ref[i], w_u));
        rcost.addCost("foot_slip_clearance_cost", fscc_fini_diff);

        StageModel sm = StageModel(rcost, discrete_dyn);

        sm.addConstraint(control_error, BoxConstraint(-u_max, u_max));
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

    std::string urdf_filename = "/home/zishang/cpp_workspace/aligator_cimpc/robot/galileo_v1d6_description/urdf/galileo_v1d6.urdf";

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

    /************************initial state**********************/
    VectorXd x0 = VectorXd::Zero(nq + nv);
    x0.head(nq) << 0.0, 0.0, 0.4,
        0.0, 0.0, 0.0, 1.0,
        0.0, 0.72, -1.44,
        0.0, 0.72, -1.44,
        0.0, 0.72, -1.44,
        0.0, 0.72, -1.44;

    /************************reference state**********************/
    double vx = 0;
    double dx = 0;

    std::vector<VectorXd> x_ref(nsteps, x0);
    // computeFutureStates(model, vx, x0, timestep, x_ref);
    computeFutureStates(dx, x0, x_ref);
    VectorXd u_nom = calcNominalTorque(model, x0.head(nq));
    std::vector<VectorXd> u_ref(nsteps, u_nom);

    /************************create problem**********************/
    auto problem = createTrajOptProblem(dynamics, nsteps, timestep, x_ref, u_ref, x0);
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

    x_guess = solver.results_.xs;
    u_guess = solver.results_.us;
    solver.max_iters = yaml_loader.max_iter;

    /************************webots仿真**********************/
    WebotsInterface webots_interface;
    ContactAssessment contact_assessment(dynamics);
    RelaxedWbcSettings Rwbc_settings;
    Interpolator interpolator(model);
    Rwbc_settings.contact_ids = std::vector<pinocchio::FrameIndex>{11, 19, 27, 35};
    Rwbc_settings.mu = 1;
    Rwbc_settings.force_size = 3;
    Rwbc_settings.w_acc = 1;
    Rwbc_settings.w_force = 100;
    Rwbc_settings.verbose = false;
    RelaxedWbc relaxed_wbc(Rwbc_settings, model);

    int mpc_cycle = 10;
    int itr = 0;
    const double dt = webots_interface.timestep();
    std::vector<VectorXd> x_log, u_log;
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
            // 更新当前位置
            problem->setInitState(x0);

            // 求解
            solver.run(*problem, x_guess, u_guess);
            itr = 0;
        }

        // 评估接触信息
        contact_assessment.update(solver.results_.xs[0]);

        // // wbc
        // VectorXd q = x0.head(nq);
        // VectorXd v = x0.tail(nv);
        // auto int_data = std::dynamic_pointer_cast<ExplicitIntegratorData>(
        //     solver.workspace_.problem_data.stage_data[0]->dynamics_data);
        // VectorXd a = int_data->continuous_data->xdot_;
        // std::vector<bool> contact_state = contact_assessment.contact_state();
        // for (size_t i = 0; i < 4; i++)
        //     contact_forces.segment(i * 3, 3) = contact_assessment.contact_forces()[i];
        // relaxed_wbc.solveQP(contact_state, q, v, a, VectorXd::Zero(12), contact_forces);

        // // 发送力矩
        // webots_interface.sendCmd(relaxed_wbc.solved_torque_);
        // std::cout << "xs: " << solver.results_.xs[1].transpose() << std::endl;
        // std::cout << "u: " << relaxed_wbc.solved_torque_.transpose() << std::endl;

        double delay = itr * dt;
        std::cout << "delay: " << delay << std::endl;
        VectorXd x_interp(nq + nv), tau_interp(nu);
        interpolator.interpolateState(delay, timestep, solver.results_.xs, x_interp);
        interpolator.interpolateLinear(delay, timestep, solver.results_.us, tau_interp);
        VectorXd qd = x_interp.segment(6, nu), vd = x_interp.segment(nq + 6, nu);
        VectorXd q = x0.segment(6, nu), v = x0.segment(nq + 6, nu);
        VectorXd tau = tau_interp + kp.cwiseProduct(qd - q) + kd.cwiseProduct(vd - v);
        std::cout << "qd: " << qd.transpose() << std::endl;
        std::cout << "q: " << q.transpose() << std::endl;
        std::cout << "vd: " << vd.transpose() << std::endl;
        std::cout << "v: " << v.transpose() << std::endl;
        std::cout << "tau_interp: " << tau_interp.transpose() << std::endl;
        std::cout << "tau: " << tau.transpose() << std::endl;
        webots_interface.sendCmd(tau);

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
        x_log.push_back(x0);
        u_log.push_back(solver.results_.us[0]);
        std::cout << "contact forces: ";
        for (size_t i = 0; i < 4; i++)
            std::cout << contact_assessment.contact_forces()[i].transpose() << "  ";
        std::cout << std::endl;
        contact_forces_log.push_back(contact_forces);
        cost_log.push_back(solver.results_.traj_cost_);
    }
    // saveVectorsToCsv("idea_sim_x.csv", x_log);
    // saveVectorsToCsv("idea_sim_u.csv", u_log);
    saveVectorsToCsv("idea_sim_contact_forces.csv", contact_forces_log);
    // saveVectorsToCsv("idea_sim_cost.csv", cost_log);

    return 0;
}
