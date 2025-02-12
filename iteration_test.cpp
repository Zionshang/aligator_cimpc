#include <aligator/core/traj-opt-problem.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>

#include <aligator/modelling/costs/sum-of-costs.hpp>
#include <aligator/modelling/costs/quad-state-cost.hpp>
#include <aligator/modelling/dynamics/integrator-semi-euler.hpp>
#include <aligator/solvers/proxddp/solver-proxddp.hpp>
#include <aligator/modelling/autodiff/finite-difference.hpp>
#include <aligator/modelling/dynamics/integrator-euler.hpp>
#include <fstream>

#include "CompliantContactFwdDynamics.hpp"
#include "logger.hpp"

namespace pin = pinocchio;
using aligator::context::TrajOptProblem;
using StageModel = aligator::StageModelTpl<double>;
using IntegratorSemiImplEuler = aligator::dynamics::IntegratorSemiImplEulerTpl<double>;
using IntegratorEuler = aligator::dynamics::IntegratorEulerTpl<double>;
using CostStack = aligator::CostStackTpl<double>;
using QuadraticStateCost = aligator::QuadraticStateCostTpl<double>;
using QuadraticControlCost = aligator::QuadraticControlCostTpl<double>;
using DynamicsFiniteDifference = aligator::autodiff::DynamicsFiniteDifferenceHelper<double>;

std::shared_ptr<TrajOptProblem> createTrajOptProblem(const CompliantContactFwdDynamics &dynamics,
                                                     int nsteps, double timestep,
                                                     const std::vector<VectorXd> &x_ref,
                                                     const VectorXd &x0, const VectorXd &u0)
{
    const auto space = dynamics.space();
    const int nu = dynamics.nu(); // Number of control inputs
    const int ndx = space.ndx();  // Number of state variables

    // Define stage state weights
    VectorXd w_pos_diag = VectorXd::Zero(space.getModel().nv);
    VectorXd w_vel_diag = VectorXd::Zero(space.getModel().nv);
    w_pos_diag << 10, 10, 10, // linear part
        1, 1, 1,              // angular part
        0, 0, 0,              // leg part
        0, 0, 0,
        0, 0, 0,
        0, 0, 0;
    w_vel_diag << 1, 1, 1, // linear part
        1, 1, 1,           // angular part
        0.1, 0.1, 0.1,     // leg part
        0.1, 0.1, 0.1,
        0.1, 0.1, 0.1,
        0.1, 0.1, 0.1;
    VectorXd w_x_diag = VectorXd::Zero(ndx);
    w_x_diag << w_pos_diag, w_vel_diag;
    MatrixXd w_x = w_x_diag.asDiagonal();

    // Define terminal state weights
    w_pos_diag << 10, 10, 10,
        10, 10, 10,
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
        1, 1, 1;
    w_vel_diag << 1, 1, 1,
        1, 1, 1,
        0.1, 0.1, 0.1,
        0.1, 0.1, 0.1,
        0.1, 0.1, 0.1,
        0.1, 0.1, 0.1;
    w_x_diag << w_pos_diag, w_vel_diag;
    MatrixXd w_x_term = w_x_diag.asDiagonal();

    // Define input state weights
    MatrixXd w_u = MatrixXd::Identity(nu, nu);
    w_u.diagonal().array() = 1e-6;

    IntegratorSemiImplEuler discrete_dyn = IntegratorSemiImplEuler(dynamics, timestep);
    // IntegratorEuler discrete_dyn = IntegratorEuler(dynamics, timestep);
    DynamicsFiniteDifference finite_diff_dyn(space, discrete_dyn, 1e-6);

    std::vector<xyz::polymorphic<StageModel>> stage_models;
    for (size_t i = 0; i < nsteps; i++)
    {
        auto rcost = CostStack(space, nu);
        rcost.addCost("state_cost", QuadraticStateCost(space, nu, x_ref[i], w_x));
        rcost.addCost("control_cost", QuadraticControlCost(space, u0, w_u));

        // stage_models.push_back(StageModel(rcost, discrete_dyn));
        stage_models.push_back(StageModel(rcost, finite_diff_dyn));
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

    std::string urdf_filename = "/home/zishang/pinocchio_idto_drake_simulator/pinocchio_idto/robot/mini_cheetah/mini_cheetah_ground.urdf";
    std::string srdf_filename = "/home/zishang/pinocchio_idto_drake_simulator/pinocchio_idto/robot/mini_cheetah/mini_cheetah.srdf";

    pin::Model rmodel;
    pin::GeometryModel geom_model;
    pin::urdf::buildModel(urdf_filename, rmodel);
    pin::urdf::buildGeom(rmodel, urdf_filename, pinocchio::COLLISION, geom_model);
    geom_model.addAllCollisionPairs();
    pin::srdf::removeCollisionPairs(rmodel, geom_model, srdf_filename);
    MultibodyPhaseSpace space(rmodel);
    MatrixXd actuation = MatrixXd::Zero(rmodel.nv, 12);
    actuation.bottomRows(12).setIdentity();
    CompliantContactParameter contact_param;
    CompliantContactFwdDynamics dynamics(space, actuation, geom_model, contact_param);

    int nsteps = 100;
    double timestep = 0.05;
    const int nq = rmodel.nq;
    const int nv = rmodel.nv;

    VectorXd x0 = VectorXd::Zero(nq + nv);
    VectorXd u0 = VectorXd::Zero(dynamics.nu());
    x0.head(nq) << 0.0, 0.0, 0.16,
        0.0, 0.0, 0.0, 1.0,
        0.0, -1.3, 2.4,
        0.0, -1.3, 2.4,
        0.0, -1.3, 2.4,
        0.0, -1.3, 2.4;

    VectorXd x_nom = VectorXd::Zero(nq + nv);
    x_nom.head(nq) << 0.0, 0.0, 0.16,
        0.0, 0.0, 0.0, 1.0,
        0.0, -1.3, 2.4,
        0.0, -1.3, 2.4,
        0.0, -1.3, 2.4,
        0.0, -1.3, 2.4;

    std::vector<VectorXd> x_ref(nsteps, x_nom);

    auto problem = createTrajOptProblem(dynamics, nsteps, timestep, x_ref, x0, u0);

    double tol = 1e-4;
    int max_iters = 100;
    double mu_init = 1e-8;

    aligator::SolverProxDDPTpl<double> solver(tol, mu_init, max_iters, aligator::VerboseLevel::VERBOSE);

    std::vector<VectorXd> x_guess, u_guess;
    x_guess.assign(nsteps + 1, x0);
    u_guess.assign(nsteps, u0);

    solver.rollout_type_ = aligator::RolloutType::LINEAR;
    // solver.sa_strategy_ = aligator::StepAcceptanceStrategy::FILTER;
    // solver.filter_.beta_ = 1e-5;
    solver.force_initial_condition_ = true;
    // solver.reg_min = 1e-6;
    solver.linear_solver_choice = aligator::LQSolverChoice::PARALLEL;
    solver.setNumThreads(8); // call before setup()
    solver.setup(*problem);

    auto start = std::chrono::high_resolution_clock::now();

    // first solve
    solver.run(*problem, x_guess, u_guess);
    x_guess = solver.results_.xs;
    u_guess = solver.results_.us;

    // warm start
    std::cout << "----------------------warm start----------------------" << std::endl;
    for (int i = 0; i < nsteps; i++)
    {
        x_ref[i].head(nq) << 0.0, 0.0, 0.29,
            0.0, 0.0, 0.0, 1.0,
            0.0, -0.8, 1.6,
            0.0, -0.8, 1.6,
            0.0, -0.8, 1.6,
            0.0, -0.8, 1.6;
    }
    updateStateReferences(problem, x_ref);
    solver.max_iters = 1;
    solver.run(*problem, x_guess, u_guess);

    std::vector<VectorXd> x_result = solver.results_.xs;
    saveVectorsToCsv("aligator_cimpc.csv", x_result);

    return 0;
}
