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

#include <pinocchio/algorithm/rnea.hpp>
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

VectorXd calcNominalTorque(const pin::Model &model, const VectorXd &q_nom)
{
    int nq = model.nq;
    int nv = model.nv;
    pin::Data data(model);
    pin::rnea(model, data, q_nom, VectorXd::Zero(nv), VectorXd::Zero(nv));
    return data.tau.tail(nv - 6);
}

void computeFutureStates(const pin::Model &model,
                         const VectorXd &v_ref,
                         const VectorXd &x0,
                         std::vector<VectorXd> &x_ref)
{
    pin::Data data(model);

    int nq = model.nq;
    int nv = model.nv;

    VectorXd q = x0.head(nq);
    VectorXd v = x0.tail(nv);

    x_ref[0] = x0;

    for (int i = 1; i < x_ref.size(); ++i)
    {

        integrate(model, x_ref[i - 1].head(nq), x_ref[i - 1].tail(nv), x_ref[i].head(nq));
        x_ref[i].tail(nv) = v_ref;
    }
}

std::shared_ptr<TrajOptProblem> createTrajOptProblem(const CompliantContactFwdDynamics &dynamics,
                                                     int nsteps, double timestep,
                                                     const std::vector<VectorXd> &x_ref,
                                                     const std::vector<VectorXd> &u_ref,
                                                     const VectorXd &x0)
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
    w_u.diagonal().array() = 1e-4;

    IntegratorSemiImplEuler discrete_dyn = IntegratorSemiImplEuler(dynamics, timestep);
    DynamicsFiniteDifference finite_diff_dyn(space, discrete_dyn, 1e-8);
    // DynamicsFiniteDifference finite_diff_dyn(space, discrete_dyn, timestep);

    std::vector<xyz::polymorphic<StageModel>> stage_models;
    for (size_t i = 0; i < nsteps; i++)
    {
        auto rcost = CostStack(space, nu);
        rcost.addCost("state_cost", QuadraticStateCost(space, nu, x_ref[i], w_x));
        rcost.addCost("control_cost", QuadraticControlCost(space, u_ref[i], w_u));

        StageModel sm = StageModel(rcost, finite_diff_dyn);
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

    std::string urdf_filename = "/home/zishang/cpp_workspace/aligator_cimpc/robot/mini_cheetah/urdf/mini_cheetah_ground.urdf";
    std::string srdf_filename = "/home/zishang/cpp_workspace/aligator_cimpc/robot/mini_cheetah/srdf/mini_cheetah.srdf";

    pin::Model model;
    pin::GeometryModel geom_model;
    pin::urdf::buildModel(urdf_filename, model);
    pin::urdf::buildGeom(model, urdf_filename, pinocchio::COLLISION, geom_model);
    geom_model.addAllCollisionPairs();
    pin::srdf::removeCollisionPairs(model, geom_model, srdf_filename);
    MultibodyPhaseSpace space(model);
    MatrixXd actuation = MatrixXd::Zero(model.nv, 12);
    actuation.bottomRows(12).setIdentity();
    CompliantContactParameter contact_param;
    CompliantContactFwdDynamics dynamics(space, actuation, geom_model, contact_param);

    int nsteps = 100;
    double timestep = 0.01;
    const int nq = model.nq;
    const int nv = model.nv;

    /************************initial state**********************/
    VectorXd x0 = VectorXd::Zero(nq + nv);
    x0.head(nq) << 0.0, 0.0, 0.29,
        0.0, 0.0, 0.0, 1.0,
        0.0, -0.8, 1.6,
        0.0, -0.8, 1.6,
        0.0, -0.8, 1.6,
        0.0, -0.8, 1.6;

    /************************reference state**********************/
    VectorXd v_ref = VectorXd::Zero(nv); 
    v_ref(0) = 0.01;

    std::vector<VectorXd> x_ref(nsteps, x0);
    computeFutureStates(model, v_ref, x0, x_ref);
    VectorXd u_nom = calcNominalTorque(model, x0.head(nq));
    std::vector<VectorXd> u_ref(nsteps, u_nom);

    /************************print reference state**********************/
    std::cout << "Printing x_ref values:" << std::endl;
    for (size_t i = 0; i < x_ref.size(); ++i) {
        std::cout << "x_ref[" << i << "] = " << std::endl << x_ref[i].transpose() << std::endl;
    }
    /************************create problem**********************/
    auto problem = createTrajOptProblem(dynamics, nsteps, timestep, x_ref, u_ref, x0);
    double tol = 1e-4;
    int max_iters = 1000;
    double mu_init = 1e-6;
    aligator::SolverProxDDPTpl<double> solver(tol, mu_init, max_iters, aligator::VerboseLevel::VERBOSE);
    std::vector<VectorXd> x_guess, u_guess;
    x_guess.assign(nsteps + 1, x0);
    u_guess.assign(nsteps, u_nom);
    solver.rollout_type_ = aligator::RolloutType::LINEAR;
    solver.force_initial_condition_ = true;
    solver.linear_solver_choice = aligator::LQSolverChoice::PARALLEL;
    solver.sa_strategy_ = aligator::StepAcceptanceStrategy::LINESEARCH_NONMONOTONE;
    solver.filter_.beta_ = 1e-5;
    solver.reg_min = 1e-6;
    solver.setNumThreads(8);
    solver.setup(*problem);

    /************************first solve**********************/
    solver.run(*problem, x_guess, u_guess);
    saveVectorsToCsv("offline_test.csv", solver.results_.xs);


    return 0;
}
