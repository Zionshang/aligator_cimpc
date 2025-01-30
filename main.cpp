#include <aligator/core/traj-opt-problem.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>

#include <aligator/modelling/costs/sum-of-costs.hpp>
#include <aligator/modelling/costs/quad-state-cost.hpp>
#include <aligator/modelling/dynamics/integrator-semi-euler.hpp>
#include <aligator/solvers/proxddp/solver-proxddp.hpp>
#include <aligator/modelling/autodiff/finite-difference.hpp>
#include <fstream>

#include "CompliantContactFwdDynamics.hpp"

using aligator::context::TrajOptProblem;
using StageModel = aligator::StageModelTpl<double>;
using IntegratorSemiImplEuler = aligator::dynamics::IntegratorSemiImplEulerTpl<double>;
using CostStack = aligator::CostStackTpl<double>;
using QuadraticStateCost = aligator::QuadraticStateCostTpl<double>;
using QuadraticControlCost = aligator::QuadraticControlCostTpl<double>;
using DynamicsFiniteDifference = aligator::autodiff::DynamicsFiniteDifferenceHelper<double>;

TrajOptProblem createTrajOptProblem(const CompliantContactFwdDynamics &dynamics,
                                    int nsteps, double timestep,
                                    const VectorXd &x_ref_start, const VectorXd &x_ref_end,
                                    const VectorXd &x0, const VectorXd &u0)
{
    const auto space = dynamics.space();
    const int nu = dynamics.nu(); // Number of control inputs
    const int ndx = space.ndx();  // Number of state variables

    // Define stage state weights
    VectorXd w_pos_diag = VectorXd::Zero(space.getModel().nv);
    VectorXd w_vel_diag = VectorXd::Zero(space.getModel().nv);
    w_pos_diag << 10, 10, 50, // linear part
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
    w_pos_diag << 10, 10, 50,
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
    w_u.diagonal().array() = 0.01;

    IntegratorSemiImplEuler discrete_dyn = IntegratorSemiImplEuler(dynamics, timestep);
    DynamicsFiniteDifference finite_diff_dyn(space, discrete_dyn, 1e-6);

    std::vector<xyz::polymorphic<StageModel>> stage_models;
    for (size_t i = 0; i < nsteps; i++)
    {
        // 计算当前时刻的参考状态（线性插值）
        double alpha = static_cast<double>(i) / (nsteps - 1);
        VectorXd x_ref = (1.0 - alpha) * x_ref_start + alpha * x_ref_end;
        std::cout << "x_ref: " << x_ref.transpose() << std::endl;
        auto rcost = CostStack(space, nu);
        rcost.addCost("quad_state", QuadraticStateCost(space, nu, x_ref, w_x));
        rcost.addCost("quad_control", QuadraticControlCost(space, u0, w_u));

        // stage_models.push_back(StageModel(rcost, discrete_dyn));
        stage_models.push_back(StageModel(rcost, finite_diff_dyn));
    }

    auto term_cost = QuadraticStateCost(space, nu, x_ref_end, w_x_term);
    TrajOptProblem problem(x0, stage_models, term_cost);
    std::cout << "problem.numSteps() " << problem.numSteps() << std::endl;
    return problem;
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
    VectorXd x_ref_start = VectorXd::Zero(nq + nv);
    x_ref_start.head(nq) << 0.0, 0.0, 0.29,
        0.0, 0.0, 0.0, 1.0,
        0.0, -0.8, 1.6,
        0.0, -0.8, 1.6,
        0.0, -0.8, 1.6,
        0.0, -0.8, 1.6;
    VectorXd x_ref_end = VectorXd::Zero(nq + nv);
    x_ref_end.head(nq) << 0.5, 0.0, 0.29,
        0.0, 0.0, 0.0, 1.0,
        0.0, -0.8, 1.6,
        0.0, -0.8, 1.6,
        0.0, -0.8, 1.6,
        0.0, -0.8, 1.6;
    VectorXd x0 = x_ref_start;
    VectorXd u0 = VectorXd::Zero(dynamics.nu());

    auto problem = createTrajOptProblem(dynamics,
                                        nsteps, timestep,
                                        x_ref_start, x_ref_end,
                                        x0, u0);

    double tol = 1e-4;
    int max_iters = 100;
    double mu_init = 1e-8;

    aligator::SolverProxDDPTpl<double> solver(tol, mu_init, max_iters, aligator::VerboseLevel::VERBOSE);

    std::vector<VectorXd> x_guess, u_guess;
    x_guess.assign(nsteps + 1, x0);
    u_guess.assign(nsteps, u0);

    solver.rollout_type_ = aligator::RolloutType::LINEAR;
    solver.sa_strategy_ = aligator::StepAcceptanceStrategy::FILTER;
    solver.filter_.beta_ = 1e-5;
    solver.force_initial_condition_ = true;
    solver.reg_min = 1e-6;
    solver.linear_solver_choice = aligator::LQSolverChoice::PARALLEL;
    solver.setNumThreads(4); // call before setup()
    solver.setup(problem);

    auto start = std::chrono::high_resolution_clock::now();

    solver.run(problem, x_guess, u_guess);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    fmt::print("Total execution time: {} ms\n", duration.count());

    auto &res = solver.results_;
    fmt::print("Results: {}\n", res);

    // 保存优化结果到CSV文件
    std::ofstream outFile("trajectory_results.csv");
    if (outFile.is_open())
    {
        // 写入数据
        for (size_t i = 0; i < solver.results_.xs.size(); i++)
        {
            VectorXd state = solver.results_.xs[i].head(nq);
            for (int j = 0; j < nq; ++j)
            {
                outFile << state[j];
                if (j < nq - 1)
                    outFile << ",";
            }
            outFile << "\n";
        }
        outFile.close();
        std::cout << "Results saved to trajectory_results.csv" << std::endl;
    }
    else
    {
        std::cerr << "Unable to open file for writing" << std::endl;
    }

    return 0;
}
