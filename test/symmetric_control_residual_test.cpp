#include "symmetric_control_residual.hpp"
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <proxsuite-nlp/modelling/spaces/multibody.hpp>

using MultibodyPhaseSpace = proxsuite::nlp::MultibodyPhaseSpace<double>;

int main()
{
    std::string urdf_filename = EXAMPLE_ROBOT_DATA_MODEL_DIR "/go2_description/urdf/go2.urdf";

    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf_filename, pinocchio::JointModelFreeFlyer(), model);
    MultibodyPhaseSpace space(model);

    int ndx = space.ndx();
    int na = model.nv - 6;
    int nu = model.nv + na;

    SymmetricControlResidual residual(ndx, nu, na);

    Eigen::VectorXd x(model.nq + model.nv);
    Eigen::VectorXd u(nu);
    x << pinocchio::neutral(model), Eigen::VectorXd::Zero(model.nv);
    u << Eigen::VectorXd::Zero(model.nv),
        1, 1, 1,
        2, 2, 2,
        3, 3, 3,
        4, 4, 4;

    auto data = residual.createData();

    residual.evaluate(x, u, *data);
    residual.computeJacobians(x, u, *data);

    std::cout << "Value: \n"
              << data->value_ << std::endl;
    std::cout << "Jacobian w.r.t. u: \n"
              << data->Ju_ << std::endl;
    return 0;
}