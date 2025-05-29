#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/frames.hpp>

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main(int argc, char const *argv[])
{
    std::string urdf_filename = "/home/zishang/cpp_workspace/aligator_cimpc/robot/mini_cheetah/urdf/mini_cheetah_ground_mesh.urdf";

    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf_filename, model);
    pinocchio::Data data(model);

    const int nu = model.nv - 6;
    const int nq = model.nq;
    const int nv = model.nv;
    MatrixXd actuation = MatrixXd::Zero(model.nv, nu);
    actuation.bottomRows(nu).setIdentity();

    pinocchio::FrameIndex foot_frame_id1 = model.getFrameId("LF_FOOT");
    pinocchio::FrameIndex foot_frame_id2 = model.getFrameId("RF_FOOT");
    pinocchio::FrameIndex foot_frame_id3 = model.getFrameId("LH_FOOT");
    pinocchio::FrameIndex foot_frame_id4 = model.getFrameId("RH_FOOT");

    std::vector<pinocchio::FrameIndex> foot_frame_ids = {
        foot_frame_id1, foot_frame_id2, foot_frame_id3, foot_frame_id4};

    VectorXd q(nq), v(nv), tau(nu), f(12);
    q << 0.000, 0.000, 0.295, 0.000, 0.000, 0.000, 1.000, 0.000, -0.800, 1.600, -0.000, -0.800, 1.600, 0.000, -0.800, 1.600, -0.000, -0.800, 1.600;
    v << 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.001, -0.000, 0.003, -0.001, -0.000, 0.002, 0.001, -0.000, 0.003, -0.001, -0.000, 0.002;
    tau << -2.741, -0.136, -6.323, 2.976, -0.067, -5.274, -2.268, 0.251, -6.140, 2.431, 0.267, -5.090;
    f << -0.010, -0.006, 13.901, -0.007, 0.008, 13.914, -0.009, -0.005, 13.908, -0.006, 0.007, 13.920;

    // q << 0.000, 0.000, 0.295, -0.000, 0.000, -0.000, 1.000, 0.000, -0.800, 1.600, -0.000, -0.800, 1.600, 0.000, -0.800, 1.600, -0.000, -0.800, 1.600;
    // v << 0.000, 0.000, -0.000, -0.001, 0.000, -0.000, 0.001, -0.000, 0.003, -0.001, -0.000, 0.002, 0.001, -0.000, 0.003, -0.001, -0.000, 0.002;
    // tau << -0.455, 0.131, -1.743, 0.659, 0.096, -1.783, -0.431, 0.135, -1.741, 0.582, 0.075, -1.789;
    // f << -0.010, -0.006, 13.927, -0.007, 0.008, 13.919, -0.009, -0.005, 13.913, -0.006, 0.007, 13.905;

    MatrixXd J(12, nv);
    for (size_t i = 0; i < 4; i++)
    {
        MatrixXd J_temp = MatrixXd::Zero(6, nv);
        pinocchio::computeFrameJacobian(model, data, q, foot_frame_ids[i], pinocchio::LOCAL_WORLD_ALIGNED, J_temp);
        J.middleRows<3>(i * 3) = J_temp.topRows<3>();
    }

    pinocchio::crba(model, data, q);
    data.M.triangularView<Eigen::StrictlyLower>() = data.M.transpose().triangularView<Eigen::StrictlyLower>();
    pinocchio::nonLinearEffects(model, data, q, v);

    VectorXd a = data.M.inverse() * (-data.nle + actuation * tau + J.transpose() * f);

    std::cout << "a:\n"
              << a.transpose() << std::endl;
    return 0;
}
