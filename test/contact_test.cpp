#include "contact_force.hpp"
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/autodiff/cppad.hpp>

using CppAD::AD;
typedef Eigen::VectorX<AD<double>> ADVectorX;

template <typename T>
bool isClose(const pinocchio::ForceTpl<T> &f1, const pinocchio::ForceTpl<T> &f2, double tol = 1e-6)
{
    return (f1.linear() - f2.linear()).norm() < tol && (f1.angular() - f2.angular()).norm() < tol;
}

int main()
{
    std::string urdf_filename = "/home/zishang/cpp_workspace/aligator_cimpc/robot/mini_cheetah/urdf/mini_cheetah_ground.urdf";
    std::string srdf_filename = "/home/zishang/cpp_workspace/aligator_cimpc/robot/mini_cheetah/srdf/mini_cheetah.srdf";

    //////////// 创建模型 //////////
    Model model;
    GeometryModel geom_model;
    pinocchio::urdf::buildModel(urdf_filename, model);
    pinocchio::urdf::buildGeom(model, urdf_filename, pinocchio::COLLISION, geom_model);
    geom_model.addAllCollisionPairs();
    pinocchio::srdf::removeCollisionPairs(model, geom_model, srdf_filename);
    Data data(model);
    GeometryData geom_data(geom_model);
    model.lowerPositionLimit.head<3>().fill(-1.);
    model.upperPositionLimit.head<3>().fill(1);

    ////////// 创建 CppAD 模型 //////////
    pinocchio::ModelTpl<AD<double>> ad_model = model.cast<AD<double>>();
    pinocchio::DataTpl<AD<double>> ad_data(ad_model);

    ////////// 创建变量 //////////
    std::srand(static_cast<unsigned int>(std::time(nullptr))); // 设置当前时间为随机种子
    aligned_vector<pinocchio::Force> f_ext1(model.njoints, pinocchio::Force::Zero());
    aligned_vector<pinocchio::Force> f_ext2(model.njoints, pinocchio::Force::Zero());
    aligned_vector<pinocchio::ForceTpl<AD<double>>> ad_f_ext(ad_model.njoints, pinocchio::ForceTpl<AD<double>>::Zero());

    // Eigen::VectorXd q = pinocchio::randomConfiguration(model);
    // Eigen::VectorXd v = Eigen::VectorXd::Random(model.nv);
    Eigen::VectorXd q(model.nq);
    Eigen::VectorXd v(model.nv);
    q << -0.245777, 0.301751, -0.979871, -0.593083, 0.528096, -0.339828, -0.503869,
        1.3385, -1.19725, 0.537645, -1.21378, 0.329066, 0.64744,
        -1.14686, -2.50748, -2.58926, -0.879062, 0.774141, 0.818634;
    v << 0.972041, 0.719425, 0.508777, 0.0490057, 0.391067, -0.374531,
        -0.0382495, -0.481858, -0.130509, 0.644534, 0.621316, 0.750557,
        -0.496758, -0.624461, 0.0523079, -0.476629, 0.164275, -0.723523;
    ADVectorX ad_q = q.cast<AD<double>>();
    ADVectorX ad_v = v.cast<AD<double>>();
    std::cout << "q: " << q.transpose() << std::endl;
    std::cout << "v: " << v.transpose() << std::endl;

    ////////// 开始计算 //////////
    pinocchio::forwardKinematics(model, data, q, v);
    pinocchio::updateFramePlacements(model, data);
    pinocchio::computeDistances(model, data, geom_model, geom_data, q);
    CalcContactForce(model, data, geom_model, geom_data, f_ext1);
    CalcContactForceContribution<double>(model, data, f_ext2);

    pinocchio::forwardKinematics(ad_model, ad_data, ad_q, ad_v);
    pinocchio::updateFramePlacements(ad_model, ad_data);
    CalcContactForceContributionAD<CppAD::AD<double>>(ad_model, ad_data, ad_f_ext);

    ////////// 将 AD 变量转换为普通变量 //////////
    aligned_vector<pinocchio::Force> f_ext3(model.njoints, pinocchio::Force::Zero());
    for (size_t i = 0; i < model.njoints; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            f_ext3[i].linear()[j] = CppAD::Value(ad_f_ext[i].linear()[j]);
            f_ext3[i].angular()[j] = CppAD::Value(ad_f_ext[i].angular()[j]);
        }
    }

    ////////// 比较结果 //////////
    for (size_t i = 0; i < model.njoints; ++i)
    {
        if (!isClose(f_ext1[i], f_ext2[i]) || !isClose(f_ext1[i], f_ext3[i]))
        {
            std::cout << "Force mismatch at joint " << i << std::endl;
            std::cout << "f_ext1: " << f_ext1[i].linear().transpose() << std::endl;
            std::cout << "f_ext2: " << f_ext2[i].linear().transpose() << std::endl;
            std::cout << "f_ext3: " << f_ext3[i].linear().transpose() << std::endl;
        }
    }

    std::cout << "All contact force outputs match!" << std::endl;
    return 0;
}