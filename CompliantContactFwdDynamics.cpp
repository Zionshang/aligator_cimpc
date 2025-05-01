#include "CompliantContactFwdDynamics.hpp"

CompliantContactFwdDynamics::CompliantContactFwdDynamics(const MultibodyPhaseSpace &space, const MatrixXd &actuation,
                                                         const pin::GeometryModel &geom_model,
                                                         const CompliantContactParameter &param)
    : ODEAbstract(space, (int)actuation.cols()),
      space_(space), actuation_matrix_(actuation), geom_model_(geom_model),
      contact_param_(param)
{
    const int nv = space.getModel().nv;
    if (nv != actuation.rows())
    {
        ALIGATOR_DOMAIN_ERROR(
            fmt::format("actuation matrix should have number of rows = pinocchio "
                        "model nv ({} and {}).",
                        actuation.rows(), nv));
    }
}

void CompliantContactFwdDynamics::forward(const ConstVectorRef &x, const ConstVectorRef &u,
                                          ODEData &data) const
{
    CompliantContactFwdData &d = static_cast<CompliantContactFwdData &>(data);
    const pin::Model &model = space_.getModel();
    d.tau_.noalias() = actuation_matrix_ * u;

    const int nq = model.nq;
    const int nv = model.nv;
    const auto &q = x.head(nq);
    const auto &v = x.segment(nq, nv);

    pin::forwardKinematics(model, d.pin_data_, q, v);
    // pin::computeDistances(model, d.pin_data_, geom_model_, d.geom_data_, q);
    pin::updateFramePlacements(model, d.pin_data_);
    CalcContactForceContribution(model, d.pin_data_, geom_model_, d.geom_data_, d.f_ext_);

    data.xdot_.head(nv) = v;
    data.xdot_.segment(nv, nv) = pin::aba(model, d.pin_data_, q, v, d.tau_, d.f_ext_);
}

void CompliantContactFwdDynamics::dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                                           ODEData &data) const
{
    // const double eps = 1e-6;
    // const int nx = x.size();
    // const int nu = u.size();
    // const int nd = data.xdot_.size();

    // // 计算基准输出 f(x,u)
    // forward(x, u, data);
    // VectorXd f0 = data.xdot_;

    // // 初始化雅可比矩阵
    // data.Jx_.resize(nd, nx);
    // data.Ju_.resize(nd, nu);

    // // 针对状态进行有限差分
    // for (int i = 0; i < nx; i++)
    // {
    //     VectorXd x_pert = x;
    //     x_pert(i) += eps;
    //     auto dataPert = createData();
    //     forward(x_pert, u, *dataPert);
    //     data.Jx_.col(i) = (dataPert->xdot_ - f0) / eps;
    // }

    // // 针对控制进行有限差分
    // for (int i = 0; i < nu; i++)
    // {
    //     VectorXd u_pert = u;
    //     u_pert(i) += eps;
    //     auto dataPert = createData();
    //     forward(x, u_pert, *dataPert);
    //     data.Ju_.col(i) = (dataPert->xdot_ - f0) / eps;
    // }
}

void CompliantContactFwdDynamics::CalcContactForceContribution(const pin::Model &rmodel, const pin::Data &rdata,
                                                               const pin::GeometryModel &geom_model, pin::GeometryData &geom_data,
                                                               pin::container::aligned_vector<pin::Force> &f_ext) const
{
    const std::vector<int> foot_frame_ids{11, 19, 27, 35};
    const std::vector<int> joint_ids{4, 7, 10, 13};
    const double foot_radius = 0.0175;

    using std::abs, std::exp, std::log, std::max, std::pow, std::sqrt;

    // Compliant contact parameters
    const double &k = contact_param_.contact_stiffness;
    const double &sigma = contact_param_.smoothing_factor;
    const double &dissipation_velocity = contact_param_.dissipation_velocity;

    // Friction parameters.
    const double &vs = contact_param_.stiction_velocity;    // Regularization.
    const double &mu = contact_param_.friction_coefficient; // Coefficient of friction.

    const double eps = sqrt(std::numeric_limits<double>::epsilon());
    double threshold = -sigma * log(exp(eps / (sigma * k)) - 1.0);

    // position of contact poiont, expressed in world frame
    Vector3d pos_contact;

    // velocity of robot's geometry relative to environment's geometry at contact poiont, expressed in world frame.
    Vector3d vel_contact, vel_contact_t;

    // Contact force from geometry B to geometry A, expressed in world frame
    Vector3d force_contact, force_contact_t;
    pin::Force force6d_contact = pin::Force::Zero();

    // Transformation of contact point relative to geometry object A, geometry object B and wrold frame
    // Rotation is not taken into account
    pin::SE3 X_AC = pin::SE3::Identity();
    pin::SE3 X_BC = pin::SE3::Identity();
    pin::SE3 X_WC = pin::SE3::Identity();

    // Transformation of contact point relative to joint local frame A and joint local frame B
    pin::SE3 X_JaC = pin::SE3::Identity();
    pin::SE3 X_JbC = pin::SE3::Identity();

    // Motion of geometry object A and geometry object B at contact point
    pin::Motion motion_geomAc = pin::Motion::Zero();
    pin::Motion motion_geomBc = pin::Motion::Zero();

    // Normal vector from geometry object B to geometry object A
    Vector3d nhat;

    for (size_t pair_id = 0; pair_id < geom_model.collisionPairs.size(); pair_id++)
    {
        Eigen::Vector3<Scalar> foot_frame_pos = rdata.oMf[foot_frame_ids[pair_id]].translation();

        Eigen::Vector3<Scalar> pos_geomA, pos_geomB;
        pos_geomA << foot_frame_pos(0), foot_frame_pos(1), foot_frame_pos(2) - foot_radius; // foot
        pos_geomB << foot_frame_pos(0), foot_frame_pos(1), 0;                               // ground

        const Scalar signed_distance = pos_geomA(2) - pos_geomB(2);
        nhat << 0, 0, 1;
        if (signed_distance < threshold)
        {
            // The contact point is defined as the midpoint of the two nearest points
            pos_contact = (pos_geomA + pos_geomB) / 2;

            // Veolcity shift transformation
            X_AC.translation(pos_contact - pos_geomA);
            X_BC.translation(pos_contact - pos_geomB);

            // Relative velocity at contact point
            motion_geomAc = X_AC.actInv(pinocchio::getFrameVelocity(rmodel, rdata, foot_frame_ids[pair_id], pinocchio::LOCAL_WORLD_ALIGNED));
            motion_geomBc = pinocchio::MotionTpl<Scalar>::Zero();
            vel_contact = motion_geomAc.linear() - motion_geomBc.linear();

            // Split velocity into normal and tangential components.
            const Scalar vn = nhat.dot(vel_contact);
            vel_contact_t = vel_contact - vn * nhat;

            // (Compliant) force in the normal direction increases linearly at a rate of k Newtons per meter,
            // with some smoothing defined by sigma.
            Scalar compliant_fn;
            const Scalar exponent = -signed_distance / sigma;
            if (exponent >= 37)
                // If the exponent is going to be very large, replace with the functional limit.
                // N.B. x = 37 is the first integer such that exp(x)+1 = exp(x) in double precision.
                compliant_fn = -k * signed_distance;
            else
                compliant_fn = sigma * k * log(1 + exp(exponent));

            // Normal dissipation follows a smoothed Hunt and Crossley model
            Scalar dissipation_factor = 0.0;
            const Scalar s = vn / dissipation_velocity;
            if (s < 0)
                dissipation_factor = 1 - s;
            else if (s < 2)
                dissipation_factor = (s - 2) * (s - 2) / 4;

            const Scalar fn = compliant_fn * dissipation_factor;

            // Tangential frictional component.
            // N.B. This model is algebraically equivalent to: ft = -mu*fn*sigmoid(||vt||/vs)*vt/||vt||.
            //      with the algebraic sigmoid function defined as sigmoid(x) = x/sqrt(1+x^2).
            //      The algebraic simplification here is performed to avoid division by zero when vt = 0
            //      (or loss of precision when close to zero).
            force_contact_t = -vel_contact_t / sqrt(vs * vs + vel_contact_t.squaredNorm()) * mu * fn;
            force_contact = nhat * fn + force_contact_t;
            force6d_contact.linear(force_contact);

            // Transformation of joint local frame relative to contact frame.
            // Contact frame is fixed at contact point and alienged with world frame.
            X_WC.translation(pos_contact);
            const pinocchio::SE3Tpl<Scalar> &X_WJa = rdata.oMi[joint_ids[pair_id]];
            const pinocchio::SE3Tpl<Scalar> &X_WJb = rdata.oMi[0];
            X_JaC = X_WJa.inverse() * X_WC;
            X_JbC = X_WJb.inverse() * X_WC;

            // Transform contact force from contact frame to joint local frame
            f_ext[joint_ids[pair_id]] = X_JaC.act(force6d_contact);
            f_ext[0] = X_JbC.act(-force6d_contact);
        }
        else
        {
            f_ext[joint_ids[pair_id]].setZero();
            f_ext[0].setZero();
        }
    }
}

// 必须得有，否则会报错
std::shared_ptr<ODEData> CompliantContactFwdDynamics::createData() const
{
    return std::make_shared<CompliantContactFwdData>(*this);
}

CompliantContactFwdData::CompliantContactFwdData(const CompliantContactFwdDynamics &dynamics)
    : ODEData(dynamics.ndx(), dynamics.nu()),
      tau_(dynamics.space_.getModel().nv),
      geom_model_(dynamics.geom_model_),
      contact_param_(dynamics.contact_param_)
{
    const pin::Model &model = dynamics.space_.getModel();
    pin_data_ = pin::Data(model);
    geom_data_ = pin::GeometryData(geom_model_);
    f_ext_.assign(model.njoints, pin::Force::Zero());
}
