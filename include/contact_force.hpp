#pragma once
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/geometry.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/collision/collision.hpp>
#include <cppad/cppad.hpp>

using pinocchio::Data;
using pinocchio::Force;
using pinocchio::GeometryData;
using pinocchio::GeometryModel;
using pinocchio::Model;
using pinocchio::Motion;
using pinocchio::SE3;
using pinocchio::container::aligned_vector;

using Eigen::Vector3d;

void CalcContactForce(const Model &model, const Data &data,
                      const GeometryModel &geom_model, GeometryData &geom_data,
                      aligned_vector<Force> &f_ext)
{
    using std::abs, std::exp, std::log, std::max, std::pow, std::sqrt;

    double contact_stiffness = 2000;   // normal force stiffness, in N/m
    double smoothing_factor = 0.01;    // Amount of smoothing to apply when computing normal forces.
    double dissipation_velocity = 0.1; // Hunt & Crossley-like model parameter, in m/s.

    double stiction_velocity = 0.5;    // Regularization velocity, in m/s.
    double friction_coefficient = 1.0; // Coefficient of friction.

    // Compliant contact parameters
    const double &k = contact_stiffness;
    const double &sigma = smoothing_factor;
    // const double &dissipation_velocity = dissipation_velocity;

    // Friction parameters.
    const double &vs = stiction_velocity;    // Regularization.
    const double &mu = friction_coefficient; // Coefficient of friction.

    const double eps = sqrt(std::numeric_limits<double>::epsilon());
    double threshold = -sigma * log(exp(eps / (sigma * k)) - 1.0);

    // TODO: 是否可以删除临时变量？
    // position of contact poiont, expressed in world frame
    Vector3d pos_contact;

    // velocity of robot's geometry relative to environment's geometry at contact poiont, expressed in world frame.
    Vector3d vel_contact, vel_contact_t;

    // Contact force from geometry B to geometry A, expressed in world frame
    Vector3d force_contact, force_contact_t;
    Force force6d_contact = Force::Zero();

    // Transformation of contact point relative to geometry object A, geometry object B and wrold frame
    // Rotation is not taken into account
    SE3 X_AC = SE3::Identity();
    SE3 X_BC = SE3::Identity();
    SE3 X_WC = SE3::Identity();

    // Transformation of contact point relative to joint local frame A and joint local frame B
    SE3 X_JaC = SE3::Identity();
    SE3 X_JbC = SE3::Identity();

    // Motion of geometry object A and geometry object B at contact point
    Motion motion_geomAc = Motion::Zero();
    Motion motion_geomBc = Motion::Zero();

    // Normal vector from geometry object B to geometry object A
    Vector3d nhat;

    for (size_t pair_id = 0; pair_id < geom_model.collisionPairs.size(); pair_id++)
    {
        const pinocchio::CollisionPair &cp = geom_model.collisionPairs[pair_id];
        const auto &geomA_id = cp.first;
        const auto &geomB_id = cp.second;

        const hpp::fcl::DistanceResult &dr = geom_data.distanceResults[pair_id];
        const double &signed_distance = dr.min_distance;
        nhat = -dr.normal;

        // Joint index of geometry object in pin::Data
        const auto &jointA_id = geom_model.geometryObjects[geomA_id].parentJoint;
        const auto &jointB_id = geom_model.geometryObjects[geomB_id].parentJoint;

        if (signed_distance < threshold)
        {
            // The contact point is defined as the midpoint of the two nearest points
            const auto &pos_geomA = dr.nearest_points[0];
            const auto &pos_geomB = dr.nearest_points[1];
            pos_contact = (pos_geomA + pos_geomB) / 2;

            // Frame index of geometry object in pin::Data
            const auto &frameA_id = geom_model.geometryObjects[geomA_id].parentFrame;
            const auto &frameB_id = geom_model.geometryObjects[geomB_id].parentFrame;

            // Veolcity shift transformation
            X_AC.translation(pos_contact - pos_geomA);
            X_BC.translation(pos_contact - pos_geomB);

            // Relative velocity at contact point
            motion_geomAc = X_AC.actInv(pinocchio::getFrameVelocity(model, data, frameA_id, pinocchio::LOCAL_WORLD_ALIGNED));
            motion_geomBc = X_BC.actInv(pinocchio::getFrameVelocity(model, data, frameB_id, pinocchio::LOCAL_WORLD_ALIGNED));
            vel_contact = motion_geomAc.linear() - motion_geomBc.linear();

            // Split velocity into normal and tangential components.
            const double vn = nhat.dot(vel_contact);
            vel_contact_t = vel_contact - vn * nhat;

            // (Compliant) force in the normal direction increases linearly at a rate of k Newtons per meter,
            // with some smoothing defined by sigma.
            double compliant_fn;
            const double exponent = -signed_distance / sigma;
            if (exponent >= 37)
                // If the exponent is going to be very large, replace with the functional limit.
                // N.B. x = 37 is the first integer such that exp(x)+1 = exp(x) in double precision.
                compliant_fn = -k * signed_distance;
            else
                compliant_fn = sigma * k * log(1 + exp(exponent));

            // Normal dissipation follows a smoothed Hunt and Crossley model
            double dissipation_factor = 0.0;
            const double s = vn / dissipation_velocity;
            if (s < 0)
                dissipation_factor = 1 - s;
            else if (s < 2)
                dissipation_factor = (s - 2) * (s - 2) / 4;
            std::cout << "s: " << s << std::endl;
            const double fn = compliant_fn * dissipation_factor;

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
            const SE3 &X_WJa = data.oMi[jointA_id];
            const SE3 &X_WJb = data.oMi[jointB_id];
            X_JaC = X_WJa.inverse() * X_WC;
            X_JbC = X_WJb.inverse() * X_WC;

            // Transform contact force from contact frame to joint local frame
            f_ext[jointA_id] = X_JaC.act(force6d_contact);
            f_ext[jointB_id] = X_JbC.act(-force6d_contact);
        }
        else
        {
            f_ext[jointA_id].setZero();
            f_ext[jointB_id].setZero();
        }
    }
    std::cout << "---------------------------------------" << std::endl;
}

template <typename Scalar>
void CalcContactForceContribution(const pinocchio::ModelTpl<Scalar> &model,
                                  const pinocchio::DataTpl<Scalar> &data,
                                  aligned_vector<pinocchio::ForceTpl<Scalar>> &f_ext)
{

    const std::vector<int> foot_frame_ids{11, 19, 27, 35};
    const std::vector<int> joint_ids{4, 7, 10, 13};
    const double foot_radius = 0.0175;

    double contact_stiffness = 2000;   // normal force stiffness, in N/m
    double smoothing_factor = 0.01;    // Amount of smoothing to apply when computing normal forces.
    double dissipation_velocity = 0.1; // Hunt & Crossley-like model parameter, in m/s.

    double stiction_velocity = 0.5;    // Regularization velocity, in m/s.
    double friction_coefficient = 1.0; // Coefficient of friction.

    using std::abs, std::exp, std::log, std::max, std::pow, std::sqrt;

    // Compliant contact parameters
    const double &k = contact_stiffness;
    const double &sigma = smoothing_factor;
    // const double &dissipation_velocity = dissipation_velocity;

    // Friction parameters.
    const double &vs = stiction_velocity;    // Regularization.
    const double &mu = friction_coefficient; // Coefficient of friction.

    const double eps = sqrt(std::numeric_limits<double>::epsilon());
    double threshold = -sigma * log(exp(eps / (sigma * k)) - 1.0);

    // position of contact poiont, expressed in world frame
    Eigen::Vector3<Scalar> pos_contact;

    // velocity of robot's geometry relative to environment's geometry at contact poiont, expressed in world frame.
    Eigen::Vector3<Scalar> vel_contact, vel_contact_t;

    // Contact force from geometry B to geometry A, expressed in world frame
    Eigen::Vector3<Scalar> force_contact, force_contact_t;
    pinocchio::ForceTpl<Scalar> force6d_contact = pinocchio::ForceTpl<Scalar>::Zero();

    // Transformation of contact point relative to geometry object A, geometry object B and wrold frame
    // Rotation is not taken into account
    pinocchio::SE3Tpl<Scalar> X_AC = pinocchio::SE3Tpl<Scalar>::Identity();
    pinocchio::SE3Tpl<Scalar> X_BC = pinocchio::SE3Tpl<Scalar>::Identity();
    pinocchio::SE3Tpl<Scalar> X_WC = pinocchio::SE3Tpl<Scalar>::Identity();

    // Transformation of contact point relative to joint local frame A and joint local frame B
    pinocchio::SE3Tpl<Scalar> X_JaC = pinocchio::SE3Tpl<Scalar>::Identity();
    pinocchio::SE3Tpl<Scalar> X_JbC = pinocchio::SE3Tpl<Scalar>::Identity();

    // Motion of geometry object A and geometry object B at contact point
    pinocchio::MotionTpl<Scalar> motion_geomAc = pinocchio::MotionTpl<Scalar>::Zero();
    pinocchio::MotionTpl<Scalar> motion_geomBc = pinocchio::MotionTpl<Scalar>::Zero();

    // Normal vector from geometry object B to geometry object A
    Eigen::Vector3<Scalar> nhat;

    for (size_t pair_id = 0; pair_id < foot_frame_ids.size(); pair_id++)
    {
        Eigen::Vector3<Scalar> foot_frame_pos = data.oMf[foot_frame_ids[pair_id]].translation();

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
            motion_geomAc = X_AC.actInv(pinocchio::getFrameVelocity(model, data, foot_frame_ids[pair_id], pinocchio::LOCAL_WORLD_ALIGNED));
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
            const pinocchio::SE3Tpl<Scalar> &X_WJa = data.oMi[joint_ids[pair_id]];
            const pinocchio::SE3Tpl<Scalar> &X_WJb = data.oMi[0];
            X_JaC = X_WJa.inverse() * X_WC;
            X_JbC = X_WJb.inverse() * X_WC;

            // Transform contact force from contact frame to joint local frame
            f_ext[joint_ids[pair_id]] = X_JaC.act(force6d_contact);
            f_ext[0] = X_JbC.act(-force6d_contact);
        }
    }
}

template <typename Scalar>
void CalcContactForceContributionAD(const pinocchio::ModelTpl<Scalar> &model,
                                    const pinocchio::DataTpl<Scalar> &data,
                                    aligned_vector<pinocchio::ForceTpl<Scalar>> &f_ext)
{
    using CppAD::CondExpGe;
    using CppAD::CondExpGt;
    using CppAD::CondExpLt;
    using CppAD::exp;
    using CppAD::log;
    using CppAD::pow;
    using CppAD::sqrt;

    const std::vector<int> foot_frame_ids{11, 19, 27, 35};
    const std::vector<int> joint_ids{4, 7, 10, 13};
    const Scalar foot_radius = Scalar(0.0175);

    Scalar contact_stiffness = Scalar(2000);
    Scalar smoothing_factor = Scalar(0.01);
    Scalar dissipation_velocity = Scalar(0.1);
    Scalar stiction_velocity = Scalar(0.5);
    Scalar friction_coefficient = Scalar(1.0);

    const Scalar k = contact_stiffness;
    const Scalar sigma = smoothing_factor;
    const Scalar vs = stiction_velocity;
    const Scalar mu = friction_coefficient;

    const Scalar eps = sqrt(std::numeric_limits<double>::epsilon());
    Scalar threshold = -sigma * log(exp(eps / (sigma * k)) - Scalar(1.0));

    Eigen::Vector3<Scalar> nhat;
    nhat << 0, 0, 1;

    for (size_t pair_id = 0; pair_id < foot_frame_ids.size(); pair_id++)
    {
        Eigen::Vector3<Scalar> foot_frame_pos = data.oMf[foot_frame_ids[pair_id]].translation();

        Eigen::Vector3<Scalar> pos_geomA, pos_geomB;
        pos_geomA << foot_frame_pos(0), foot_frame_pos(1), foot_frame_pos(2) - foot_radius;
        pos_geomB << foot_frame_pos(0), foot_frame_pos(1), Scalar(0);

        const Scalar signed_distance = pos_geomA(2) - pos_geomB(2);
        Scalar is_contact = CondExpLt(signed_distance, threshold, Scalar(1), Scalar(0));

        // 中间变量初始化
        Eigen::Vector3<Scalar> pos_contact = (pos_geomA + pos_geomB) / Scalar(2);
        Eigen::Vector3<Scalar> vel_contact = Eigen::Vector3<Scalar>::Zero();
        Eigen::Vector3<Scalar> vel_contact_t = Eigen::Vector3<Scalar>::Zero();
        Eigen::Vector3<Scalar> force_contact = Eigen::Vector3<Scalar>::Zero();
        Eigen::Vector3<Scalar> force_contact_t = Eigen::Vector3<Scalar>::Zero();
        pinocchio::ForceTpl<Scalar> force6d_contact = pinocchio::ForceTpl<Scalar>::Zero();

        // Transformations
        pinocchio::SE3Tpl<Scalar> X_AC = pinocchio::SE3Tpl<Scalar>::Identity();
        pinocchio::SE3Tpl<Scalar> X_BC = pinocchio::SE3Tpl<Scalar>::Identity();
        pinocchio::SE3Tpl<Scalar> X_WC = pinocchio::SE3Tpl<Scalar>::Identity();
        pinocchio::SE3Tpl<Scalar> X_JaC = pinocchio::SE3Tpl<Scalar>::Identity();
        pinocchio::SE3Tpl<Scalar> X_JbC = pinocchio::SE3Tpl<Scalar>::Identity();

        pinocchio::MotionTpl<Scalar> motion_geomAc = pinocchio::MotionTpl<Scalar>::Zero();
        pinocchio::MotionTpl<Scalar> motion_geomBc = pinocchio::MotionTpl<Scalar>::Zero();

        // 位置、速度计算（只有在接触时才有效，但需保留计算图）
        X_AC.translation(pos_contact - pos_geomA);
        X_BC.translation(pos_contact - pos_geomB);
        motion_geomAc = X_AC.actInv(pinocchio::getFrameVelocity(model, data, foot_frame_ids[pair_id], pinocchio::LOCAL_WORLD_ALIGNED));
        motion_geomBc = pinocchio::MotionTpl<Scalar>::Zero();
        vel_contact = motion_geomAc.linear() - motion_geomBc.linear();

        const Scalar vn = nhat.dot(vel_contact);
        vel_contact_t = vel_contact - vn * nhat;

        // compliant normal force
        const Scalar exponent = -signed_distance / sigma;
        Scalar compliant_fn = CondExpGe(exponent, Scalar(37),
                                        -k * signed_distance,
                                        sigma * k * log(Scalar(1) + exp(exponent)));

        // dissipation model
        const Scalar s = vn / dissipation_velocity;
        Scalar dissipation_factor =
            CondExpLt(s, Scalar(0),
                      Scalar(1) - s,
                      CondExpLt(s, Scalar(2),
                                pow(s - Scalar(2), Scalar(2)) / Scalar(4),
                                Scalar(0)));

        const Scalar fn = compliant_fn * dissipation_factor;

        // tangential force (algebraic sigmoid model)
        force_contact_t = -vel_contact_t / sqrt(vs * vs + vel_contact_t.squaredNorm()) * mu * fn;

        // total force
        force_contact = is_contact * (nhat * fn + force_contact_t);
        force6d_contact.linear(force_contact);

        // contact transform
        X_WC.translation(pos_contact);
        const pinocchio::SE3Tpl<Scalar> &X_WJa = data.oMi[joint_ids[pair_id]];
        const pinocchio::SE3Tpl<Scalar> &X_WJb = data.oMi[0];
        X_JaC = X_WJa.inverse() * X_WC;
        X_JbC = X_WJb.inverse() * X_WC;

        // apply forces
        f_ext[joint_ids[pair_id]] = X_JaC.act(force6d_contact);
        f_ext[0] = X_JbC.act(-force6d_contact);
    }
}
