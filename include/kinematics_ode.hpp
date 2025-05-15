#pragma once

#include <aligator/modelling/dynamics/ode-abstract.hpp>
#include <proxsuite-nlp/modelling/spaces/multibody.hpp>

/**
 * @brief   Linear ordinary differential equation \f$\dot{x} = Ax + Bu\f$.
 *
 * @details This equation may be defined over a manifold's tangent space.
 */
struct KinematicsODE : aligator::dynamics::ODEAbstractTpl<double>
{
    using Base = ODEAbstractTpl<double>;
    using ODEData = aligator::dynamics::ContinuousDynamicsDataTpl<double>;
    using Manifold = proxsuite::nlp::MultibodyPhaseSpace<double>;

    MatrixXs A_, B_;
    Manifold space_;

    KinematicsODE(const Manifold &space)
        : Base(space, 2 * space.getModel().nv - 6)
        , space_(space)
    {
        const auto &model = space.getModel();
        const int &nv = model.nv;
        const int &nq = model.nq;

        A_.setZero(space.ndx(), nq + nv);
        A_.topRightCorner(nv, nv) = MatrixXs::Identity(nv, nv);
        B_.setZero(space.ndx(), nv + nv - 6);
        B_.bottomLeftCorner(nv, nv) = MatrixXs::Identity(nv, nv);
    }

    const Manifold &space() const { return space_; }

    void forward(const ConstVectorRef &x, const ConstVectorRef &u,
                 ODEData &data) const
    {
        data.xdot_ = A_ * x + B_ * u;
    }

    void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                  ODEData &data) const
    {
        return;
    }

    virtual std::shared_ptr<ODEData> createData() const
    {
        auto data = Base::createData();
        data->Jx_ = A_;
        data->Ju_ = B_;
        return data;
    }
};
