#pragma once

template <typename Scalar>
struct ContactParams
{
  // normal force stiffness, in N/m
  Scalar contact_stiffness = 2000;
  // Amount of smoothing to apply when computing normal forces.
  Scalar smoothing_factor = 0.01;
  // Hunt & Crossley-like model parameter, in m/s.
  Scalar dissipation_velocity = 0.1;
  // Regularization velocity, in m/s.
  Scalar stiction_velocity = 0.5;
  // Coefficient of friction.
  Scalar friction_coefficient = 1.0;

  ContactParams() = default;

  template <typename OtherScalar>
  explicit ContactParams(const ContactParams<OtherScalar> &other)
      : contact_stiffness(static_cast<Scalar>(other.contact_stiffness)),
        smoothing_factor(static_cast<Scalar>(other.smoothing_factor)),
        dissipation_velocity(static_cast<Scalar>(other.dissipation_velocity)),
        stiction_velocity(static_cast<Scalar>(other.stiction_velocity)),
        friction_coefficient(static_cast<Scalar>(other.friction_coefficient)) {}
};
