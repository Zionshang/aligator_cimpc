#pragma once

struct CompliantContactParameter
{
    double contact_stiffness = 2000;   // normal force stiffness, in N/m
    double smoothing_factor = 0.01;    // Amount of smoothing to apply when computing normal forces.
    double dissipation_velocity = 0.1; // Hunt & Crossley-like model parameter, in m/s.

    double stiction_velocity = 0.5;    // Regularization velocity, in m/s.
    double friction_coefficient = 1.0; // Coefficient of friction.
};
