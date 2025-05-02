#pragma once
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/geometry.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/collision/collision.hpp>
#include <cppad/cppad.hpp>

using pinocchio::Model;
using pinocchio::Data;
using pinocchio::Force;
using pinocchio::GeometryModel;
using pinocchio::GeometryData;
using pinocchio::container::aligned_vector;

using Eigen::Vector3d;

void CalcContactForce(const Model &model, const Data &data,
                      const GeometryModel &geom_model, GeometryData &geom_data,
                      aligned_vector<Force> &f_ext);


template <typename Scalar>
void CalcContactForceContribution(const pinocchio::ModelTpl<Scalar> &model,
                                  const pinocchio::DataTpl<Scalar> &data,
                                  aligned_vector<pinocchio::ForceTpl<Scalar>> &f_ext);

template <typename Scalar>
void CalcContactForceContributionAD(const pinocchio::ModelTpl<Scalar> &model,
                                    const pinocchio::DataTpl<Scalar> &data,
                                    aligned_vector<pinocchio::ForceTpl<Scalar>> &f_ext);