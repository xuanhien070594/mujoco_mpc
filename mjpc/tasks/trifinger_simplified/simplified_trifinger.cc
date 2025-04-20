// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/tasks/trifinger_simplified/simplified_trifinger.h"

#include <string>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>

#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
std::string SimplifiedTrifinger::XmlPath() const {
  return GetModelPath("trifinger_simplified/task.xml");
}
std::string SimplifiedTrifinger::Name() const { return "Simplified Trifinger"; }

// ----------- Residuals for jack manipulation task -----------
//   Number of residuals: 42
//     Residual (0-2) Norm (Quaratic): cube - target
//     Residual (3-5) Norm (Quadratic): cube orientation - target orientation
//     Residual (6-8) Norm (Quadratic): cube linear velocity
//     Residual (9-11) Norm (Quadratic): cube angular velocity
//     Residual (12-20) Norm (Quadratic): control inputs
//     Residual (21-23) Norm (Quadratic): fingertip0-cube distance
//     Residual (24-26) Norm (Quadratic): fingertip0-cube distance
//     Residual (27-29) Norm (Quadratic): fingertip0-cube distance
//     Residual (30) Norm (Null - Residual is Cost): fingertip0 to avoid
//     corners
//     Residual (31) Norm (Null - Residual is Cost): fingertip1 to void
//     corners
//     Residual (32) Norm (Null - Residual is Cost): fingertip2 to avoid
//     corners
//     Residual (33-35): fingertip0 linear velocity
//     Residual (36-38): fingertip1 linear velocity
//     Residual (39-41): fingertip2 linear velocity
// ------------------------------------------------------------
double CalcCostOnDistFingertipToOneCorner(double *fingertip_pos,
                                          double *corner_pos) {
  // any distance from a fingertip to a corner of the cube that is less than
  // penalized_radius will be penalized. The cost is 1 / squared distance
  // between fingertip and corner. This is to encourage the fingertips to be
  // away from the corners of the cube.
  double penalized_radius = 0.019;
  double squared_dist = 0;
  for (int i = 0; i < 3; i++) {
    squared_dist +=
        (fingertip_pos[i] - corner_pos[i]) * (fingertip_pos[i] - corner_pos[i]);
  }
  if (sqrt(squared_dist) < penalized_radius) {
    return 0;
  }
  return 1 / squared_dist;
}

double CalcCostOnDistFingertipToCorners(double *fingertip_pos,
                                        double *cube_corner_1_pos,
                                        double *cube_corner_2_pos,
                                        double *cube_corner_3_pos,
                                        double *cube_corner_4_pos) {
  double cost_on_dist_finger_corners = 0;
  cost_on_dist_finger_corners +=
      CalcCostOnDistFingertipToOneCorner(fingertip_pos, cube_corner_1_pos);
  cost_on_dist_finger_corners +=
      CalcCostOnDistFingertipToOneCorner(fingertip_pos, cube_corner_2_pos);
  cost_on_dist_finger_corners +=
      CalcCostOnDistFingertipToOneCorner(fingertip_pos, cube_corner_3_pos);
  cost_on_dist_finger_corners +=
      CalcCostOnDistFingertipToOneCorner(fingertip_pos, cube_corner_4_pos);
  return cost_on_dist_finger_corners;
}

void SimplifiedTrifinger::ResidualFn::Residual(const mjModel *model,
                                               const mjData *data,
                                               double *residual) const {
  int counter = 0;

  double *cube = SensorByName(model, data, "cube");
  double *cube_orientation = SensorByName(model, data, "cube_quat");
  double *target = SensorByName(model, data, "target");
  double *target_orientation = SensorByName(model, data, "target_quat");
  double *cube_linear_vel = SensorByName(model, data, "cube_linear_vel");
  double *cube_angular_vel = SensorByName(model, data, "cube_angular_vel");
  double *fingertip0_linear_vel =
      SensorByName(model, data, "fingertip0_linear_vel");
  double *fingertip1_linear_vel =
      SensorByName(model, data, "fingertip1_linear_vel");
  double *fingertip2_linear_vel =
      SensorByName(model, data, "fingertip2_linear_vel");
  double *fingertip0_pos = SensorByName(model, data, "fingertip0");
  double *fingertip1_pos = SensorByName(model, data, "fingertip1");
  double *fingertip2_pos = SensorByName(model, data, "fingertip2");
  double *cube_corner_1_pos = SensorByName(model, data, "corner_1");
  double *cube_corner_2_pos = SensorByName(model, data, "corner_2");
  double *cube_corner_3_pos = SensorByName(model, data, "corner_3");
  double *cube_corner_4_pos = SensorByName(model, data, "corner_4");

  mju_sub3(residual + counter, cube, target);
  counter += 3;
  // mju_sub3(residual + counter, cube_orientation, target_orientation);
  // counter += 3;
  for (int i = 0; i < 4; i++) {
    *(residual + counter) = *(cube_orientation + i) - *(target_orientation + i);
    counter++;
  }
  mju_copy3(residual + counter, cube_linear_vel);
  counter += 3;
  mju_copy3(residual + counter, cube_angular_vel);
  counter += 3;

  mju_copy(residual + counter, data->actuator_force, model->nu);
  counter += model->nu;

  mju_sub3(residual + counter, fingertip0_pos, cube);
  counter += 3;
  mju_sub3(residual + counter, fingertip1_pos, cube);
  counter += 3;
  mju_sub3(residual + counter, fingertip2_pos, cube);
  counter += 3;

  // Add cost to encourage fingertips to be far away from the corners of the
  // cube
  double cost_on_dist_finger0_corner = CalcCostOnDistFingertipToCorners(
      fingertip0_pos, cube_corner_1_pos, cube_corner_2_pos, cube_corner_3_pos,
      cube_corner_4_pos);
  double cost_on_dist_finger1_corner = CalcCostOnDistFingertipToCorners(
      fingertip1_pos, cube_corner_1_pos, cube_corner_2_pos, cube_corner_3_pos,
      cube_corner_4_pos);
  double cost_on_dist_finger2_corner = CalcCostOnDistFingertipToCorners(
      fingertip2_pos, cube_corner_1_pos, cube_corner_2_pos, cube_corner_3_pos,
      cube_corner_4_pos);
  *(residual + counter) = cost_on_dist_finger0_corner;
  counter++;
  *(residual + counter) = cost_on_dist_finger1_corner;
  counter++;
  *(residual + counter) = cost_on_dist_finger2_corner;
  counter++;

  mju_copy3(residual + counter, fingertip0_linear_vel);
  counter += 3;
  mju_copy3(residual + counter, fingertip1_linear_vel);
  counter += 3;
  mju_copy3(residual + counter, fingertip2_linear_vel);
  counter += 3;
  // sensor dim sanity check
  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    mju_error_i("mismatch between total user-sensor dimension "
                "and actual length of residual %d",
                counter);
  }
}
} // namespace mjpc