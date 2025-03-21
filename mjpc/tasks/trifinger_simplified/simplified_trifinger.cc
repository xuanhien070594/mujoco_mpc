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
//   Number of residuals: 12
//     Residual (0-2): cube - target
//     Residual (3-5): cube orientation - target orientation
//     Residual (6-8): cube linear velocity
//     Residual (9-11): cube angular velocity
// ------------------------------------------------------------
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

  mju_sub3(residual + counter, cube, target);
  counter += 3;
  mju_subQuat(residual + counter, cube_orientation, target_orientation);
  counter += 3;
  mju_copy3(residual + counter, cube_linear_vel);
  counter += 3;
  mju_copy3(residual + counter, cube_angular_vel);
  counter += 3;
  mju_copy3(residual + counter, fingertip0_linear_vel);
  counter += 3;
  mju_copy3(residual + counter, fingertip1_linear_vel);
  counter += 3;
  mju_copy3(residual + counter, fingertip2_linear_vel);
  counter += 3;

  // sensor dim sanity check
  // TODO: use this pattern everywhere and make this a utility function
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

// void SimplifiedTrifinger::TransitionLocked(mjModel *model, mjData *data) {
//   /* Order of mocap bodies:  intermediate goal, final goal, end effector goal
//   */
//   // First, always update the end effector desired location to be right above
//   // the jack's current location.
//   double *cube_position = SensorByName(model, data, "cube");
//   data->mocap_pos[6] = cube_position[0];
//   data->mocap_pos[7] = cube_position[1];
//   data->mocap_pos[8] = cube_position[2] + 0.1;

//   // Second, update the jack final desired location if the jack has reached
//   its
//   // final goal.
//   double residuals[100];
//   residual_.Residual(model, data, residuals);
//   double position_delta_vector[3] = {-residuals[18], -residuals[19],
//                                      -residuals[20]};
//   double rotation_delta_vector[3] = {-residuals[21], -residuals[22],
//                                      -residuals[23]};
//   double position_error = (mju_norm3(position_delta_vector));
//   double rotation_error = (mju_norm3(rotation_delta_vector));

//   if (data->time > 0 && position_error < POSITION_SUCCESS_THRESHOLD_ &&
//       rotation_error < ROTATION_SUCCESS_THRESHOLD_) {
//     std::cout << "Reached position (" << position_error << "m) and rotation
//     ("
//               << rotation_error << "rad) tolerances";

//     // Toggle between two goals.
//     if (data->mocap_pos[0] == 0.45 && data->mocap_pos[1] == 0.2) {
//       std::cout << " --> Switching to goal 2" << std::endl;
//       data->mocap_pos[3] = 0.5;
//       data->mocap_pos[4] = -0.15;
//       data->mocap_pos[5] = 0.032;
//       data->mocap_quat[4] = -0.452;
//       data->mocap_quat[5] = -0.627;
//       data->mocap_quat[6] = 0.629;
//       data->mocap_quat[7] = 0;
//     } else {
//       std::cout << " --> Switching to goal 1" << std::endl;
//       data->mocap_pos[3] = 0.45;
//       data->mocap_pos[4] = 0.2;
//       data->mocap_pos[5] = 0.032;
//       data->mocap_quat[4] = 0.880;
//       data->mocap_quat[5] = 0.280;
//       data->mocap_quat[6] = -0.365;
//       data->mocap_quat[7] = -0.116;
//     }
//   }

//   // Third, adjust the intermediate target to impose a lookahead on the final
//   // goal.
//   mjtNum intermediate_pos[3] = {data->mocap_pos[3], data->mocap_pos[4],
//                                 data->mocap_pos[5]};
//   mjtNum intermediate_quat[4] = {data->mocap_quat[4], data->mocap_quat[5],
//                                  data->mocap_quat[6], data->mocap_quat[7]};
//   if (position_error > POSITION_LOOKAHEAD_) {
//     double scale = POSITION_LOOKAHEAD_ / position_error;
//     double scaled_position_delta_vec[3] = {scale * position_delta_vector[0],
//                                            scale * position_delta_vector[1],
//                                            scale * position_delta_vector[2]};
//     intermediate_pos[0] = cube_position[0] + scaled_position_delta_vec[0];
//     intermediate_pos[1] = cube_position[1] + scaled_position_delta_vec[1];
//     intermediate_pos[2] = cube_position[2] + scaled_position_delta_vec[2];
//   }
//   if (rotation_error > ROTATION_LOOKAHEAD_) {
//     double scale = ROTATION_LOOKAHEAD_ / rotation_error;
//     double scaled_rotation_delta_vec[3] = {scale * rotation_delta_vector[0],
//                                            scale * rotation_delta_vector[1],
//                                            scale * rotation_delta_vector[2]};
//     double *jack_quat = SensorByName(model, data, "jack_quat");
//     mjtNum jack_quat_copy[4];
//     mju_copy4(jack_quat_copy, jack_quat);
//     mju_quatIntegrate(jack_quat_copy, scaled_rotation_delta_vec, 1);
//     intermediate_quat[0] = jack_quat_copy[0];
//     intermediate_quat[1] = jack_quat_copy[1];
//     intermediate_quat[2] = jack_quat_copy[2];
//     intermediate_quat[3] = jack_quat_copy[3];
//   }
//   data->mocap_pos[0] = intermediate_pos[0];
//   data->mocap_pos[1] = intermediate_pos[1];
//   data->mocap_pos[2] = intermediate_pos[2];
//   data->mocap_quat[0] = intermediate_quat[0];
//   data->mocap_quat[1] = intermediate_quat[1];
//   data->mocap_quat[2] = intermediate_quat[2];
//   data->mocap_quat[3] = intermediate_quat[3];
// }
} // namespace mjpc