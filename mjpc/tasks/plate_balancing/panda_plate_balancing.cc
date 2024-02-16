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

#include "mjpc/tasks/plate_balancing/panda_plate_balancing.h"

#include <string>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
std::string PlateBalancing::XmlPath() const {
  return GetModelPath("plate_balancing/task.xml");
}
std::string PlateBalancing::Name() const { return "PlateBalancing"; }

// ---------- Residuals for plate balancing manipulation task ---------
//   Number of residuals: 5
//     Residual (0): end effector - target
//     Residual (1): tray - target
//     Residual (3): tray orientation
// ------------------------------------------------------------
void PlateBalancing::ResidualFn::Residual(const mjModel* model, const mjData* data,
                     double* residual) const {
  int counter = 0;

  // reach
  double* end_effector = SensorByName(model, data, "end_effector");
  double* tray = SensorByName(model, data, "tray");
  double* tray_orientation = SensorByName(model, data, "tray_quat");
  double* target = SensorByName(model, data, "target");
  double* end_effector_target = SensorByName(model, data, "end_effector_target");
  double* neutral_orientation = SensorByName(model, data, "center_support");
  
  mju_sub3(residual + counter, end_effector, end_effector_target);
  counter += 3;
  mju_sub3(residual + counter, tray, target);
  counter += 3;
  mju_subQuat(residual + counter, tray_orientation, neutral_orientation);
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
    mju_error_i(
        "mismatch between total user-sensor dimension "
        "and actual length of residual %d",
        counter);
  }
}

void PlateBalancing::TransitionLocked(mjModel* model, mjData* data) {
  double residuals[100];
  residual_.Residual(model, data, residuals);
  double bring_dist = (mju_norm3(residuals+3)) / 2;

  // reset:
  
  if (data->time > 0 && bring_dist < .05) {
    if ((data->mocap_pos[0] == 0.45 && data->mocap_pos[2] == 0.485) || data->time < 10.0) {
      // target:
      data->mocap_pos[0] = 0.45;
      data->mocap_pos[1] = 0.0;
      data->mocap_pos[2] = 0.6;
      data->mocap_quat[0] = 1;
      data->mocap_quat[1] = 0;
      data->mocap_quat[2] = 0;
      data->mocap_quat[3] = 0;
      data->mocap_pos[3] = 0.45;
      data->mocap_pos[4] = 0.0;
      data->mocap_pos[5] = 0.584;
      data->mocap_quat[4] = 1;
      data->mocap_quat[5] = 0;
      data->mocap_quat[6] = 0;
      data->mocap_quat[7] = 0;
      mju_normalize4(data->mocap_quat);
    } else{
      data->mocap_pos[0] = 0.7;
      data->mocap_pos[1] = 0.0;
      data->mocap_pos[2] = 0.485;
      data->mocap_quat[0] = 1;
      data->mocap_quat[1] = 0;
      data->mocap_quat[2] = 0;
      data->mocap_quat[3] = 0;
      data->mocap_pos[3] = 0.6;
      data->mocap_pos[4] = 0.0;
      data->mocap_pos[5] = 0.469;
      data->mocap_quat[4] = 1;
      data->mocap_quat[5] = 0;
      data->mocap_quat[6] = 0;
      data->mocap_quat[7] = 0;
    }
    
  }

}
}  // namespace mjpc
