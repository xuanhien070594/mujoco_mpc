#include <cstdlib>
#include <cstring>
#include <iostream>
#include <ostream>
#include <vector>

#include <absl/flags/parse.h>
#include <lcm/lcm-cpp.hpp>

#include "dairlib/lcmt_c3_state.hpp"
#include "dairlib/lcmt_object_state.hpp"
#include "dairlib/lcmt_robot_output.hpp"
#include "dairlib/lcmt_timestamped_saved_traj.hpp"
#include "mjpc/agent.h"
#include "mjpc/array_safety.h"
#include "mjpc/tasks/tasks.h"
#include "mjpc/threadpool.h"

// #include "mjpc/task.h"
namespace mju = ::mujoco::util_mjpc;

std::unique_ptr<mjpc::ResidualFn> residual;

extern "C" {
void sensor(const mjModel* m, mjData* d, int stage);
}

// sensor callback
void sensor(const mjModel* model, mjData* data, int stage) {
  residual->Residual(model, data, data->sensordata);
}

class Handler {
 public:
  Handler() {
    franka_positions_ = std::vector<double>(3);
    franka_velocities_ = std::vector<double>(3);
    tray_positions_ = std::vector<double>(7);
    tray_velocities_ = std::vector<double>(6);
  }
  ~Handler() {}
  void handle_mpc_state(const lcm::ReceiveBuffer* rbuf, const std::string& chan,
                        const dairlib::lcmt_c3_state* msg) {
    time_ = 1e-6 * msg->utime;
    franka_positions_ = {msg->state.begin(), msg->state.begin() + 3};
    franka_velocities_ = {msg->state.begin() + 10, msg->state.begin() + 10 + 3};
    tray_positions_ = {msg->state.begin() + 3, msg->state.begin() + 3 + 7};
    // mujoco state goes pos then orientation, msg does orientation then pos
    tray_positions_[0] = msg->state[7];
    tray_positions_[1] = msg->state[8];
    tray_positions_[2] = msg->state[9];
    tray_positions_[3] = msg->state[3];
    tray_positions_[4] = msg->state[4];
    tray_positions_[5] = msg->state[5];
    tray_positions_[6] = msg->state[6];
    tray_velocities_ = {msg->state.begin() + 10 + 3,
                        msg->state.begin() + 10 + 3 + 6};
  }
  double time_;
  std::vector<double> franka_positions_;
  std::vector<double> franka_velocities_;
  std::vector<double> tray_positions_;
  std::vector<double> tray_velocities_;
};

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  // Set up the variables
  mjModel* model = new mjModel();
  std::vector<std::shared_ptr<mjpc::Task>> tasks = mjpc::GetTasks();
  std::shared_ptr<mjpc::Task> task = tasks[14];
  std::unique_ptr<mjpc::ResidualFn> residual_fn;

  // Load the model
  std::string filename = task->XmlPath();
  std::cout << task->XmlPath() << std::endl;
  constexpr int kErrorLength = 1024;
  char load_error[kErrorLength] = "";
  model = mj_loadXML(filename.c_str(), nullptr, load_error, kErrorLength);
  if (load_error[0]) {
    int error_length = mju::strlen_arr(load_error);
    if (load_error[error_length - 1] == '\n') {
      load_error[error_length - 1] = '\0';
    }
  }

  lcm::LCM lcm;
  if (!lcm.good()) return 1;

  // Initialize the agent
  auto agent = std::make_shared<mjpc::Agent>(model, task);

  // Weird thing, where I have to define the sensor separate from the model??
  residual = agent->ActiveTask()->Residual();
  // order matters, need to set the global variable as the sensor callback
  mjcb_sensor = sensor;

  std::vector<double> action = {-0.1, 0.1, 0};
  std::vector<double> qpos = {0.7, 0.00, 0.485, 1, 0, 0, 0, 0.55, 0.0, 0.45};
  std::vector<double> qvel = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<double> mocap_pos = {0.45, 0, 0.6, 0.45, 0, 0.584};
  std::vector<double> mocap_quat = {1, 0, 0, 0, 1, 0, 0, 0};
  std::vector<double> user_data = {0};  // size is actually 0

  std::cout << agent->ActiveTask()->Name() << std::endl;
  mjpc::ThreadPool plan_pool(agent->planner_threads());
  double time = 0;
  mjpc::State mpc_state = mjpc::State();
  mpc_state.Initialize(model);
  mpc_state.Allocate(model);

  dairlib::lcmt_robot_output robot_state;
  dairlib::lcmt_object_state tray_state;
  dairlib::lcmt_timestamped_saved_traj actor_traj;
  dairlib::lcmt_timestamped_saved_traj object_traj;
  dairlib::lcmt_saved_traj raw_actor_traj;
  dairlib::lcmt_saved_traj raw_object_traj;
  dairlib::lcmt_trajectory_block actor_pos_traj;
  dairlib::lcmt_trajectory_block actor_force_traj;
  dairlib::lcmt_trajectory_block object_pos_traj;
  dairlib::lcmt_trajectory_block object_quat_traj;
  int horizon = mjpc::GetNumberOrDefault(1.0e-2, model, "agent_horizon") /
                mjpc::GetNumberOrDefault(1.0e-2, model, "agent_timestep");
  horizon += 1;
  raw_actor_traj.trajectories = std::vector<dairlib::lcmt_trajectory_block>(2);
  raw_object_traj.trajectories = std::vector<dairlib::lcmt_trajectory_block>(2);
  actor_force_traj.num_points = horizon;
  actor_force_traj.num_datatypes = 3;
  actor_force_traj.datapoints =
      std::vector<std::vector<double>>(3, std::vector<double>(horizon, 0.0));
  actor_force_traj.time_vec = std::vector<double>(horizon);
  actor_force_traj.datatypes = std::vector<std::string>(3);
  actor_force_traj.trajectory_name = "end_effector_force_target";

  actor_pos_traj.num_points = horizon;
  actor_pos_traj.num_datatypes = 3;
  actor_pos_traj.datapoints =
      std::vector<std::vector<double>>(3, std::vector<double>(horizon));
  actor_pos_traj.time_vec = std::vector<double>(horizon);
  actor_pos_traj.datatypes = std::vector<std::string>(3);
  actor_pos_traj.trajectory_name = "end_effector_position_target";

  object_pos_traj.num_points = horizon;
  object_pos_traj.num_datatypes = 3;
  object_pos_traj.datapoints =
      std::vector<std::vector<double>>(3, std::vector<double>(horizon));
  object_pos_traj.time_vec = std::vector<double>(horizon);
  object_pos_traj.datatypes = std::vector<std::string>(3);
  object_pos_traj.trajectory_name = "object_position_target";

  object_quat_traj.num_points = horizon;
  object_quat_traj.num_datatypes = 4;
  object_quat_traj.datapoints =
      std::vector<std::vector<double>>(4, std::vector<double>(horizon));
  object_quat_traj.time_vec = std::vector<double>(horizon);
  object_quat_traj.datatypes = std::vector<std::string>(4);
  object_quat_traj.trajectory_name = "object_orientation_target";

  raw_actor_traj.trajectory_names = {actor_force_traj.trajectory_name,
                                     actor_pos_traj.trajectory_name};
  raw_actor_traj.num_trajectories = 2;
  raw_object_traj.trajectory_names = {object_pos_traj.trajectory_name,
                                      object_quat_traj.trajectory_name};
  raw_object_traj.num_trajectories = 2;

  auto planner_id = mjpc::GetNumberOrDefault(0, model, "agent_planner");

  std::cout << "planner id: " << planner_id << std::endl;
  std::cout << "horizon: " << agent->ActivePlanner().BestTrajectory()->horizon
            << std::endl;

  agent->GetModel()->opt.timestep =
      mjpc::GetNumberOrDefault(1.0e-2, model, "agent_timestep");
  Handler handlerObject;
  lcm.subscribe("C3_ACTUAL", &Handler::handle_mpc_state, &handlerObject);
  int actor_pos_start = 7;
  int object_pos_start = 0;
  int object_quat_start = 3;

  std::cout << "planning threads: " << agent->planner_threads() << std::endl;
  std::cout << "num parameters: " << agent->ActivePlanner().NumParameters()
            << std::endl;

  while (true) {
    if (lcm.getFileno() != 0) {
      lcm.handle();
    }
    time = handlerObject.time_;
    // the order of the positions are different between C3 and mjmpc
    mju_copy(qpos.data(), handlerObject.tray_positions_.data(), 7);
    mju_copy(qpos.data() + 7, handlerObject.franka_positions_.data(), 3);
    mju_copy(qvel.data(), handlerObject.tray_velocities_.data(), 6);
    mju_copy(qvel.data() + 6, handlerObject.franka_velocities_.data(), 3);

    action[0] = actor_force_traj.datapoints[0][0];
    action[1] = actor_force_traj.datapoints[1][0];
    action[2] = actor_force_traj.datapoints[2][0];

    mpc_state.Set(model, qpos.data(), qvel.data(), action.data(),
                  mocap_pos.data(), mocap_quat.data(), user_data.data(), time);
    agent->ActivePlanner().SetState(mpc_state);
    agent->ActivePlanner().OptimizePolicy(horizon, plan_pool);
    auto trajectory = agent->ActivePlanner().BestTrajectory();
    actor_force_traj.time_vec = trajectory->times;

    // Scaling the action
    for (int k = 0; k < trajectory->horizon; ++k) {
      actor_force_traj.datapoints[0][k] =
          10 * trajectory->actions[0 + k * trajectory->dim_action];
      actor_force_traj.datapoints[1][k] =
          10 * trajectory->actions[1 + k * trajectory->dim_action];
      actor_force_traj.datapoints[2][k] =
          30 * trajectory->actions[2 + k * trajectory->dim_action];
    }
    actor_pos_traj.time_vec = trajectory->times;
    for (int k = 0; k < trajectory->horizon; ++k) {
      for (int n = 0; n < actor_pos_traj.datatypes.size(); ++n) {
        actor_pos_traj.datapoints[n][k] =
            trajectory
                ->states[(n + actor_pos_start) + k * trajectory->dim_state];
      }
    }
    object_pos_traj.time_vec = trajectory->times;
    for (int k = 0; k < trajectory->horizon; ++k) {
      for (int n = 0; n < object_pos_traj.datatypes.size(); ++n) {
        object_pos_traj.datapoints[n][k] =
            trajectory
                ->states[(n + object_pos_start) + k * trajectory->dim_state];
      }
    }
    object_quat_traj.time_vec = trajectory->times;
    for (int k = 0; k < trajectory->horizon; ++k) {
      for (int n = 0; n < object_quat_traj.datatypes.size(); ++n) {
        object_quat_traj.datapoints[n][k] =
            trajectory
                ->states[(n + object_quat_start) + k * trajectory->dim_state];
      }
    }

    raw_actor_traj.trajectories.at(0) = actor_force_traj;
    raw_actor_traj.trajectories.at(1) = actor_pos_traj;
    raw_object_traj.trajectories.at(0) = object_pos_traj;
    raw_object_traj.trajectories.at(1) = object_quat_traj;
    actor_traj.saved_traj = raw_actor_traj;
    actor_traj.utime = time * 1e6;
    object_traj.saved_traj = raw_object_traj;
    object_traj.utime = time * 1e6;
    lcm.publish("C3_TRAJECTORY_ACTOR", &actor_traj);
    lcm.publish("C3_TRAJECTORY_TRAY", &object_traj);
  }

  return 0;
}