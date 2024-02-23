#include <cstdlib>
#include <cstring>
#include <iostream>
#include <ostream>
#include <vector>

#include <absl/flags/parse.h>
#include <lcm/lcm-cpp.hpp>

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
    franka_positions_ = std::vector<double>(7);
    franka_velocities_ = std::vector<double>(7);
    tray_positions_ = std::vector<double>(7);
    tray_velocities_ = std::vector<double>(6);
  }
  ~Handler() {}
  void handle_franka(const lcm::ReceiveBuffer* rbuf, const std::string& chan,
                     const dairlib::lcmt_robot_output* msg) {
    franka_positions_ = msg->position;
    franka_velocities_ = msg->velocity;
    time_ = 1e-6 * msg->utime;
  }
  void handle_tray(const lcm::ReceiveBuffer* rbuf, const std::string& chan,
                   const dairlib::lcmt_object_state* msg) {
    tray_positions_ = msg->position;
    tray_velocities_ = msg->velocity;
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
  std::shared_ptr<mjpc::Task> task = tasks[12];
  std::unique_ptr<mjpc::ResidualFn> residual_fn;

  // Load the model
  std::string filename = task->XmlPath();
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

  std::cout << "planning threads: " << agent->planner_threads() << std::endl;
  std::cout << "num parameters: " << agent->ActivePlanner().NumParameters()
            << std::endl;

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
  int horizon =
      mjpc::GetNumberOrDefault(1.0e-2, model, "sampling_spline_points");
  raw_actor_traj.trajectories = std::vector<dairlib::lcmt_trajectory_block>(2);
  raw_object_traj.trajectories = std::vector<dairlib::lcmt_trajectory_block>(2);
  actor_force_traj.num_points = horizon;
  actor_force_traj.num_datatypes = 3;
  actor_force_traj.datapoints =
      std::vector<std::vector<double>>(3, std::vector<double>(horizon));
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
                                     actor_force_traj.trajectory_name};
  raw_actor_traj.num_trajectories = 2;
  raw_object_traj.trajectory_names = {object_pos_traj.trajectory_name,
                                      object_quat_traj.trajectory_name};
  raw_object_traj.num_trajectories = 2;

  // dairlib_lcmt_object_state tray_state;
  // dairlib_lcmt_timestamped_saved_traj actor_traj;
  agent->GetModel()->opt.timestep =
      mjpc::GetNumberOrDefault(1.0e-2, model, "agent_timestep");
  Handler handlerObject;
  lcm.subscribe("FRANKA_STATE_SIMULATION", &Handler::handle_franka,
                &handlerObject);
  lcm.subscribe("TRAY_STATE_SIMULATION", &Handler::handle_tray, &handlerObject);
  int actor_pos_start = 7;
  int object_pos_start = 4;
  int object_quat_start = 0;

  while (true) {
    if (lcm.getFileno() != 0) {
      lcm.handle();
    }
    time = handlerObject.time_;
    // the order of the positions are different between C3 and mjmpc
    mju_copy(qpos.data(), handlerObject.tray_positions_.data(), 7);

    mpc_state.Set(model, qpos.data(), qvel.data(), action.data(),
                  mocap_pos.data(), mocap_quat.data(), user_data.data(), time);
    agent->ActivePlanner().SetState(mpc_state);
    agent->ActivePlanner().OptimizePolicy(5, plan_pool);
    auto trajectory = agent->ActivePlanner().BestTrajectory();
    std::cout << "total return: " << trajectory->total_return << std::endl;
    actor_force_traj.time_vec = trajectory->times;
    for (int k = 0; k < trajectory->horizon; ++k) {
      for (int m = 0; m < trajectory->dim_action; ++m) {
        actor_force_traj.datapoints[m][k] =
            trajectory->actions[m + k * trajectory->dim_action];
      }
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
    lcm.publish("MJPC_TRAJECTORY_ACTOR", &actor_traj);
    lcm.publish("MJPC_TRAJECTORY_TRAY", &object_traj);
    // lcmt_robot_output_publish(lcm, "CASSIE_STATE_SIMULATION", &actor_traj);
  }

  // Seems like trajectory actions and states never update the size, always max
  // length, so manually copy over the correct size only dim_action * horizon is
  // non-zero Note this is diff than the documentation that says dim_action *
  // (horizon - 1) for (int i = 0; i < trajectory->dim_action *
  // (trajectory->horizon); ++i){
  //   std::cout << trajectory->actions[i] << " ";
  // }
  // std::cout << std::endl;
  // for (int i = 0; i < trajectory->dim_state * (trajectory->horizon); ++i){
  //   std::cout << trajectory->states[i] << " ";
  // }
  // std::cout << std::endl;

  return 0;
}