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
  dairlib::lcmt_saved_traj raw_traj;
  dairlib::lcmt_trajectory_block actor_pos_traj;
  dairlib::lcmt_trajectory_block actor_force_traj;
  actor_force_traj.num_points = 5;
  actor_force_traj.num_datatypes = 3;
  actor_force_traj.datapoints =
      std::vector<std::vector<double>>(3, std::vector<double>(5));
  actor_force_traj.time_vec = std::vector<double>(5);
  actor_force_traj.datatypes = std::vector<std::string>(3);
  actor_force_traj.trajectory_name = "actor_force_traj";

  // dairlib_lcmt_object_state tray_state;
  // dairlib_lcmt_timestamped_saved_traj actor_traj;

  for (int i = 0; i < 5; ++i) {
    // robot_state.position
    // mju_copy3(qpos.data(), robot_state.position.data());
    mpc_state.Set(model, qpos.data(), qvel.data(), action.data(),
                  mocap_pos.data(), mocap_quat.data(), user_data.data(), time);
    agent->ActivePlanner().SetState(mpc_state);
    agent->ActivePlanner().OptimizePolicy(5, plan_pool);
    auto trajectory = agent->ActivePlanner().BestTrajectory();
    std::cout << "total return: " << trajectory->total_return << std::endl;
    // actor_force_traj.time_vec = trajectory->times;
    lcm.publish("MJPC_TRAJECTORY_ACTOR", &actor_force_traj);
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