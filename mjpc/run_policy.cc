#include <cstdlib>
#include <cstring>
#include <iostream>
#include <ostream>
#include <vector>
#include <absl/flags/parse.h>
#include "mjpc/agent.h"
#include "mjpc/threadpool.h"
#include "mjpc/array_safety.h"
#include "mjpc/tasks/tasks.h"

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

  // Set up the task
  std::vector<std::shared_ptr<mjpc::Task>> tasks = mjpc::GetTasks();
  std::shared_ptr<mjpc::Task> task = tasks[12];
  mjModel* model = new mjModel();
  std::unique_ptr<mjpc::ResidualFn> residual_fn;

  // Load the model
  std::string filename = task->XmlPath();
  constexpr int kErrorLength = 1024;
  char load_error[kErrorLength] = "";
  model = mj_loadXML(filename.c_str(), nullptr, load_error,
                        kErrorLength);
  if (load_error[0]) {
    int error_length = mju::strlen_arr(load_error);
    if (load_error[error_length - 1] == '\n') {
        load_error[error_length - 1] = '\0';
    }
  }

  auto agent = std::make_shared<mjpc::Agent>(model, task);

  // Weird thing, where I have to define the sensor separate from the model??
  residual = agent->ActiveTask()->Residual();
  mjcb_sensor = sensor;
  
  std::vector<double> action = {-0.1, 0.1, 0};
  std::vector<double> qpos = {0.7, 0.00, 0.485, 1, 0, 0, 0, 0.55, 0.0, 0.45};
  std::vector<double> qvel = { 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<double> mocap_pos = {0.45, 0, 0.6, 0.45, 0, 0.584};
  std::vector<double> mocap_quat = {1, 0, 0, 0, 1, 0, 0, 0};
  std::vector<double> user_data = {0}; // size is actually 0

  std::cout << agent->ActiveTask()->Name() << std::endl;
  std::cout << "user data size " << model->nuserdata << std::endl;
  std::cout << "nmocap " << model->nmocap << std::endl;
  std::cout << "nq " << model->nq << std::endl;
  std::cout << "nv " << model->nv << std::endl;
  std::cout << "sensor dim " << model->sensor_dim[0] << std::endl;
  std::cout << "task residual " << agent->ActiveTask()->num_residual << std::endl;
  double time = 0;
  mjpc::State mpc_state = mjpc::State();
  mpc_state.Initialize(model);
  mpc_state.Allocate(model);
  mpc_state.Set(model, qpos.data(), qvel.data(), action.data(), mocap_pos.data(), mocap_quat.data(), user_data.data(), time);

  mjpc::ThreadPool plan_pool(agent->planner_threads());
  // mjpc::ThreadPool plan_pool(1);
  std::cout << "planning threads: " << agent->planner_threads() << std::endl;
  std::cout << "num parameters: " << agent->ActivePlanner().NumParameters() << std::endl;

  for (int i = 0; i < 5; ++i){
    residual_fn = agent->ActiveTask()->Residual();
    agent->ActivePlanner().SetState(mpc_state);
    agent->ActivePlanner().OptimizePolicy(5, plan_pool);
    residual_fn.reset();
  }
  
  auto trajectory = agent->ActivePlanner().BestTrajectory();

  std::cout << "horizon: " << trajectory->horizon << std::endl;
  std::cout << "dim_action: " << trajectory->dim_action << std::endl;
  std::cout << "dim_state: " << trajectory->dim_state << std::endl;

  // Seems like trajectory actions and states never update the size, always max length, so manually copy over the correct size
  std::cout << "actions size: " << trajectory->actions.size() << std::endl;
  std::cout << "states size: " << trajectory->states.size() << std::endl;
  std::cout << "times size: " << trajectory->times.size() << std::endl;
  std::cout << "total return: " << trajectory->total_return << std::endl;
  
  // only dim_action * horizon is non-zero
  // Note this is diff than the documentation that says dim_action * (horizon - 1)
  for (int i = 0; i < trajectory->dim_action * (trajectory->horizon); ++i){
    std::cout << trajectory->actions[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}