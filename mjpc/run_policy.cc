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


int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  // Set up the task
  std::vector<std::shared_ptr<mjpc::Task>> tasks = mjpc::GetTasks();
  std::shared_ptr<mjpc::Task> task = tasks[12];
  // int horizon = task->parameters

  mjModel* model = new mjModel();


  // Load the model
  std::string filename = task->XmlPath();
  constexpr int kErrorLength = 1024;
  char load_error[kErrorLength] = "";
  model = mj_loadXML(filename.c_str(), nullptr, load_error,
                        kErrorLength);
  // remove trailing newline character from load_error
  if (load_error[0]) {
    int error_length = mju::strlen_arr(load_error);
    if (load_error[error_length - 1] == '\n') {
        load_error[error_length - 1] = '\0';
    }
  }


  mjpc::ThreadPool plan_pool(10);
  // Create the agent (TODO):multithreading
  auto agent = std::make_shared<mjpc::Agent>(model, task);


  // agent.PlanIteration()
  std::vector<double> action = {-0.1, 0.1, 0};
  std::vector<double> state = {0.7, 0.00, 0.485, 1, 0, 0, 0, 0.55, 0.0, 0.45, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  // double* action = new double[3] {0, 0, 0}; // n_u = 3
  // double* state = new double[19] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // n_x = n_q + n_v = 7 + 3 + 6 + 3 = 19
  double time = 0;
  
  for (int i = 0; i < 5; ++i){
    agent->ActivePlanner().OptimizePolicy(5, plan_pool);
    agent->ActivePlanner().NominalTrajectory(5, plan_pool);
  }
  agent->ActivePlanner().ActionFromPolicy(action.data(), state.data(), time);
  
  auto trajectory = agent->ActivePlanner().BestTrajectory();

  std::cout << "horizon: " << trajectory->horizon << std::endl;
  std::cout << "dim_action: " << trajectory->dim_action << std::endl;;
  std::cout << "dim_state: " << trajectory->dim_state << std::endl;;
  std::cout << "actions size: " << trajectory->actions.size() << std::endl;;
  std::cout << "states size: " << trajectory->states.size() << std::endl;;
  

  // for (auto& val : action){
  // for (const auto& val : trajectory->actions){
  //   std::cout << val << " ";
  // }

  return 0;
}