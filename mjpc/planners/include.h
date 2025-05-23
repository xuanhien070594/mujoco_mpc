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

#ifndef MJPC_PLANNERS_INCLUDE_H_
#define MJPC_PLANNERS_INCLUDE_H_

#include <memory>
#include <vector>

#include "mjpc/planners/planner.h"

namespace mjpc {

// planner types
enum PlannerType : int {
  kSamplingPlanner = 0,
  kGradientPlanner,
  kILQGPlanner,
  kILQSPlanner,
  kRobustPlanner,
  kCrossEntropyPlanner,
  kSampleGradientPlanner,
};

// Planner names, separated by '\n'.
extern const char kPlannerNames[];

// Loads all available planners
std::vector<std::unique_ptr<mjpc::Planner>> LoadPlanners();

}  // namespace mjpc

#endif  // MJPC_PLANNERS_INCLUDE_H_
