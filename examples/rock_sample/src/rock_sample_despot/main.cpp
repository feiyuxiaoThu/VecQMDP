#include <despot/planner.h>
#include "rock_sample_despot.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace despot;

class MyPlanner: public Planner {
  int rock_bits_; // -1 = random, >= 0 = fixed good/bad bitmask
public:
  explicit MyPlanner(int rock_bits = -1) : rock_bits_(rock_bits) {}

  DSPOMDP* InitializeModel(option::Option* options) {
    RockSample* model = NULL;
		if (options[E_PARAMS_FILE]) {
			model = new RockSample(options[E_PARAMS_FILE].arg);
		} else {
			int size = 7, number = 8;
			if (options[E_SIZE])
				size = atoi(options[E_SIZE].arg);
			else {
				cerr << "Specify map size using --size option" << endl;
				exit(0);
			}
			if (options[E_NUMBER]) {
				number = atoi(options[E_NUMBER].arg);
			} else {
				cerr << "Specify number of rocks using --number option" << endl;
				exit(0);
			}
			model = new RockSample(size, number);
		}
		if (rock_bits_ >= 0) {
			model->SetStartRockBits(rock_bits_);
			cout << "[rock_sample_despot] Fixed rock bits: " << rock_bits_
			     << " (0b";
			for (int i = model->GetStartRockBits(), w = 8; w > 0; --w)
				cout << ((i >> (w-1)) & 1);
			cout << ")" << endl;
		}
    return model;
  }

  World* InitializeWorld(std::string&  world_type, DSPOMDP* model, option::Option* options)
  {
      return InitializePOMDPWorld(world_type, model, options);
  }

  void InitializeDefaultParameters() {
  }

  std::string ChooseSolver(){
	  return "DESPOT";
  }
};

int main(int argc, char* argv[]) {
  // Parse and strip --rock-bits <N> before handing argv to the DESPOT framework
  // (the DESPOT option parser does not know this flag).
  int rock_bits = -1;
  std::vector<char*> fwd_argv;
  for (int i = 0; i < argc; ++i) {
    if (std::string(argv[i]) == "--rock-bits" && i + 1 < argc) {
      rock_bits = std::atoi(argv[++i]);
    } else {
      fwd_argv.push_back(argv[i]);
    }
  }
  int fwd_argc = static_cast<int>(fwd_argv.size());
  return MyPlanner(rock_bits).RunEvaluation(fwd_argc, fwd_argv.data());
}

// time_per_move(1),
// sim_len(90),
// num_scenarios(500),
// search_depth(90),
// max_policy_sim_len(90),
// discount(0.95),
// pruning_constant(0),
// xi(0.95),
// root_seed(42),
// default_action(""),