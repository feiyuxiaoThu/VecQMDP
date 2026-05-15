import os
import argparse
from datetime import datetime

import hydra
from nuplan.planning.script.run_simulation import run_simulation as main_simulation
from tutorials.utils.tutorial_utils import construct_simulation_hydra_paths

HOME = os.path.expanduser('~')

VALID_SPLITS = ['val14_split', 'test_hard', 'test14_random']
VALID_CHALLENGES = ['closed_loop_nonreactive_agents', 'closed_loop_reactive_agents']
VALID_WORKERS = ['ray_distributed', 'sequential']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run Vec-QMDP closed-loop simulation evaluation on nuPlan.'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='val14_split',
        choices=VALID_SPLITS,
        help=(
            'Scenario split to evaluate on. '
            'val14_split: 14-scenario validation set; '
            'test_hard: hard test scenarios; '
            'test14_random: 14 randomly sampled test scenarios. '
            'Default: val14_split.'
        ),
    )
    parser.add_argument(
        '--challenge',
        type=str,
        default='closed_loop_nonreactive_agents',
        choices=VALID_CHALLENGES,
        help=(
            'Challenge type (simulation mode). '
            'closed_loop_nonreactive_agents: other agents follow fixed log replay; '
            'closed_loop_reactive_agents: other agents react to the ego vehicle. '
            'Default: closed_loop_nonreactive_agents.'
        ),
    )
    parser.add_argument(
        '--worker',
        type=str,
        default='ray_distributed',
        choices=VALID_WORKERS,
        help=(
            'Worker backend for parallel simulation. '
            'ray_distributed: parallel execution via Ray (recommended for large splits); '
            'sequential: single-process sequential execution (useful for debugging). '
            'Default: ray_distributed.'
        ),
    )
    return parser.parse_args()


def generate_timestamp():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def main():
    args = parse_args()
    split = args.split
    challenge = args.challenge
    worker = args.worker

    BASE_CONFIG_PATH = os.path.join(
        os.getenv('NUPLAN_TUTORIAL_PATH', ''),
        '../../../nuplan-devkit/nuplan/planning/script',
    )
    simulation_hydra_paths = construct_simulation_hydra_paths(BASE_CONFIG_PATH)

    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=simulation_hydra_paths.config_path)

    timestamp = generate_timestamp()

    # Compose the configuration
    cfg = hydra.compose(
        config_name=simulation_hydra_paths.config_name,
        overrides=[
            f'experiment_name={challenge}',
            f'job_name={challenge}',
            'experiment=${experiment_name}/${job_name}',
            f'worker={worker}',
            'planner=vec_qmdp_planner',
            f'scenario_filter={split}',
            'scenario_builder=nuplan',
            (
                f'hydra.searchpath=['
                f'file://{HOME}/VecQMDP/python_planner/scripts/config/common,'
                f'file://{HOME}/VecQMDP/python_planner/scripts/config/simulation,'
                f'pkg://nuplan.planning.script.config.common,'
                f'pkg://nuplan.planning.script.experiments'
                f']'
            ),
            f'output_dir={HOME}/nuplan/exp/simulation/{challenge}/{timestamp}',
            f'+simulation={challenge}',
            'number_of_gpus_allocated_per_simulation=1',
            'number_of_cpus_allocated_per_simulation=1',
        ],
    )

    # Run the simulation loop
    main_simulation(cfg)


if __name__ == '__main__':
    main()
