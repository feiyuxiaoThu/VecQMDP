import os
from pathlib import Path
import hydra
import argparse

from nuplan.planning.script.run_nuboard import main as main_nuboard
from tutorials.utils.tutorial_utils import construct_nuboard_hydra_paths

HOME = os.path.expanduser('~')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run nuboard.'
    )
    parser.add_argument(
        '--log_path',
        type=str,
        default='',
        help=(
            'Scenario log_path to evaluate on. '
        ),
    )
    return parser.parse_args()

def main():
    args = parse_args()
    log_path = args.log_path
    if log_path == '':
        raise ValueError('Please provide a log_path to run nuboard on.')

    simulation_file = [str(file) for file
                        in Path(str(log_path)).iterdir() if file.is_file() and file.suffix == '.nuboard']

    BASE_CONFIG_PATH = os.path.join(os.getenv('NUPLAN_TUTORIAL_PATH', ''), '../../../nuplan-devkit/nuplan/planning/script')
    nuboard_hydra_paths = construct_nuboard_hydra_paths(BASE_CONFIG_PATH)

    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=nuboard_hydra_paths.config_path)

    # Compose the configuration
    cfg = hydra.compose(config_name=nuboard_hydra_paths.config_name, overrides=[
        'scenario_builder=nuplan',
        'port_number=5600',
        f'simulation_path={simulation_file}',  # nuboard file path, if left empty the user can open the file inside nuBoard
        f'hydra.searchpath=[file:///{HOME}/nuplan-devkit/nuplan/planning/script/config/common, {nuboard_hydra_paths.common_dir}, {nuboard_hydra_paths.experiment_dir}]',
    ])

    main_nuboard(cfg)

if __name__ == '__main__':
    main()