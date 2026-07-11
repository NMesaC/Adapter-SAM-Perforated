################################################################################
# Author:                                                                      #
# Nicholas Mesa-Cucalon (nicholas@perforatedai.com)                            #
#                                                                              #
# Training wrapper to support Perforation and DDP.                             #
################################################################################

#
"""
Imports
"""
import os
import sys
import time
import argparse
import subprocess

from typing import List, Optional

#
"""
Helpers
"""
def derive_save_path(config: str, name: str, tag: str) -> str:
    '''
    Compute train.py's save directory from the same config/name/tag inputs
    it uses, so this wrapper can find the checkpoints train.py writes

    Signature:
        config (str):
            - Path to the training yaml config
        name (str):
            - Explicit save name override, empty string if unset
        tag (str):
            - Suffix appended to the save name, empty string if unset
    '''
    if name:
        save_name = name
    else:
        config_basename = os.path.splitext(os.path.basename(config))[0]
        save_name        = f'_{config_basename}'
    if tag:
        save_name = f'{save_name}_{tag}'
    return os.path.join('./save', save_name)

def build_train_command(
    config         : str,
    name           : str,
    tag            : str,
    num_gpus       : int,
    pai_load_folder: Optional[str],
) -> List[str]:
    '''
    Build the torchrun command line used to launch train.py

    Signature:
        config (str):
            - Path to the training yaml config
        name (str):
            - Explicit save name override, empty string if unset
        tag (str):
            - Suffix appended to the save name, empty string if unset
        num_gpus (int):
            - Number of processes torchrun should launch
        pai_load_folder (Optional[str]):
            - Folder to resume PerforatedAI dendrite state from, None to
              start fresh
    '''
    command = [
        'torchrun',
        f'--nproc_per_node={num_gpus}',
        'train.py',
        '--config', config,
    ]
    if pai_load_folder is not None:
        command += ['--pai_load_folder', pai_load_folder]
    if name:
        command += ['--name', name]
    if tag:
        command += ['--tag', tag]
    return command

#
"""
Main
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Restart-on-restructure wrapper for PAI + DDP training'
    )
    parser.add_argument(
        '--config',
        type    = str,
        default = 'configs/drive-sam-vit-b.yaml',
        help    = 'Training yaml config to pass through to train.py',
    )
    parser.add_argument(
        '--name',
        type    = str,
        default = '',
        help    = 'Save name override, mirrors train.py --name',
    )
    parser.add_argument(
        '--tag',
        type    = str,
        default = '',
        help    = 'Save name suffix, mirrors train.py --tag',
    )
    parser.add_argument(
        '--num_gpus',
        type    = int,
        default = int(os.environ.get('NUM_GPUS', '1')),
        help    = 'Number of processes to pass to torchrun --nproc_per_node',
    )
    args = parser.parse_args()

    save_path = derive_save_path(args.config, args.name, args.tag)
    print(f'Config:    {args.config}')
    print(f'Save path: {save_path}')
    print(f'GPUs:      {args.num_gpus}')
    print('')

    while True:
        # Stop once train.py has written the completion marker
        complete_marker = os.path.join(save_path, '.training_complete')
        if os.path.exists(complete_marker):
            print('Training already completed.')
            break

        # Resume from the tracker's own auto-saved checkpoint if one exists
        latest_checkpoint = os.path.join(save_path, 'latest.pt')
        if os.path.exists(latest_checkpoint):
            print(f'Resuming from {latest_checkpoint} ...')
            pai_load_folder = save_path
        else:
            print('Starting training from scratch...')
            pai_load_folder = None

        command = build_train_command(
            args.config, args.name, args.tag, args.num_gpus, pai_load_folder
        )
        exit_code = subprocess.run(command).returncode

        if os.path.exists(complete_marker):
            print('Training completed successfully.')
            break

        # A clean exit with no completion marker means dendrites restructured
        if exit_code == 0:
            print('Dendrites restructured. Restarting in 2 seconds...')
            time.sleep(2)
        else:
            print(f'train.py exited with error code {exit_code}.')
            sys.exit(exit_code)
