#this script submits multiple scripts to HPC

import os

script = 'do_MCMC_gj436b_new.py'
config_files = ['config_extendedp', 'config_extendedp2', 'config_hill', 'config_hill2', 'configENAhill1', 'configENAhill2']
config_path = ['tail', 'tail', 'tail_hill', 'tail_hill', 'tail_hill_ena', 'tail_hill_ena']
backend_files = ['../rds/rds-dirac-dp100-n0fcCTMMDq4/dc-schr2/' + config_file + 'real2.h5' for config_file in config_files]
flags = ['', '', '--hill_sphere', '--hill_sphere', '--hill_sphere --ena', '--hill_sphere --ena']

for file_index in range(len(config_files)):
    os.popen(f'sbatch -J ethan_script all_submit_script "{script} -b {backend_files[file_index]} -c config_files.{config_path[file_index]}.{config_files[file_index]} -f b {flags[file_index]}"')
