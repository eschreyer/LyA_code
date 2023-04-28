#this script submits multiple scripts to HPC

import os

script = 'do_MCMC_gj436b_new.py'
config_files = ['config', 'config2', 'config_sonic_cuts', 'config_sonic_cuts2', 'config_var1', 'config_var2', 'config_hill', 'config_hill2', 'configENA1', 'configENA2', 'configENAhill1', 'configENAhill2']
config_path = ['tail', 'tail', 'tail', 'tail', 'tail', 'tail', 'tail_hill', 'tail_hill', 'tail_ena', 'tail_ena', 'tail_hill_ena', 'tail_hill_ena']
backend_files = ['../rds/rds-dirac-dp100-n0fcCTMMDq4/dc-schr2/' + config_file + 'real.h5' for config_file in config_files]
flags = ['', '', '', '', '', '', '--hill_sphere', '--hill_sphere', '--ena', '--ena', '--hill_sphere --ena', '--hill_sphere --ena']

for file_index in range(len(config_files)):
    os.popen(f'sbatch -J ethan_script all_submit_script "{script} -b {backend_files[file_index]} -c config_files.{config_path[file_index]}.{config_files[file_index]} -f b {flags[file_index]}"')
