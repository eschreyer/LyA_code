#this script submits multiple scripts to HPC

import os

script = 'do_MCMC_TOI-776.py'
config_files = ['TOI766_O_var', 'TOI766_O_stoich2']
config_path = ['TOI_776']
backend_files = ['../rds/rds-dirac-dp100-n0fcCTMMDq4/dc-schr2/' + config_file + 'real.h5' for config_file in config_files]
flags = ['', '']

for file_index in range(len(config_files)):
    os.popen(f'sbatch -J ethan_script all_submit_script "{script} -b {backend_files[file_index]} -c config_files.{config_path[0]}.{config_files[file_index]} {flags[file_index]}"')
