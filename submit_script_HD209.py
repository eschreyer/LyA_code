#this script submits multiple scripts to HPC

import os

script = 'do_MCMC_TOI-766.py'
config_files = ['HD209']
config_path = ['HD209']
backend_files = ['../rds/rds-dirac-dp100-n0fcCTMMDq4/dc-schr2/' + config_file for config_file in config_files]
flags = ['']

for file_index in range(len(config_files)):
    os.popen(f'sbatch -J ethan_script all_submit_script "{script} -r -b {backend_files[file_index]} -c config_files.{config_path[0]}.{config_files[file_index]} -f b {flags[file_index]}"')
