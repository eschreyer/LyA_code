#this script submits multiple scripts to HPC

import os

script = 'do_MCMC_gj436b_new.py'
config_files = ['config', 'config2', 'config2xSWTemp1', 'config2xSWTemp2', 'configTtail1', 'configTtail2', 'configENA1', 'configENA2']
backend_files = ['../rds/rds-dirac-dp100-n0fcCTMMDq4/dc-schr2/' + config_file + '.h5' for config_file in config_files]
flags = ['', '', '', '','', '', '--ena', '--ena']

for file_index in range(len(config_files)):
    os.popen(f'sbatch -J ethan_script all_submit_script "{script} -r -b {backend_files[file_index]} -c config_files.{config_files[file_index]} -f b {flags[file_index]}"')
