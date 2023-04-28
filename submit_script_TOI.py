#this script submits multiple scripts to HPC

import os

script = 'do_MCMC_TOI-766.py'
config_files = ['TOI766', 'TOI7662', 'TOI766_diff_height', 'TOI766_diff_height2', 'TOI766_sonic_cuts', 'TOI766_sonic_cuts2', 'TOI766_ENA', 'TOI766_ENA2']
config_path = ['TOI_776']
backend_files = ['../rds/rds-dirac-dp100-n0fcCTMMDq4/dc-schr2/' + config_file + 'real.h5' for config_file in config_files]
flags = ['', '', '', '', '', '', '--ena', '--ena']

for file_index in range(len(config_files)):
    os.popen(f'sbatch -J ethan_script all_submit_script "{script} -r -b {backend_files[file_index]} -c config_files.{config_path[0]}.{config_files[file_index]} -f b {flags[file_index]}"')
