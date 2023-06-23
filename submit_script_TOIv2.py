#this script submits multiple scripts to HPC

import os

script = 'do_MCMC_TOI-776.py'
config_files = ['TOI766_massp_ENA', 'TOI766_massp_ENA2', 'TOI766_mlr_fixed_ENA', 'TOI766_mlr_fixed_ENA2']
config_path = ['TOI_776']
backend_files = ['../' + config_file + 'pbsv2.h5' for config_file in config_files]
flags = ['--ena', '--ena']

for file_index in range(len(config_files)):
    os.popen(f'qsub -F "do_MCMC_TOI-776.py \'-b {backend_files[file_index]} -c config_files.{config_path[0]}.{config_files[file_index]} {flags[file_index]}\'" submit_script_pbs')
