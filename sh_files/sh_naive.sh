#!/bin/bash

#SBATCH -A grp_gpedriel      # account to charge for the job one of grp_gpedriel or grp_spanias
#SBATCH -N 1                # number of nodes
#SBATCH --mem=16G            # amount of memory requested
#SBATCH -c 1                # number of cores
#SBATCH -G 1                # number of gpus
#SBATCH -t 0-04:00:00       # time in d-hh:mm:ss
#SBATCH -p htc              # partition general or htc
#SBATCH -q public           # QOS
#SBATCH -o jobs/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e jobs/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL      # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="%u@asu.edu"
#SBATCH --export=NONE        # Purge the job-submitting shell environment

#Load required software
module load mamba/latest

#Activate our enviornment
source activate CBO

#Change to the directory of our script
cd /home/mmalu/CL_HAR/code
# cd ~/home/mmalu/cbo_code_011525

#Run the software/python script
python modular_code/main_naive.py \
        --data /home/mmalu/CL_HAR/code/GSC_data/embeddings \
        --epochs_per_task 100 \
        --dataset 'GSC' \
        --model_type 'resnet18'


# --data /home/mmalu/CL_HAR/code/ESC-50-data/ESC-50-master/new_CLAP_embeddings \ --- IGNORE ---
# --data /home/mmalu/CL_HAR/code/GSC_data/embeddings \ --- IGNORE ---
# --model_type 'resnet18' \ --- IGNORE ---
# --model_type 'cnn2' \ --- IGNORE ---
# --dataset 'ESC-50' --- IGNORE ---
# --dataset 'GSC' --- IGNORE ---
