#!/bin/bash
#SBATCH --job-name=redq
#SBATCH --account=project_2003582
#SBATCH --output=redq_out_%A_%a.txt
#SBATCH --error=redq_err_%A_%a.txt
#SBATCH --time=72:00:00
#SBATCH --mem=60G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --array=0-7

case $SLURM_ARRAY_TASK_ID in
    0) ENV="acrobot_swingup";;
    1) ENV="cheetah_run" ;;
    2) ENV="fish_swim";;
    3) ENV="dog_run" ;;
    4) ENV="quadruped_walk" ;;
    5) ENV="walker_walk" ;;
    6) ENV="humanoid_walk" ;;
    7) ENV="dog_walk" ;;
esac

export PROJID=project_2003582
export PROJAPPL=/projappl/${PROJID}
export SCRATCH=/scratch/${PROJID}

export LC_ALL=en_US.UTF-8
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK # set the number of threads based on --cpus-per-task

srun python3 train.py env=$ENV
