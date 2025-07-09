#!/bin/bash
#SBATCH --job-name=Detectron2
#SBATCH --partition=gpu              # on the partition "gpu"
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1                   # with a single task (this should always be 1, apart from special cases)
#SBATCH --cpus-per-task=64            # with that many cpu cores
#SBATCH --mem=32G                    # will require that amount of RAM at maximum (if the process takes more it gets killed)
#SBATCH --time=4-00:00               # maximum runtime of the job as "d-hh:mm"
#SBATCH --chdir=/work/akinfalabi/machine_learning/hypocotyl_segment/Detectron2   # working directory of the job
#SBATCH --mail-type=ALL              # always get mail notifications
#SBATCH --output=/work/akinfalabi/machine_learning/hypocotyl_segment/Detectron2/slurm-%j.out        # standard output of the job into this file (also stderr)

module load lang/Anaconda3/2024.02-1
source activate detectron2

python3 scripts/detectron_script.py
