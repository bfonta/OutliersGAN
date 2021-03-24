#!/bin/bash
#
#SBATCH --job-name=qsoNoNorm
#SBATCH --output=outputs/qsoNoNorm.out \
#SBATCH --ntasks=1 \
#SBATCH --cpus-per-task=1 \
#SBATCH --ntasks-per-node=1 \
#SBATCH --time=0-20:00:00 \
#SBATCH --mem-per-cpu=8GB \
#SBATCH --gres=gpu:1 \
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 \

module load git/2.18.0
module load gcc/7.3.0
module load python/3.6.4
module load numpy/1.14.2-python-3.6.4
module load matplotlib/2.2.2-python-3.6.4
module load h5py/2.7.1-python-3.6.4
module load pandas/0.22.0-python-3.6.4
module load scipy/1.0.1-python-3.6.4
module load astropy/2.0.3-python-3.6.4
module load cuda/9.2.88
module load tensorflowgpu/1.12.0-python-3.6.4

###export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
. /home/balves/keras_env/bin/activate

python spectra_dcgan.py --checkpoint 999 --mode train
