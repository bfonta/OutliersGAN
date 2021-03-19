#!/usr/bin/env bash 
#
SLURM_JOBS_FOLDER="${HOME}/keras_env/SLURM_FREEZE"
if [ ! -d "${SLURM_JOBS_FOLDER}" ]; then
    mkdir "${SLURM_JOBS_FOLDER}"
fi
cp -r src/ "${SLURM_JOBS_FOLDER}" &&
    cp -r pre_processing/ "${SLURM_JOBS_FOLDER}" &&
    cp -r post_processing/ "${SLURM_JOBS_FOLDER}" &&
    cp *.py "${SLURM_JOBS_FOLDER}"

CHECKPOINT=999
NAME='qsoNoNorm'
sbatch --job-name="${NAME}" \
       --output=outputs/"${NAME}".out \
       --ntasks=1 \
       --cpus-per-task=1 \
       --ntasks-per-node=1 \
       --time=7-00:00:00 \
       --mem-per-cpu=16GB \
       --gres=gpu:1 \
       --mail-type=BEGIN,END,FAIL,TIME_LIMIT_50,TIME_LIMIT_80 \
       job_scripts/gpu_spectra.sh "${SLURM_JOBS_FOLDER}" "${CHECKPOINT}"
#
#name='writer'
#sbatch -J "${name}" --output outputs/"${name}".out --ntasks=1 --cpus-per-task=1 --ntasks-per-node=1 --time=40:00:00 --mem-per-cpu=1GB --gres=gpu:1 job_scripts/writer.sh
#
#
#######Checkpoints#############
#71: legacy data; extra loss term; grid; valid: 391672
#    name: wgangp_grid_newloss
#72: galaxy with STARFORMING and STARBURST plus zWarning=0; extra loss term; grid; valid: 321520
#    name: gal_zWarning
#73: qso plus zWarning=0; extra loss term; grid; valid: 22791
#    name: qso_zWarning
#74: qso plus zWarning=0; extra loss term; grid; valid: 
#    different wavelength range: 1800-4150 Angstroms
#    name: qso2_zWarning
