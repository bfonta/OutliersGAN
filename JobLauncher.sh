#!/usr/bin/env bash 
#
mnist_checkpoint=1
name='mnist'
#sbatch -J "${name}" --output outputs/mnist.out --ntasks=1 --cpus-per-task=1 --ntasks-per-node=1 --time=100:00:00 --mem-per-cpu=16GB --gres=gpu:1 job_scripts/gpu_mnist.sh "${mnist_checkpoint}"
#
fmnist_checkpoint=10
name='fmnist_v2'
#sbatch -J "${name}" --output outputs/"${name}".out --ntasks=1 --cpus-per-task=1 --ntasks-per-node=1 --time=100:00:00 --mem-per-cpu=16GB --gres=gpu:1 job_scripts/gpu_fmnist.sh "${fmnist_checkpoint}"
#
cifar10_checkpoint=3
#name='cifar10'
#sbatch -J cifar10 --output outputs/cifar10.out --ntasks=1 --cpus-per-task=1 --ntasks-per-node=1 --time=100:00:00 --mem-per-cpu=16GB --gres=gpu:1 job_scripts/gpu_cifar10.sh "${cifar10_checkpoint}"
#
spectra_checkpoint=900
name='qso2NoTanh'
sbatch -J "${name}" --output outputs/"${name}".out --ntasks=1 --cpus-per-task=1 --ntasks-per-node=1 --time=0-10:00:00 --mem-per-cpu=8GB --gres=gpu:2 job_scripts/gpu_spectra.sh "${spectra_checkpoint}"
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
