#!/usr/bin/env bash
##the user must provide 2 arguments: 1: folder number 2) intermediate number
##example: bash spectra_download.sh 3590 55201

#rsync --no-motd --partial-dir=.rsync-partial -aLv --include "spec-*.fits" --exclude "*" --prune-empty-dirs --progress rsync://data.sdss.org/dr12/boss/spectro/redux/v5_7_0/spectra/3[0-8][0-9][0-9]/ /fred/oz012/Bruno/data/spectra/

rsync -aLzv --files-from=/fred/oz012/Bruno/data/gal_starforming_starburst_zWarning_List.txt --log-file "log2.log" --progress rsync://data.sdss3.org/dr15/ /fred/oz012/Bruno/data/spectra/gal_starforming_starburst_zWarning/
