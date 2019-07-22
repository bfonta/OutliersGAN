#!/usr/bin/env bash
ls origdata/*fits | awk -f shuffle.awk | shuf -o shuffled_names.csv
bash hide_names.sh
