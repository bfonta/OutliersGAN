#!/usr/bin/env bash
#Arguments: 1: data_type [qso|gal] str
#           2: nbatches int
mkdir quiz/shufdata_"$1"/;
FITSPERBATCH=$(($2-1))
for i in $(seq 0 "$FITSPERBATCH"); do
    mkdir quiz/shufdata_"$1"/batch"$i"/;
    ls quiz/origdata_"$1"/batch"$i"/*.fits | awk -f quiz/shuffle.awk | shuf -o quiz/origdata_"$1"/shufnames_batch"$i".csv;
    bash quiz/hide_names.sh "$1" "$i";
done;
