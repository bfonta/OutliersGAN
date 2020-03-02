#!/usr/bin/env bash
for i in {0..14}; do
    mkdir quiz/shufdata_"$1"/batch"$i"/;
    ls quiz/origdata_"$1"/batch"$i"/*.fits | awk -f quiz/shuffle.awk | shuf -o quiz/origdata_"$1"/shufnames_batch"$i".csv;
    bash quiz/hide_names.sh "$1" "$i";
done;
