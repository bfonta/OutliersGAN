#!/usr/bin/env bash
for i in {0..14}; do
    mkdir shufdata/batch"$i"/;
    ls origdata/batch"$i"/*.fits | awk -f shuffle.awk | shuf -o origdata/shufnames_batch"$i".csv;
    bash hide_names.sh "$i";
done;
