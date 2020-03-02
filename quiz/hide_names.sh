#!/usr/bin/env bash
INPUT=origdata/shufnames_batch"$1".csv
OLDIFS=$IFS
IFS=,
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
i=0
while read flname fake
do
    cp "$flname" shufdata/batch"$1"/sp"${i}".fits
    if [ "${i}" -eq 0 ]; then 
	echo sp"${i}".fits "${fake}" > shufdata/truelist_"$1".txt
    else
	echo sp"${i}".fits "${fake}" >> shufdata/truelist_"$1".txt
    fi
    i=$((${i}+1))
done < $INPUT
IFS=$OLDIFS
