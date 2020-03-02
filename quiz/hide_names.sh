#!/usr/bin/env bash
INPUT=quiz/origdata_"$1"/shufnames_batch"$2".csv
OLDIFS=$IFS
IFS=,
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
i=0
while read flname fake
do
    cp "$flname" quiz/shufdata_"$1"/batch"$2"/sp"${i}".fits
    if [ "${i}" -eq 0 ]; then 
	echo sp"${i}".fits "${fake}" > quiz/shufdata_"$1"/truelist_"$2".txt
    else
	echo sp"${i}".fits "${fake}" >> quiz/shufdata_"$1"/truelist_"$2".txt
    fi
    i=$((${i}+1))
done < $INPUT
IFS=$OLDIFS
