#!/usr/bin/env bash
INPUT=shuffled_names.csv
OLDIFS=$IFS
IFS=,
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
i=0
while read flname fake
do
    cp "$flname" shufdata/sp"${i}".fits
    if [ "${i}" -eq 0 ]; then 
	echo sp"${i}".fits "${fake}" > shufdata/truelist.txt
    else
	echo sp"${i}".fits "${fake}" >> shufdata/truelist.txt
    fi
    i=$((${i}+1))
done < $INPUT
IFS=$OLDIFS
