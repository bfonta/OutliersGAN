#!/bin/awk -f

$NF ~ "data" { map[$NF]=0 }
$NF ~ "gen" { map[$NF]=1 }
END {
    for (key in map)
	printf("%s,%d\n", key,map[key])
}
