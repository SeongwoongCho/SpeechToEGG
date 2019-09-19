#!/usr/bin/env bash

SET=$(seq 1 2225)

for i in $SET
do

    echo "Running loop seq "$i

    wget "http://stimmdb.coli.uni-saarland.de/csl2wav.php4?file=${i}-phrase" -O "${i}_sentence.wav"
    wget "http://stimmdb.coli.uni-saarland.de/csl2wav.php4?file=${i}-iau" -O "${i}_iau.wav"
done