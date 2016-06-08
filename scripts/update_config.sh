#!/bin.bash

modeldir="dailymail-ensemble/"

for dir in $modeldir*/ ; do
    sed -i "26s:.*:DATASET = 'dailymail/questions':" $dir/config.py
done
