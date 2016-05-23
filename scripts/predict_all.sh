#!/bin/bash

modeldir="ensemble/"
dataset="test"

for dir in $modeldir*/ ; do
    k=$(echo $dir | cut -f2 -d_ | cut -c1)
    cp $dir/config.py .
    python predict.py $dir/best_model.p $dir/${dataset}.preds $dataset $k
done
