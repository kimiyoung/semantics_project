#!/bin/bash

quesdir="cnn/questions/test/"
targetdir="sample/"
N=100

ls $quesdir | sort -R | tail -$N | while read file; do
    cp $quesdir/$file $targetdir
done
