#!/bin/bash

modeldir="dailymail-ensemble/"
dataset="test"
vocabfile="word2vec/dailymail/vocab.txt"
w2vfile="word2vec/dailymail/word2vec_embed.txt"

cp $vocabfile .
cp $w2vfile .

for dir in ${modeldir}mul*/ ; do
    echo $dir
    k=$(echo $dir | cut -f2 -d_ | cut -c1)
    cp $dir/config.py .
    python predict.py $dir/best_model.p $dir/${dataset}.preds $dataset $k
done
