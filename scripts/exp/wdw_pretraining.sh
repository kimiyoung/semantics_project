#!/bin/bash

SEED=2

python run.py --nhidden 128 --train_emb 1 --seed $SEED
python run.py --nhidden 128 --word2vec word2vec/word2vec_glove.txt --train_emb 0 --seed $SEED
python run.py --nhidden 128 --word2vec word2vec/word2vec_glove.txt --train_emb 1 --seed $SEED
python run.py --nhidden 128 --word2vec word2vec/wdw/wdw-train-vector-128.txt --train_emb 0 --seed $SEED
python run.py --nhidden 128 --word2vec word2vec/wdw/wdw-train-vector-128.txt --train_emb 1 --seed $SEED
