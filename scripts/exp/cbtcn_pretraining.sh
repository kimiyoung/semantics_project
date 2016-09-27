#!/bin/bash

SEED=1

python run.py --nhidden 128 --train_emb 1 --seed $SEED --dataset cbtcn
python run.py --nhidden 128 --word2vec word2vec/word2vec_glove.txt --train_emb 0 --seed $SEED --dataset cbtcn
python run.py --nhidden 128 --word2vec word2vec/word2vec_glove.txt --train_emb 1 --seed $SEED --dataset cbtcn
python run.py --nhidden 128 --word2vec word2vec/cbt/cn/cbtcn-train-vector-128.txt --train_emb 0 --seed $SEED --dataset cbtcn
python run.py --nhidden 128 --word2vec word2vec/cbt/cn/cbtcn-train-vector-128.txt --train_emb 1 --seed $SEED --dataset cbtcn
