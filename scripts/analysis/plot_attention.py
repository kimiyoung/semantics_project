"""
Script to plot D x Q attention matrix from each layer of GA Reader.
Usage: python plot_attention.py <path_to_model>
Where <path_to_model> has val.attns and val.ids
"""

import sys
import random
import numpy as np
import cPickle as pkl
import matplotlib.pylab as plt

def softmax(x,ax,m):
    """Compute softmax values for each sets of scores in x."""
    if ax==0: 
        exps = np.exp(x) 
        exps = exps*m[:,None]
        return exps / np.sum(exps, axis=ax)[None,:]
    else: 
        exps = np.exp(x) 
        exps = exps*m[None,:]
        return exps / np.sum(exps, axis=ax)[:,None]

WRAP=True

attns = pkl.load(open(sys.argv[1]))
X = attns[int(sys.argv[2])]

text = open(X[0]).readlines()
doc_raw = text[2].split()
qry_raw = text[4].split()
if WRAP: qry_raw = ['<beg>']+qry_raw+['<end>']
M = X[2:]
print X[0]
print text[2], '\n'
print text[4], '\n'
print text[6], '\n'
for ii,item in enumerate(text[8:]):
    print item, X[1][ii]
doc_mask = np.zeros((M[0].shape[0],))
doc_mask[:len(doc_raw)] = 1
qry_mask = np.zeros((M[0].shape[1],))
qry_mask[:len(qry_raw)] = 1

for j,item in enumerate(M):
    alphas = softmax(M[j], 0, doc_mask)
    betas = softmax(M[j], 1, qry_mask)
    fig = plt.figure(j)
    ax = fig.add_subplot(1,2,1)
    plt.imshow(alphas[:len(doc_raw),:len(qry_raw)], 
            interpolation='nearest',
            cmap=plt.cm.ocean)
    plt.colorbar()
    plt.xticks(range(len(qry_raw)))
    plt.yticks(range(len(doc_raw)))
    ax.set_xticklabels(qry_raw, rotation=45)
    ax.set_yticklabels(doc_raw)
    ax = fig.add_subplot(1,2,2)
    plt.imshow(betas[:len(doc_raw),:len(qry_raw)], 
            interpolation='nearest',
            cmap=plt.cm.ocean)
    plt.colorbar()
    plt.xticks(range(len(qry_raw)))
    plt.yticks(range(len(doc_raw)))
    ax.set_xticklabels(qry_raw, rotation=45)
    ax.set_yticklabels(doc_raw)
plt.show()
