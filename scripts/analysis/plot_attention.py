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

WRAP=False

attns = pkl.load(open(sys.argv[1]))

X = random.choice(attns)
text = open(X[0]).readlines()
doc_raw = text[2].split()
qry_raw = text[4].split()
if WRAP: qry_raw = ['<beg>']+qry_raw+['<end>']
M = X[1:]
print X[0]

for j,item in enumerate(M):
    fig = plt.figure(j)
    ax = fig.add_subplot(1,1,1)
    plt.imshow(M[j][:len(doc_raw),:len(qry_raw)], 
            interpolation='nearest',
            cmap=plt.cm.ocean)
    plt.colorbar()
    plt.xticks(range(len(qry_raw)))
    plt.yticks(range(len(doc_raw)))
    ax.set_xticklabels(qry_raw, rotation=45)
    ax.set_yticklabels(doc_raw)
plt.show()
