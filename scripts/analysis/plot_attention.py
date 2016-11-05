"""
Script to plot D x Q attention matrix from each layer of GA Reader.
Usage: python plot_attention.py <path_to_model>
Where <path_to_model> has val.attns and val.ids
"""

import io
import os
import sys
import random
import numpy as np
import cPickle as pkl
import matplotlib
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec

#FS = 12
CH = 160
attns = pkl.load(open(sys.argv[1]))
root_dir = 'scripts/analysis/interesting/'

matplotlib.rcParams.update({'font.size': 16})

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

def contiguous(list1,list2):
    # return contiguous peices of list1 that are also 
    # in list2
    contg = []
    buf = []
    for ii,item in enumerate(list1):
        if item in list2:
            buf.append(ii)
        elif buf: 
            contg.append(buf)
            buf = []
    return contg

def disp_index(list1):
    # return flat array of indices to display
    # from list of lists
    all_ids = []
    if not list1[0]==0:
        all_ids.append(list1[0]-1)
    all_ids.extend(list1)
    all_ids.append(list1[-1]+1)
    return all_ids

WRAP=True

save_path = root_dir+attns[0][0].rsplit('/',1)[0]
if not os.path.exists(save_path): os.makedirs(save_path)

for X in attns:
    # read query and display
    text = io.open(X[0]).readlines()
    doc_raw = text[2].split()
    qry_raw = text[4].split()
    if WRAP: qry_raw = ['<beg>']+qry_raw+['<end>']
    placeholder = qry_raw.index('@placeholder')
    qry_raw[placeholder] = 'XXX'
    ph_range = range(max(0,placeholder-2),min(len(qry_raw),placeholder+3))
    M = X[3:]
    # collect candidates and mask
    cand_idx = []
    for ii,item in enumerate(text[8:]):
        tokens = item.split()
        cand_idx.extend(contiguous(doc_raw, tokens))
    # masks
    doc_mask = np.zeros((M[0].shape[0],))
    doc_mask[:len(doc_raw)] = 1
    qry_mask = np.zeros((M[0].shape[1],))
    qry_mask[:len(qry_raw)] = 1

    # gated attention layers
    nc = len(cand_idx)
    gs = GridSpec(nc,len(M),width_ratios=[len(qry_raw)]*(len(M)-1)+[len(ph_range)])
    fig = plt.figure(figsize=[20,11])
    for j in range(len(M)-1):
        # only display candidates
        betas = softmax(M[j], 1, qry_mask)
        for ic,cc in enumerate(cand_idx):
            ax = fig.add_subplot(gs[ic,j])
            all_ids = disp_index(cc)
            im = ax.imshow(betas[all_ids,:len(qry_raw)], 
                    interpolation='nearest',
                    vmin=0., vmax=1.,
                    cmap=plt.cm.Reds,
                    aspect='auto')
            ax.set_xticks([])
            ax.set_yticks(range(len(all_ids)))
            ax.set_yticklabels([doc_raw[ii][:12] for ii in all_ids], rotation=30, va='top')
        ax.set_xticks(range(len(qry_raw)))
        ax.set_xticklabels([qq[:10] for qq in qry_raw], rotation=60, ha='right')

    # attention sum layer
    alphas = softmax(M[-1], 0, doc_mask)
    # renormalize
    flat_cand_idx = [c for cc in cand_idx for c in cc]
    alpha_hat = np.zeros(alphas.shape)
    alpha_hat[flat_cand_idx,placeholder] = alphas[flat_cand_idx,placeholder]
    alpha_hat = alpha_hat/alpha_hat.sum(axis=0)[None,:]
    for ic,cc in enumerate(cand_idx):
        ax = fig.add_subplot(gs[ic,-1])
        all_ids = np.asarray(disp_index(cc))
        phr = np.asarray(ph_range)
        im = ax.imshow(alpha_hat[all_ids[:,None],phr[None,:]], 
                interpolation='nearest',
                vmin=0., vmax=1.,
                cmap=plt.cm.Reds,
                aspect='auto')
        ax.set_xticks([])
        ax.set_yticks(range(len(all_ids)))
        ax.set_yticklabels([doc_raw[ii][:12] for ii in all_ids], rotation=30, va='top')
    ax.set_xticks(range(len(ph_range)))
    ax.set_xticklabels([qry_raw[pp][:10] for pp in ph_range], rotation=60, ha='right')
    fig.subplots_adjust(hspace=0)

    # add caption
    doc_c = ''
    for w in doc_raw: doc_c += w+' '
    qry_c = ''
    for w in qry_raw: qry_c += w+' '
    cap = ('DOC: '+'\n'.join([''.join(doc_c[idx:idx+CH]) for idx in range(0,len(doc_c),CH)]) + '\n\n' +
            'QRY: '+'\n'.join([''.join(qry_c[idx:idx+CH]) for idx in range(0,len(qry_c),CH)]) + '\n' +
            'ANS: '+''.join(text[6].rstrip()))
    num_lines = cap.count('\n')
    if num_lines<10: bot = 0.35
    elif num_lines<13: bot = 0.40
    elif num_lines<16: bot = 0.45
    elif num_lines<19: bot = 0.50
    else: bot = 0.55
    fig.subplots_adjust(bottom=bot)
    plt.figtext(0.06,0.0, cap, weight='medium', )

    # add colorbar
    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([0.97, bot, 0.01, 0.90-bot])
    fig.colorbar(im, cax=cbar_ax)

    #plt.show()
    plt.savefig(root_dir+'%s.png'%X[0], bbox_inches='tight')
    plt.close()
