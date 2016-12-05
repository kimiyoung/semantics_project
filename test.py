import sys
import numpy as np
import cPickle as pkl
import shutil

from config import *
from model import GAReader, GAReaderpp_prior, StanfordAR, GAReaderpp, GAReaderppp
from model import DeepASReader, DeepAoAReader
from utils import Helpers, DataPreprocessor, MiniBatchLoader

def main(load_path, params, mode='test'):

    regularizer = params['regularizer']
    rlambda = params['lambda']
    nhidden = params['nhidden']
    dropout = params['dropout']
    word2vec = params['word2vec']
    dataset = params['dataset']
    nlayers = params['nlayers']
    train_emb = params['train_emb']
    subsample = params['subsample']
    base_model = params['model']
    char_dim = params['char_dim']
    use_feat = params['use_feat']
    gating_fn = params['gating_fn']
    coref = params['coref']

    use_chars = char_dim>0
    dp = DataPreprocessor.DataPreprocessor()
    data = dp.preprocess(dataset, no_training_set=True, use_chars=use_chars)
    inv_vocab = data.inv_dictionary

    print("building minibatch loaders ...")
    if mode=='test':
        batch_loader_test = MiniBatchLoader.MiniBatchLoader(data.test, BATCH_SIZE)
    else:
        batch_loader_test = MiniBatchLoader.MiniBatchLoader(data.validation, BATCH_SIZE)

    print("building network ...")
    W_init, embed_dim = Helpers.load_word2vec_embeddings(data.dictionary[0], word2vec)
    m = eval(base_model).Model(nlayers, data.vocab_size, data.num_chars, W_init, 
            regularizer, rlambda, nhidden, embed_dim, dropout, train_emb, subsample, 
            char_dim, use_feat, gating_fn, coref, save_attn=True)
    m.load_model('%s/best_model.p'%load_path)

    # predict among corefs
    def coref_predictions(preds, ans, corefs, asum, doc, wc, pas):
        any_acc, rand_acc, freq_acc, long_acc, wc_acc, as_acc = 0., 0., 0., 0., 0., 0.
        for ii in range(ans.shape[0]):
            ans_tok = np.argwhere(asum[ii,:,ans[ii]])
            pred_tok = np.argwhere(corefs[ii,:,preds[ii]])
            rand_tok = np.random.choice(pred_tok.flatten())
            doc_preds = doc[ii, pred_tok, 0].flatten()
            counts = [wc[dd] for dd in doc_preds]
            min_cidx = min(enumerate(counts), key=lambda x:x[1])[0]
            wc_pred = doc_preds[min_cidx]
            tokens = [inv_vocab[t] for t in doc_preds]
            max_idx = max(enumerate(tokens), key=lambda x:len(x[1]))[0]
            long_pred = doc_preds[max_idx]
            (val,counts) = np.unique(doc_preds, return_counts=True)
            freq_pred = val[np.argmax(counts)]
            coref_cand = [np.argmax(asum[ii,pp,:]) for pp in pred_tok]
            as_probs = pas[ii,coref_cand]
            as_pred = np.argmax(as_probs)
            if coref_cand[as_pred]==ans[ii]: as_acc += 1.
            if long_pred in doc[ii, ans_tok, 0]: long_acc += 1.
            if wc_pred in doc[ii, ans_tok, 0]: wc_acc += 1.
            if freq_pred in doc[ii, ans_tok, 0]: freq_acc += 1.
            if any(pp in ans_tok for pp in pred_tok): 
                any_acc += 1.
                if coref_cand[as_pred]!=ans[ii]:
                    print "break here"
            if rand_tok in ans_tok: rand_acc += 1.
        return (any_acc/ans.shape[0], rand_acc/ans.shape[0], 
                freq_acc/ans.shape[0], long_acc/ans.shape[0], wc_acc/ans.shape[0], 
                as_acc/ans.shape[0])

    print("testing ...")
    pr = np.zeros((len(batch_loader_test.questions),
        batch_loader_test.max_num_cand)).astype('float32')
    pr_c = np.zeros((len(batch_loader_test.questions),
        batch_loader_test.max_doc_len)).astype('float32')
    d_pr = np.zeros((len(batch_loader_test.questions),
        batch_loader_test.max_doc_len)).astype('float32')
    fids, attns = [], []
    total_loss, total_loss_c, total_acc, total_acc_c, n = 0., 0., 0., 0., 0
    total_acc_r, total_acc_f, total_acc_l, total_acc_w, total_acc_a = 0., 0., 0., 0., 0.
    for dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, cl, cr, a_cr, fnames in batch_loader_test:
        outs = m.validate(dw, dt, qw, qt, c, a, m_dw, m_qw, tt, tm, m_c, cl, cr, a_cr)
        loss, acc, probs, loss_c, acc_c, probs_c, doc_probs = outs[:7]
        ans_c = np.argmax(probs_c,axis=1)
        acc_c_n, acc_c_r, acc_c_f, acc_c_l, acc_c_w, acc_c_a = coref_predictions(ans_c, a, cr, c, dw, 
            data.word_counts, probs)

        bsize = dw.shape[0]
        total_loss += bsize*loss
        total_loss_c += bsize*loss_c
        total_acc += bsize*acc
        total_acc_c += bsize*acc_c_n
        total_acc_r += bsize*acc_c_r
        total_acc_f += bsize*acc_c_f
        total_acc_l += bsize*acc_c_l
        total_acc_w += bsize*acc_c_w
        total_acc_a += bsize*acc_c_a

        pr[n:n+bsize,:] = probs
        pr_c[n:n+bsize,:probs_c.shape[1]] = probs_c
        d_pr[n:n+bsize,:doc_probs.shape[1]] = doc_probs
        fids += fnames
        n += bsize

    logger = open(load_path+'/log','a',0)
    message = '%s Loss %.4e coref_loss=%.4e acc=%.4f coref_acc=%.4f rand_coref_acc=%.4f freq_coref_acc=%.4f long_coref_acc=%.4f wc_coref_acc=%.4f as_coref_acc=%.4f' % (mode.upper(), total_loss/n, 
            total_loss_c, total_acc/n, total_acc_c/n, total_acc_r/n, total_acc_f/n, 
            total_acc_l/n, total_acc_w/n, total_acc_a/n)
    print message
    logger.write(message+'\n')
    logger.close()

    np.save('%s/%s.probs' % (load_path,mode),np.asarray(pr))
    pkl.dump(attns, open('%s/%s.attns' % (load_path,mode),'w'))
    f = open('%s/%s.ids' % (load_path,mode),'w')
    for item in fids: f.write(str(item)+'\n')
    f.close()
