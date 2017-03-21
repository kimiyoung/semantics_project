import sys
import numpy as np
import cPickle as pkl
import shutil

from config import *
from model import GAReader, GAReaderpp_prior, StanfordAR, GAReaderpp, GAReaderppp, GAKnowledge, GAReaderCoref, GAReaderSelect
from model import GAReaderpropnc
from model import DeepASReader, DeepAoAReader
from utils import Helpers, DataPreprocessor, MiniBatchLoader

def main(load_path, params, mode='test'):

    word2vec = params['word2vec']
    dataset = params['dataset']
    base_model = params['model']

    use_chars = params['char_dim']>0
    dp = DataPreprocessor.DataPreprocessor()
    data = dp.preprocess(dataset, no_training_set=True, use_chars=use_chars)
    inv_vocab = data.inv_dictionary
    if data.validation[0][6]==-1:
        # no clozes
        cloze=False
    else:
        cloze=True

    print("building minibatch loaders ...")
    if mode=='test':
        batch_loader_test = MiniBatchLoader.MiniBatchLoader(data.test, BATCH_SIZE)
    else:
        batch_loader_test = MiniBatchLoader.MiniBatchLoader(data.validation, BATCH_SIZE)
    num_candidates = batch_loader_test.max_num_cand

    print("building network ...")
    W_init, embed_dim = Helpers.load_word2vec_embeddings(data.dictionary[0], word2vec)
    m = eval(base_model).Model(params, data.vocab_size, data.num_chars, W_init, 
            embed_dim, num_candidates, cloze=cloze, save_attn=True)
    m.load_model('%s/best_model.p'%load_path)

    print("testing ...")
    pr = np.zeros((len(batch_loader_test.questions),
        batch_loader_test.max_num_cand)).astype('float32')
    d_pr = np.zeros((len(batch_loader_test.questions),
        batch_loader_test.max_doc_len)).astype('float32')
    fids, attns = [], []
    dreps, qreps = [], []
    total_loss, total_acc, n = 0., 0., 0
    for dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, cl, crd, crq, fnames in batch_loader_test:
        outs = m.validate(dw, dt, qw, qt, c, a, m_dw, m_qw, tt, tm, m_c, cl, crd, crq)
        loss, acc, probs, drep, qrep, doc_probs = outs[:6]
        dreps.append(drep)
        qreps.append(qrep)

        bsize = dw.shape[0]
        total_loss += bsize*loss
        total_acc += bsize*acc

        pr[n:n+bsize,:] = probs
        d_pr[n:n+bsize,:doc_probs.shape[1]] = doc_probs
        fids += fnames
        n += bsize

    logger = open(load_path+'/log','a',0)
    message = '%s Loss %.4e acc=%.4f' % (mode.upper(), total_loss/n, total_acc/n)
    print message
    logger.write(message+'\n')
    logger.close()

    np.save('%s/%s.probs' % (load_path,mode),np.asarray(pr))
    if callable(getattr(m, "get_output_weights", None)):
        np.save('%s/out_emb.npy' % load_path, m.get_output_weights())
    pkl.dump(attns, open('%s/%s.attns' % (load_path,mode),'w'))
    pkl.dump([dreps, qreps], open('%s/%s.reps' % (load_path,mode),'w'))
    f = open('%s/%s.ids' % (load_path,mode),'w')
    for item in fids: f.write(str(item)+'\n')
    f.close()
