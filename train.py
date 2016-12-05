import numpy as np
import time
import os
import shutil

from config import *
from model import GAReader, GAReaderpp_prior, StanfordAR, GAReaderpp, GAReaderppp
from model import DeepASReader, DeepAoAReader
from model import GAReaderCoref
from utils import Helpers, DataPreprocessor, MiniBatchLoader

def main(save_path, params):

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
    train_cut = params['train_cut']
    gating_fn = params['gating_fn']
    coref = params['coref']

    # save settings
    shutil.copyfile('config.py','%s/config.py'%save_path)

    use_chars = char_dim>0
    dp = DataPreprocessor.DataPreprocessor()
    data = dp.preprocess(dataset, no_training_set=False, use_chars=use_chars)

    print("building minibatch loaders ...")
    batch_loader_train = MiniBatchLoader.MiniBatchLoader(data.training, BATCH_SIZE, 
            sample=train_cut)
    batch_loader_val = MiniBatchLoader.MiniBatchLoader(data.validation, BATCH_SIZE)

    print("building network ...")
    W_init, embed_dim, = Helpers.load_word2vec_embeddings(data.dictionary[0], word2vec)
    m = eval(base_model).Model(nlayers, data.vocab_size, data.num_chars, W_init, 
            regularizer, rlambda, nhidden, embed_dim, dropout, train_emb, subsample, 
            char_dim, use_feat, gating_fn, coref)

    print("training ...")
    num_iter = 0
    max_acc = 0.
    max_acc_c = 0.
    deltas = []

    logger = open(save_path+'/log','a',0)

    if os.path.isfile('%s/best_model.p'%save_path):
        print('loading previously saved model')
        m.load_model('%s/best_model.p'%save_path)
    else:
        print('saving init model')
        m.save_model('%s/model_init.p'%save_path)
        print('loading init model')
        m.load_model('%s/model_init.p'%save_path)

    for epoch in xrange(NUM_EPOCHS):
        estart = time.time()
        new_max = False

        for (dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, 
                cl, cr, a_cr, ci, fnames) in batch_loader_train:
            loss, tr_acc, probs, loss_c, tr_acc_c, probs_c = m.train(dw, dt, qw, qt, c, a, m_dw, 
                    m_qw, tt, tm, m_c, cl, cr, a_cr, ci)

            if np.isnan(loss_c).any():
                print "break here"
            message = "Epoch %d TRAIN loss=%.4e coref_loss=%.4e acc=%.4f coref_acc=%.4f elapsed=%.1f" % (
                    epoch, loss, loss_c, tr_acc, tr_acc_c, time.time()-estart)
            print message
            logger.write(message+'\n')

            num_iter += 1
            if num_iter % VALIDATION_FREQ == 0:
                total_loss, total_loss_c, total_acc, total_acc_c, n, n_cand = 0., 0., 0., 0., 0, 0.

                for (dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, 
                        cl, cr, a_cr, ci, fnames) in batch_loader_val:
                    outs = m.validate(dw, dt, qw, qt, c, a, 
                            m_dw, m_qw, tt, tm, m_c, cl, cr, a_cr, ci)
                    loss, acc, probs, loss_c, acc_c, probs_c, doc_probs = outs[:7]
                    ans_c = np.argmax(probs_c,axis=1)
                    acc_c_n = 0.
                    for ii in range(ans_c.shape[0]):
                        ans_tok = np.argwhere(c[ii,:,a[ii]])
                        pred_tok = np.argwhere(cr[ii,:,ans_c[ii]])
                        if any(pp in ans_tok for pp in pred_tok): acc_c_n += 1.
                    acc_c_n = acc_c_n/ans_c.shape[0]

                    bsize = dw.shape[0]
                    total_loss += bsize*loss
                    total_loss_c += bsize*loss_c
                    total_acc += bsize*acc
                    total_acc_c += bsize*acc_c_n
                    n += bsize

		val_acc = total_acc/n
                if val_acc > max_acc:
                    max_acc = val_acc
                    if not coref:
                        m.save_model('%s/best_model.p'%save_path)
                        new_max = True
                val_acc_c = total_acc_c/n
                if val_acc_c > max_acc_c:
                    max_acc_c = val_acc_c
                    if coref:
                        m.save_model('%s/best_model.p'%save_path)
                        new_max = True
                message = "Epoch %d VAL loss=%.4e coref_loss=%.4e acc=%.4f coref_acc=%.4f max_acc=%.4f coref_max_acc=%.4f" % (
                    epoch, total_loss/n, total_loss_c/n, val_acc, val_acc_c, max_acc, max_acc_c)
                print message
                logger.write(message+'\n')

        m.save_model('%s/model_%d.p'%(save_path,epoch))
        message = "After Epoch %d: Train acc=%.4f, Val acc=%.4f" % (epoch, tr_acc, max_acc)
        print message
        logger.write(message+'\n')
        
        # learning schedule
        if epoch >=2:
            m.anneal()
        # stopping criterion
        if not new_max:
            break

    logger.close()
