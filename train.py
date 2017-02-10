import numpy as np
import time
import os
import shutil

from config import *
from model import GAReader, GAReaderpp_prior, StanfordAR, GAReaderpp, GAReaderppp
from model import DeepASReader, DeepAoAReader
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
    numcoref = params['num_coref']
    corefdim = params['coref_dim']

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
            char_dim, use_feat, gating_fn, numcoref, corefdim)

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
                cl, cr, fnames) in batch_loader_train:
            loss, tr_acc, probs = m.train(dw, dt, qw, qt, c, a, m_dw, 
                    m_qw, tt, tm, m_c, cl, cr)

            message = "Epoch %d TRAIN loss=%.4e acc=%.4f elapsed=%.1f" % (
                    epoch, loss, tr_acc, time.time()-estart)
            print message
            logger.write(message+'\n')

            num_iter += 1
            if num_iter % VALIDATION_FREQ == 0:
                total_loss, total_acc, n, n_cand = 0., 0., 0., 0.

                for (dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, 
                        cl, cr, fnames) in batch_loader_val:
                    outs = m.validate(dw, dt, qw, qt, c, a, 
                            m_dw, m_qw, tt, tm, m_c, cl, cr)
                    loss, acc, probs, doc_probs = outs[:4]

                    bsize = dw.shape[0]
                    total_loss += bsize*loss
                    total_acc += bsize*acc
                    n += bsize

		val_acc = total_acc/n
                if val_acc > max_acc:
                    max_acc = val_acc
                    m.save_model('%s/best_model.p'%save_path)
                    new_max = True
                message = "Epoch %d VAL loss=%.4e acc=%.4f max_acc=%.4f" % (
                    epoch, total_loss/n, val_acc, max_acc)
                print message
                logger.write(message+'\n')

        m.save_model('%s/model_%d.p'%(save_path,epoch))
        message = "After Epoch %d: Train acc=%.4f, Val acc=%.4f" % (epoch, tr_acc, max_acc)
        print message
        logger.write(message+'\n')
        
        # learning schedule
        if epoch >=2 and epoch%ANNEAL==0:
            m.anneal()
        # stopping criterion
        if STOPPING and not new_max:
            break

    logger.close()
