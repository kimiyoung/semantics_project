import numpy as np
import time
import os
import shutil

from config import *
from model import GAReaderpp_prior
from utils import Helpers, DataPreprocessor, MiniBatchLoader

def main(save_path, regularizer, rlambda, nhidden, dropout, word2vec, dataset, nlayers, 
        train_emb, subsample):
    # save settings
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    shutil.copyfile('config.py','%s/config.py'%save_path)

    dp = DataPreprocessor.DataPreprocessor()
    data = dp.preprocess(dataset, no_training_set=False)

    print("building minibatch loaders ...")
    batch_loader_train = MiniBatchLoader.MiniBatchLoader(data.training, BATCH_SIZE)
    batch_loader_val = MiniBatchLoader.MiniBatchLoader(data.validation, 128)

    print("building network ...")
    W_init, embed_dim = Helpers.load_word2vec_embeddings(data.dictionary, word2vec)
    m = GAReaderpp_prior.Model(nlayers, data.vocab_size, W_init, regularizer, rlambda, 
            nhidden, embed_dim, dropout, train_emb, subsample)

    print("training ...")
    num_iter = 0
    max_acc = 0.
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

        for d, q, a, m_d, m_q, c, m_c, fnames in batch_loader_train:
            loss, tr_acc, probs = m.train(d, q, c, a, m_d, m_q, m_c)

            message = "Epoch %d TRAIN loss=%.4e acc=%.4f elapsed=%.1f" % (
                    epoch, loss, tr_acc, time.time()-estart)
            print message
            logger.write(message+'\n')

            num_iter += 1
            if num_iter % VALIDATION_FREQ == 0:
                total_loss, total_acc, n, n_cand = 0., 0., 0, 0.

                for d, q, a, m_d, m_q, c, m_c, fnames in batch_loader_val:
                    loss, acc, probs = m.validate(d, q, c, a, m_d, m_q, m_c)

                    bsize = d.shape[0]
                    total_loss += bsize*loss
                    total_acc += bsize*acc
                    n += bsize

                if total_acc/n > max_acc:
                    max_acc = total_acc/n
                    m.save_model('%s/best_model.p'%save_path)
                message = "Epoch %d VAL loss=%.4e acc=%.4f max_acc=%.4f" % (
                    epoch, total_loss/n, total_acc/n, max_acc)
                print message
                logger.write(message+'\n')

        m.save_model('%s/model_%d.p'%(save_path,epoch))
        message = "After Epoch %d: Train acc=%.4f, Val acc=%.4f" % (epoch, tr_acc, total_acc)
        print message
        logger.write(message+'\n')
        
    logger.close()
