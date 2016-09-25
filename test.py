import numpy as np
import shutil

from config import *
from model import GAReaderpp_prior
from utils import Helpers, DataPreprocessor, MiniBatchLoader

def main(load_path, regularizer, rlambda, nhidden, dropout, word2vec, dataset, nlayers, 
        train_emb, subsample):
    # load settings
    shutil.copyfile('%s/config.py'%load_path,'config.py')

    dp = DataPreprocessor.DataPreprocessor()
    data = dp.preprocess(dataset, no_training_set=True)
    inv_vocab = data.inv_dictionary

    print("building minibatch loaders ...")
    batch_loader_test = MiniBatchLoader.MiniBatchLoader(data.test, 512)

    print("building network ...")
    W_init, embed_dim = Helpers.load_word2vec_embeddings(data.dictionary, word2vec)
    m = GAReaderpp_prior.Model(nlayers, data.vocab_size, W_init, regularizer, rlambda, 
            nhidden, embed_dim, dropout, train_emb, subsample)
    m.load_model('%s/best_model.p'%load_path)

    print("testing ...")
    pr = []
    total_loss, total_acc, n = 0., 0., 0
    for d, q, a, m_d, m_q, c, m_c, fnames in batch_loader_test:
        loss, acc, probs = m.validate(d, q, c, a, m_d, m_q, m_c)

        bsize = d.shape[0]
        total_loss += bsize*loss
        total_acc += bsize*acc
        n += bsize
        
        for i in xrange(n):
            question_id = fnames[i].split('/')[-1].split('.')[0]
            pr.append(probs[i,:])
    logger = open(load_path+'/log','a',0)
    message = 'TEST Loss %.4e acc=%.4f' % (total_loss/n, total_acc/n)
    print message
    logger.write(message+'\n')
    np.save('%s/test.probs'%load_path,np.asarray(pr))
    logger.close()
