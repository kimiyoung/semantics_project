import numpy as np
import shutil

from config import *
from model import GAReader, GAReaderpp_prior
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

    # load settings
    shutil.copyfile('%s/config.py'%load_path,'config.py')

    dp = DataPreprocessor.DataPreprocessor()
    data = dp.preprocess(dataset, no_training_set=True)
    inv_vocab = data.inv_dictionary

    print("building minibatch loaders ...")
    batch_loader_test = MiniBatchLoader.MiniBatchLoader(data.test, 512)

    print("building network ...")
    W_init, embed_dim = Helpers.load_word2vec_embeddings(data.dictionary, word2vec)
    m = eval(base_model).Model(nlayers, data.vocab_size, W_init, regularizer, rlambda, 
            nhidden, embed_dim, dropout, train_emb, subsample)
    m.load_model('%s/best_model.p'%load_path)

    print("testing ...")
    pr = np.zeros((len(batch_loader_test.questions),
        batch_loader_test.max_num_cand)).astype('float32')
    fids = []
    total_loss, total_acc, n = 0., 0., 0
    for d, q, a, m_d, m_q, c, m_c, fnames in batch_loader_test:
        loss, acc, probs = m.validate(d, q, c, a, m_d, m_q, m_c)

        bsize = d.shape[0]
        total_loss += bsize*loss
        total_acc += bsize*acc

        pr[n:n+bsize,:] = probs
        fids += fnames
        n += bsize

    logger = open(load_path+'/log','a',0)
    message = 'TEST Loss %.4e acc=%.4f' % (total_loss/n, total_acc/n)
    print message
    logger.write(message+'\n')
    logger.close()

    np.save('%s/test.probs'%load_path,np.asarray(pr))
    f = open('%s/test.ids'%load_path,'w')
    for item in fids: f.write(item+'\n')
    f.close()
