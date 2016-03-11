import numpy as np
import time
import sys

from config import *
from model import DeepLSTMReader
from model import BidirectionalLSTMReader
from utils import Helpers, DataPreprocessor, MiniBatchLoader

save_path = sys.argv[1]

dp = DataPreprocessor.DataPreprocessor()
data = dp.preprocess("cnn/questions", no_training_set=False)

print("building minibatch loaders ...")
batch_loader_train = MiniBatchLoader.MiniBatchLoader(data.training, BATCH_SIZE)
batch_loader_val = MiniBatchLoader.MiniBatchLoader(data.validation, 512)

print("building network ...")
W_init = Helpers.load_word2vec_embeddings(data.dictionary, WORD2VEC_PATH)
m = DeepLSTMReader.Model(data.vocab_size, W_init)

print("training ...")
num_iter = 0
for epoch in xrange(NUM_EPOCHS):
    estart = time.time()

    for d, q, a, m_d, m_q, c, m_c in batch_loader_train:
        loss, acc, probs = m.train(d, q, a, m_d, m_q)

        print "Epoch %d TRAIN loss=%.4e acc=%.4f elapsed=%.1f" % (
                epoch, loss, acc, time.time()-estart)

        num_iter += 1
        if num_iter % VALIDATION_FREQ == 0:
            total_loss, total_acc, n, n_cand = 0., 0., 0, 0.

            for d, q, a, m_d, m_q, c, m_c in batch_loader_val:
                loss, acc, probs = m.validate(d, q, a, m_d, m_q)

                # n_cand = #{prediction is a candidate answer}
                n_cand += Helpers.count_candidates(probs, c, m_c)

                bsize = d.shape[0]
                total_loss += bsize*loss
                total_acc += bsize*acc
                n += bsize

            print "Epoch %d VAL loss=%.4e acc=%.4f is_candidate=%.3f" % (
                    epoch, total_loss/n, total_acc/n, n_cand/n)

    m.save_model('%s/model_%d.npz'%(save_path,epoch))
