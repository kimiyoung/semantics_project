import numpy as np
import sys

from utils import Helpers, DataPreprocessor, MiniBatchLoader
from model import BidirectionalLSTMReader

# NOTE: config.py should be consistent with the training model
from config import *

model_path = sys.argv[1]
output_path = sys.argv[2]
top_K = 3

dp = DataPreprocessor.DataPreprocessor()

# NOTE: make sure vocab.txt is already there!
data = dp.preprocess("cnn/questions", no_training_set=True)
inv_vocab = data.inv_dictionary

print("building minibatch loaders ...")
batch_loader_test = MiniBatchLoader.MiniBatchLoader(data.test, 128)

print("building network ...")
m = BidirectionalLSTMReader.Model(data.vocab_size)

print("loading model from file...")
m.load_model(model_path)

print("predicting ...")

fid = open(output_path,'w',0)

for d, q, a, m_d, m_q, c, m_c, fnames in batch_loader_test:
    loss, acc, probs = m.validate(d, q, a, m_d, m_q)

    probs_sorted = np.argpartition(-probs, top_K-1)[:,:top_K]
    predicted = map(lambda x:' '.join(map(lambda i:inv_vocab[i], x)), probs_sorted)
    ground_truth = map(lambda i:inv_vocab[i], a)

    n = d.shape[0]
    for i in xrange(n):
        question_id = fnames[i].split('/')[-1].split('.')[0]
        p = predicted[i]
        g = ground_truth[i]
        doc_len = m_d[i].sum()
        qry_len = m_q[i].sum()
        ans_freq = (d[i] == a[i]).sum()
        fid.write('%s %d %d %d %s %s\n' % (question_id, doc_len, qry_len, ans_freq, p, g))

fid.close()

