import glob
import numpy as np
import random

from config import MAX_WORD_LEN

class MiniBatchLoader():

    def __init__(self, questions, batch_size, shuffle=True, sample=1.0):
        self.batch_size = batch_size
        if sample==1.0: self.questions = questions
        else: self.questions = random.sample(questions, 
                int(sample*len(questions)))
        self.bins = self.build_bins(self.questions)
        self.max_doc_len = max(self.bins.keys())
        self.max_qry_len = max(map(lambda x:len(x[1]), self.questions))
        self.max_num_cand = max(map(lambda x:len(x[3]), self.questions))
        self.max_word_len = MAX_WORD_LEN
        self.shuffle = shuffle
	self.reset()

    def __iter__(self):
        """make the object iterable"""
        return self

    def build_bins(self, questions):
        """
        returns a dictionary
            key: document length (rounded to the powers of two)
            value: indexes of questions with document length equal to key
        """
        # round the input to the nearest power of two
        round_to_power = lambda x: 2**(int(np.log2(x-1))+1) if x>1 else 1

        doc_len = map(lambda x:round_to_power(len(x[0])), questions)
        bins = {}
        for i, l in enumerate(doc_len):
            if l not in bins:
                bins[l] = []
            bins[l].append(i)

        return bins

    def reset(self):
        """new iteration"""
        self.ptr = 0

        # randomly shuffle the question indices in each bin
        if self.shuffle:
            for ixs in self.bins.itervalues():
                random.shuffle(ixs)

        # construct a list of mini-batches where each batch is a list of question indices
        # questions within the same batch have identical max document length 
        self.batch_pool = []
        for l, ixs in self.bins.iteritems():
            n = len(ixs)
            k = n/self.batch_size if n % self.batch_size == 0 else n/self.batch_size+1
            ixs_list = [(ixs[self.batch_size*i:min(n, self.batch_size*(i+1))],l) for i in range(k)]
            self.batch_pool += ixs_list

        # randomly shuffle the mini-batches
        if self.shuffle:
            random.shuffle(self.batch_pool)

    def next(self):
        """load the next batch"""
        if self.ptr == len(self.batch_pool):
            self.reset()
            raise StopIteration()

        ixs = self.batch_pool[self.ptr][0]
        curr_max_doc_len = self.batch_pool[self.ptr][1]
        curr_batch_size = len(ixs)

        dw = np.zeros((curr_batch_size, curr_max_doc_len, 1), dtype='int32') # document words
        qw = np.zeros((curr_batch_size, self.max_qry_len, 1), dtype='int32') # query words
        c = np.zeros((curr_batch_size, curr_max_doc_len, self.max_num_cand), 
                dtype='int32')   # candidate answers
        cr = np.zeros((curr_batch_size, curr_max_doc_len), dtype='int32') # coref id
        cl = np.zeros((curr_batch_size,), dtype='int32') # position of cloze in query

        m_dw = np.zeros((curr_batch_size, curr_max_doc_len), dtype='int32')  # document word mask
        m_qw = np.zeros((curr_batch_size, self.max_qry_len), dtype='int32')  # query word mask
        m_c = np.zeros((curr_batch_size, curr_max_doc_len), dtype='int32') # candidate mask

        a = np.zeros((curr_batch_size, ), dtype='int32')    # correct answer
        fnames = ['']*curr_batch_size

        types = {}

        for n, ix in enumerate(ixs):

            doc_w, qry_w, ans, cand, doc_c, qry_c, cloze, coref, fname = self.questions[ix]

            # document, query and candidates
            dw[n,:len(doc_w),0] = np.array(doc_w)
            qw[n,:len(qry_w),0] = np.array(qry_w)
            m_dw[n,:len(doc_w)] = 1
            m_qw[n,:len(qry_w)] = 1
            for it, word in enumerate(doc_c):
                wtuple = tuple(word)
                if wtuple not in types:
                    types[wtuple] = []
                types[wtuple].append((0,n,it))
            for it, word in enumerate(qry_c):
                wtuple = tuple(word)
                if wtuple not in types:
                    types[wtuple] = []
                types[wtuple].append((1,n,it))

            # search candidates in doc
            ans_idx = []
            for it,cc in enumerate(cand):
                index = [ii for ii in range(len(doc_w)) if doc_w[ii] in cc]
                m_c[n,index] = 1
                c[n,index,it] = 1
                if ans==cc: 
                    ans_idx = index
                    a[n] = it # answer
            assert ans_idx, "answer index in doc empty! %s" % fname

            # build coref index 
            for ic, chain in enumerate(coref):
                cr[n,list(chain)] = ic+1

            cl[n] = cloze
            fnames[n] = fname

        # create type character matrix and indices for doc, qry
        dt = np.zeros((curr_batch_size, curr_max_doc_len), dtype='int32') # document token index
        qt = np.zeros((curr_batch_size, self.max_qry_len), dtype='int32') # query token index
        tt = np.zeros((len(types), self.max_word_len), dtype='int32') # type characters
        tm = np.zeros((len(types), self.max_word_len), dtype='int32') # type mask
        n = 0
        for k,v in types.iteritems():
            tt[n,:len(k)] = np.array(k)
            tm[n,:len(k)] = 1
            for (sw, bn, sn) in v:
                if sw==0: dt[bn,sn] = n
                else: qt[bn,sn] = n
            n += 1

        self.ptr += 1

        return dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, cl, cr, fnames

def unit_test(mini_batch_loader):
    """unit test to validate MiniBatchLoader using max-frequency (exclusive).
    The accuracy should be around 0.37 and should be invariant over different batch sizes."""
    hits, n = 0., 0
    for d, q, a, m_d, m_q, c, m_c in mini_batch_loader:
        for i in xrange(len(d)):
            prediction, max_count = -1, 0
            for cand in c[i]:
                count = (d[i]==cand).sum() + (q[i]==cand).sum()
                if count > max_count and cand not in q[i]:
                    max_count = count
                    prediction = cand
            n += 1
            hits += a[i] == prediction
        acc = hits/n
        print acc

if __name__ == '__main__':

    from DataPreprocessor import *
    
    cnn = DataPreprocessor().preprocess("lambada", True, no_training_set=True, use_chars=False)
    mini_batch_loader = MiniBatchLoader(cnn.validation, 64, True)
    for dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, cl, fnames in mini_batch_loader:
        print 'running'
