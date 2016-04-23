import glob
import numpy as np
import random

class MiniBatchLoader():

    def __init__(self, questions, batch_size):
        self.batch_size = batch_size
        self.bins = self.build_bins(questions)
        self.max_qry_len = max(map(lambda x:len(x[1]), questions))
        self.max_num_cand = max(map(lambda x:len(x[3]), questions))
        self.questions = questions
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
        round_to_power = lambda x: 2**(int(np.log2(x-1))+1)

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
        random.shuffle(self.batch_pool)

    def next(self):
        """load the next batch"""
        if self.ptr == len(self.batch_pool):
            self.reset()
            raise StopIteration()

        ixs = self.batch_pool[self.ptr][0]
        curr_max_doc_len = self.batch_pool[self.ptr][1]
        curr_batch_size = len(ixs)

        d = np.zeros((curr_batch_size, curr_max_doc_len, 1), dtype='int32') # document
        q = np.zeros((curr_batch_size, self.max_qry_len, 1), dtype='int32') # query
        c = np.zeros((curr_batch_size, self.max_num_cand), dtype='int32')   # candidate answers

        m_d = np.zeros((curr_batch_size, curr_max_doc_len), dtype='int32')  # document mask
        m_q = np.zeros((curr_batch_size, self.max_qry_len), dtype='int32')  # query mask
        m_c = np.zeros((curr_batch_size, self.max_num_cand), dtype='int32') # candidate mask

        a = np.zeros((curr_batch_size, ), dtype='int32')    # correct answer
        fnames = ['']*curr_batch_size

        for n, ix in enumerate(ixs):

            doc, qry, ans, cand, fname = self.questions[ix]

            # document, query and candidates
            d[n,:len(doc),0] = np.array(doc)
            q[n,:len(qry),0] = np.array(qry)
            c[n,:len(cand)] = np.array(cand)

            # masks for document, query and candidates
            m_d[n,:len(doc)] = 1
            m_q[n,:len(qry)] = 1
            m_c[n,:len(cand)] = 1

            a[n] = ans # answer
            fnames[n] = fname

        self.ptr += 1

        return d, q, a, m_d, m_q, c, m_c, fnames

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
    
    cnn = DataPreprocessor().preprocess("cnn/questions", no_training_set=True)
    mini_batch_loader = MiniBatchLoader(cnn.validation, 64)
    unit_test(mini_batch_loader)

