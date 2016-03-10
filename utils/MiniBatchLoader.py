import glob
import numpy as np
import random

class MiniBatchLoader():

    def __init__(self, questions, batch_size):
        self.batch_size = batch_size
        self.doc_len_map = self.build_doc_len_map(questions)
        self.max_qry_len = max(map(lambda x:len(x[1]), questions))
        self.max_num_cand = max(map(lambda x:len(x[3]), questions))
        self.questions = questions
	self.reset()

    def __iter__(self):
        """make the object iterable"""
        return self

    def build_doc_len_map(self, questions):
        """
        returns a dictionary
            key: document length (rounded to the powers of two)
            value: indexes of questions with document length equal to key
        """
        # round the input to the nearest power of two
        # example: 500 --> 512, 998 --> 1024
        round_to_power = lambda x: 2**(int(np.log2(x-1))+1)

        doc_len = map(lambda x:round_to_power(len(x[0])), questions)
        doc_len_map = {}
        for i, l in enumerate(doc_len):
            if l not in doc_len_map:
                doc_len_map[l] = []
            doc_len_map[l].append(i)

        return doc_len_map

    def reset(self):
        """new iteration"""
        # NOTE: for debugging purpuse, we will go through batches of shorter
        # doc length prior to batches of longer doc length.
        self.doc_len_iter = iter(sorted(self.doc_len_map.iterkeys()))
        self.curr_max_doc_len = self.doc_len_iter.next()
        self.ptr = 0
        # randomly shuffle the question indexes in each bin
        for question_ix in self.doc_len_map.itervalues():
            random.shuffle(question_ix)

    def next(self):
        """load the next batch"""
        if self.ptr == len(self.doc_len_map[self.curr_max_doc_len]): # end of each bin
            try:
                self.curr_max_doc_len = self.doc_len_iter.next()
            except StopIteration: # end of all bins
                self.reset()
                raise StopIteration()
            self.ptr = 0

        curr_question_ix = self.doc_len_map[self.curr_max_doc_len]
        curr_batch_size = min(self.batch_size, len(curr_question_ix)-self.ptr)

        d = np.zeros((curr_batch_size, self.curr_max_doc_len, 1), dtype='int32') # document
        q = np.zeros((curr_batch_size, self.max_qry_len, 1), dtype='int32') # query
        c = np.zeros((curr_batch_size, self.max_num_cand), dtype='int32') # candidate answers

        m_d = np.zeros((curr_batch_size, self.curr_max_doc_len), dtype='int32') # document mask
        m_q = np.zeros((curr_batch_size, self.max_qry_len), dtype='int32') # query mask
        m_c = np.zeros((curr_batch_size, self.max_num_cand), dtype='int32') # candidate mask

        a = np.zeros((curr_batch_size, ), dtype='int32') # correct answer

        for n in xrange(curr_batch_size):

            ix = curr_question_ix[self.ptr]
            doc, qry, ans, cand = self.questions[ix][:4]

            # document, query and candidates
            d[n,:len(doc),0] = np.array(doc)
            q[n,:len(qry),0] = np.array(qry)
            c[n,:len(cand)] = np.array(cand)

            # masks for document, query and candidates
            m_d[n,:len(doc)] = 1
            m_q[n,:len(qry)] = 1
            m_c[n,:len(cand)] = 1

            # correct answer
            a[n] = ans

            self.ptr += 1

        return d, q, a, m_d, m_q, c, m_c

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
    dp = DataPreprocessor()
    cnn = dp.preprocess("cnn/questions")

    mini_batch_loader = MiniBatchLoader(cnn.validation)
    unit_test(mini_batch_loader)

