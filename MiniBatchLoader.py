import glob
import numpy as np
import random
from config import BATCH_SIZE, SYMB_BEGIN, SYMB_END

class MiniBatchLoader():

    def __init__(self, directory, vocab_file):
        self.load_vocabulary(vocab_file)
        self.max_qry_len = 0
        self.data_by_lens = self.parse_all_files(directory)
	self.reset()

    def __iter__(self):
        """make the object iterable"""
        return self

    def reset(self):
        """new iteration"""
        # NOTE: for debugging purpuse, we will go through batches of shorter
        # doc length prior to batches of longer doc length.
        self.doc_len_iter = iter(sorted(self.data_by_lens.iterkeys()))
        self.curr_max_doc_len = self.doc_len_iter.next()
        self.ptr = 0
        # randomly shuffle the data in each bin
        for data in self.data_by_lens.itervalues():
            random.shuffle(data)

    def load_vocabulary(self, fname):
        """load vocabulary dictionary from external file"""
        pairs = map(lambda line:line.split(), open(fname))
        self.dictionary = {p[0]:int(p[1]) for p in pairs}
        self.inv_dictionary = {v: k for k, v in self.dictionary.items()}
        self.vocab_size = len(self.dictionary)

    def parse_all_files(self, directory):
        """parse all *.question files.
        @Returns: data_by_lens (dict)
            key: len(doc_idx) rounded to the power of two: 8, 16, ..., 1024, 2048
            value: a list of questions where each element = (doc_idx, qry_idx, ans_idx, fname)
        """
        # map x to the nearest power of 2
        # e.g., 500 --> 512; 998 --> 1024
        map_to_power = lambda x: 2**(int(np.log2(x-1))+1)
        data_by_lens = {}

        all_files = glob.glob(directory + '/*.question')
        for f in all_files:
            data = self.parse_file(f) + (f,)
            k = map_to_power(len(data[0]))
            if k not in data_by_lens:
                data_by_lens[k] = []
            data_by_lens[k].append(data)

        return data_by_lens

    def parse_file(self, fname):
        """load document, query and answer from a *.question file"""
        # load raw tokens
        content = open(fname).readlines()
        doc_raw = content[2].split() # document
        qry_raw = content[4].split() # query
        ans_raw = content[6].strip() # answer
        ent_raw = map(lambda x:x.split(':')[0], content[8:]) # entities

        # wrap the query with special symbols
        qry_raw.insert(0, SYMB_BEGIN)
        qry_raw.append(SYMB_END)

        # tokens --> indexes
        doc_idx = map(lambda w:self.dictionary[w], doc_raw)
        qry_idx = map(lambda w:self.dictionary[w], qry_raw)
        ans_idx = self.dictionary[ans_raw]
        ent_idx = map(lambda w:self.dictionary[w], ent_raw)

        self.max_qry_len = max(self.max_qry_len, len(qry_idx))

        return doc_idx, qry_idx, ans_idx, ent_idx

    def next(self):
        """load the next batch"""
        if self.ptr == len(self.data_by_lens[self.curr_max_doc_len]):
            try:
                self.curr_max_doc_len = self.doc_len_iter.next()
            except StopIteration:
                self.reset()
                raise StopIteration()
            self.ptr = 0

        data = self.data_by_lens[self.curr_max_doc_len]
        curr_batch_size = min(BATCH_SIZE, len(data)-self.ptr)

        d = np.zeros((curr_batch_size, self.curr_max_doc_len, 1), dtype='int32') # document
        q = np.zeros((curr_batch_size, self.max_qry_len, 1), dtype='int32') # query
        a = np.zeros((curr_batch_size, ), dtype='int32') # the correct answer
        md = np.zeros((curr_batch_size, self.curr_max_doc_len), dtype='float32') # mask for d
        mq = np.zeros((curr_batch_size, self.max_qry_len), dtype='float32') # mask for q
        e = [] # candidate answers (entities)

        for n in xrange(curr_batch_size):
            doc_idx, qry_idx, ans_idx, ent_idx = data[self.ptr][:4]

            d[n,:len(doc_idx),0] = np.array(doc_idx)
            q[n,:len(qry_idx),0] = np.array(qry_idx)
            a[n] = ans_idx
            md[n,:len(doc_idx)] = 1
            mq[n,:len(qry_idx)] = 1
            e.append(np.array(ent_idx))

            self.ptr += 1

        return d, q, a, md, mq, e

def unit_test():
    """unit test to validate MiniBatchLoader using max-frequency (exclusive).
    The accuracy should be around 0.37 and should be invariant over different batch sizes."""
    mini_batch_loader = MiniBatchLoader("cnn/questions/validation", "vocab.txt")
    hits, n = 0., 0
    for d, q, a, _, e in mini_batch_loader:
        for i in xrange(len(d)):
            prediction, max_count = -1, 0
            for entity in e[i]:
                count = (d[i]==entity).sum() + (q[i]==entity).sum()
                if count > max_count and entity not in q[i]:
                    max_count = count
                    prediction = entity
            n += 1
            hits += a[i] == prediction
        acc = hits/n
        print acc

if __name__ == '__main__':
    unit_test()

