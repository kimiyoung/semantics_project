import glob
import numpy as np
from config import MAX_DOC_LEN, MAX_QRY_LEN, BATCH_SIZE, SYMB_BEGIN, SYMB_END

class MiniBatchLoader():

    def __init__(self, directory, vocab_file):
        self.ptr = 0
        self.file_pool = glob.glob(directory + '/*.question')
        self.N = len(self.file_pool)
        self.load_vocabulary(vocab_file)

    def __iter__(self):
        """make the object iterable"""
        return self

    def load_vocabulary(self, fname):
        """load vocabulary dictionary from external file"""
        pairs = map(lambda line:line.split(), open(fname))
        self.dictionary = { p[0]:int(p[1]) for p in pairs }
        # self.vocab_size = len(self.dictionary)

    def parse_file(self, fname):
        """load document, query and answer from a *.question file"""
        # load raw tokens
        fp = open(fname)
        fp.readline()
        fp.readline()
        doc_raw = fp.readline().split() # document
        fp.readline()
        qry_raw = fp.readline().split() # query
        fp.readline()
        ans_raw = fp.readline().strip() # answer
        fp.close()

        # wrap the query with special symbols
        qry_raw.insert(0, SYMB_BEGIN)
        qry_raw.append(SYMB_END)

        # tokens --> indexes
        doc_idx = map(lambda w:self.dictionary[w], doc_raw)
        qry_idx = map(lambda w:self.dictionary[w], qry_raw)
        ans_idx = self.dictionary[ans_raw]

        return doc_idx, qry_idx, ans_idx

    def next(self):
        """load the next batch"""
        d = np.zeros((BATCH_SIZE, MAX_DOC_LEN, 1), dtype='int32')
        q = np.zeros((BATCH_SIZE, MAX_QRY_LEN, 1), dtype='int32')
        a = np.zeros((BATCH_SIZE, ), dtype='int32')

        mask = np.zeros((BATCH_SIZE, MAX_DOC_LEN + MAX_QRY_LEN), dtype='float32')

        for n in range(BATCH_SIZE):
            f = self.file_pool[self.ptr]

            doc_idx, qry_idx, ans_idx = self.parse_file(f)

            d[n,:len(doc_idx),0] = np.array(doc_idx)
            q[n,:len(qry_idx),0] = np.array(qry_idx)
            a[n] = ans_idx
            mask[n,:(len(doc_idx)+len(qry_idx))] = 1

            self.ptr = (self.ptr+1) % self.N

        return d, q, a, mask

if __name__ == '__main__':

    mini_batch_loader = MiniBatchLoader("cnn/questions/validation", "vocab.txt")
    d, q, a, m = mini_batch_loader.next()
    print "memory consumption: %f GB" % ((d.nbytes+q.nbytes+a.nbytes)/1e9)

