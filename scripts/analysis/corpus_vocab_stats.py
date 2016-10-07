"""
Script to build vocabulary from train, test and val sets and compare
Usage: python corpus_vocab_stats.py <corpus_dir>
"""

import glob
import sys

from matplotlib import pyplot as plt
from matplotlib_venn import venn3

GLOVE="/usr1/public/bdhingra/semantics_project/word2vec/word2vec_glove.txt"

class DataPreprocessor:

    def make_dictionary(self, question_dir):

        def get_vocab(fnames):
            vocab_set = set()
            n = 0.
            for fname in fnames:
                
                fp = open(fname)
                fp.readline()
                fp.readline()
                document = fp.readline().split()
                fp.readline()
                query = fp.readline().split()
                fp.close()

                vocab_set |= set(document) | set(query)

                # show progress
                n += 1
                if n % 10000 == 0:
                    print '%3d%%' % int(100*n/len(fnames))
            return vocab_set

        print "getting train"
        train = get_vocab(glob.glob(question_dir + "/training/*.question"))
        print "getting val"
        val = get_vocab(glob.glob(question_dir + "/validation/*.question"))
        print "getting test"
        test = get_vocab(glob.glob(question_dir + "/test/*.question"))
        
        return train, val, test

    def glove_vocabulary(self):
        
        print "getting glove"
        f = open(GLOVE, 'r')
        f.readline()
        vocab_set = set()

        for line in f:
            token = line.split()[0]
            vocab_set.add(token)
        f.close()

        return vocab_set

if __name__=='__main__':

    corpus_path = sys.argv[1]
    dp = DataPreprocessor()

    train_voc, val_voc, test_voc = dp.make_dictionary(corpus_path)
    glove_voc = dp.glove_vocabulary()
    
    plt.subplot(1,2,1)
    v = venn3([train_voc, val_voc, glove_voc], ('Train', 'Validation', 'Glove'))
    plt.title("Vocab Sizes")

    plt.subplot(1,2,2)
    v = venn3([train_voc, test_voc, glove_voc], ('Train', 'Test', 'Glove'))
    plt.title("Vocab Sizes")
    plt.show()
