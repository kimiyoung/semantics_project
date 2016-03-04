import glob
from config import SYMB_BEGIN, SYMB_END

""" output the vocabulary list of all tokens to vocab.txt.
by default, index 0, 1 correspond to special tokens @begin and @end.
"""

all_vocab = set()

fnames = []
fnames += glob.glob("cnn/questions/test/*.question")
fnames += glob.glob("cnn/questions/validation/*.question")
fnames += glob.glob("cnn/questions/training/*.question")

n = 0.
for fname in fnames:
    
    fp = open(fname)
    fp.readline()
    fp.readline()
    document = fp.readline().split()
    fp.readline()
    query = fp.readline().split()
    fp.close()

    all_vocab |= set(document) | set(query)
    n += 1
    if n % 1000 == 0:
        print "%.2f" % (100*n/len(fnames))

fp = open("vocab.txt", "w")

fp.write(SYMB_BEGIN + " 0\n")
fp.write(SYMB_END + " 1\n")

for i, w in enumerate(all_vocab):
    fp.write("%s %d\n" % (w, i+2))
fp.close()

