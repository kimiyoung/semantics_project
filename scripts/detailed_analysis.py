import statsmodels.api as sm
import numpy as np
import sys
from matplotlib import pyplot as plt

class Question:
    pass

def conditional_performance(condition, samples, measure):
    Q = filter(condition, questions)
    y = np.array(map(measure, Q), dtype=float)
    return np.mean(y)

if __name__ == '__main__':

    measure = lambda q:q.true_ans in [q.pred[0]]

    stats_file = 'corpus-stats-cnn100.txt'
    preds_file = sys.argv[1]

    samples = {}

    for line in open(stats_file):
        info = line.split()

        q = Question()

        q.doc_len        =  float(info[1])
        q.qry_len        =  int(info[2])
        q.ans_freq       =  int(info[3])
        q.ans_pos_first  =  float(info[4])/q.doc_len
        q.ans_pos_last   =  1.-float(info[5])/q.doc_len
        q.sents          =  int(info[6])
        q.ngram          =  int(info[7])  ==  1
        q.para           =  int(info[8])  ==  1
        q.logical        =  int(info[9])  ==  1
        q.temporal       =  int(info[10]) ==  1

        q.doc_len = np.log(q.doc_len) # only order matters

        samples[info[0]] = q

    for line in open(preds_file):
        info = line.split()
        if info[0] in samples:
            samples[info[0]].pred = info[-4:-1]
            samples[info[0]].true_ans = info[-1]

    questions = samples.values()
    print "Overall Performance"
    print  conditional_performance(lambda  q:True,    questions, measure)

    print "Conditional Performance:"

    print "sents==1\t",
    print  conditional_performance(lambda  q:q.sents    == 1,    questions, measure)
    print "sents==2\t",
    print  conditional_performance(lambda  q:q.sents    == 2,    questions, measure)
    print "sents==3\t",
    print  conditional_performance(lambda  q:q.sents    == 3,    questions, measure)
    print "ngram\t",
    print  conditional_performance(lambda  q:q.ngram    == True, questions, measure)
    print "paraphrase\t",
    print  conditional_performance(lambda  q:q.para     == True, questions, measure)
    print "logical\t",
    print  conditional_performance(lambda  q:q.logical  == True, questions, measure)
    print "temporal\t",
    print  conditional_performance(lambda  q:q.temporal == True, questions, measure)

    X = np.array(map(lambda q:[
        q.doc_len, q.qry_len, q.ans_freq, q.ans_pos_first, q.ans_pos_last,
        q.sents, q.ngram, q.para, q.logical, q.temporal], questions), dtype=float)

    # normalize by max
    X = X.dot(np.diag(1/np.max(X, axis=0)))

    # intercept
    X = sm.add_constant(X, prepend=False)
    y = np.array(map(measure, questions), dtype=float)
    y = y[:,np.newaxis]

    model = sm.OLS(y,X)
    result = model.fit()

    print "Regression Analysis:"
    pvalues = result.pvalues
    beta = result.params

    feature_names = ["doc_len", "qry_len", "ans_freq", "ans_pos_first", "ans_pos_last",
        "sents", "ngram", "para", "logical", "temporal", "intercept"]
    algorithm_names = ["GAR"]

    for i, (pvalue, b) in enumerate(np.vstack([pvalues, beta]).T):
        print "%s\t%f\t%f" % (feature_names[i], b, pvalue)

