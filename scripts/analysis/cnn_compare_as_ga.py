"""
compare DeepASReader with GAReader on mcnemar one-sided test
usage: python compare_as_ga.py <as_dir> <ga_dir>
"""
import numpy as np
from scipy.stats import binom

as_files = {
        'feat0' :   ['../../experiments/DeepASReader/cnn/regl20.000_nhid256_nlayers3_dropout0.2_word2vec_glove_chardim0_train1_subsample-1_seed1_use-feat0_traincut0.5',
                    '../../experiments/DeepASReader/cnn/regl20.000_nhid256_nlayers3_dropout0.2_word2vec_glove_chardim0_train1_subsample-1_seed1_use-feat0_traincut0.8',
                    '../../experiments/DeepASReader/cnn/regl20.000_nhid256_nlayers3_dropout0.2_word2vec_glove_chardim0_train1_subsample-1_seed1_use-feat0'],
        'feat1' :   ['../../experiments/DeepASReader/cnn/regl20.000_nhid256_nlayers3_dropout0.2_word2vec_glove_chardim0_train1_subsample-1_seed1_use-feat1_traincut0.5',
                    '../../experiments/DeepASReader/cnn/regl20.000_nhid256_nlayers3_dropout0.2_word2vec_glove_chardim0_train1_subsample-1_seed1_use-feat1_traincut0.8',
                    '../../experiments/DeepASReader/cnn/regl20.000_nhid256_nlayers3_dropout0.2_word2vec_glove_chardim0_train1_subsample-1_seed1_use-feat1'],
        }
ga_files = {
        'feat0' :   ['../../experiments/GAReaderpp/cnn/regl20.000_nhid256_nlayers3_dropout0.2_word2vec_glove_chardim0_train1_subsample-1_seed1_use-feat0_traincut0.5',
                    '../../experiments/GAReaderpp/cnn/regl20.000_nhid256_nlayers3_dropout0.2_word2vec_glove_chardim0_train1_subsample-1_seed1_use-feat0_traincut0.8',
                    '../../experiments/GAReaderpp/cnn/regl20.000_nhid256_nlayers3_dropout0.2_word2vec_glove_chardim0_train1_subsample-1_seed1_use-feat0'],
        'feat1' :   ['../../experiments/GAReaderpp/cnn/regl20.000_nhid256_nlayers3_dropout0.2_word2vec_glove_chardim0_train1_subsample-1_seed1_use-feat1_traincut0.5',
                    '../../experiments/GAReaderpp/cnn/regl20.000_nhid256_nlayers3_dropout0.2_word2vec_glove_chardim0_train1_subsample-1_seed1_use-feat1_traincut0.8',
                    '../../experiments/GAReaderpp/cnn/regl20.000_nhid256_nlayers3_dropout0.2_word2vec_glove_chardim0_train1_subsample-1_seed2_use-feat1_traincut1.0_gfT.mul'],
        }

gt_lb = np.load('cnn_test.npy')
gt_id = open('cnn_test.ids').read().splitlines()

data_cuts = ['50%','75%','100%']
print 'feature\tdata-cut\tdeepAS acc\tga acc\tmcnemars p'
for k in ['feat0','feat1']:
    for di in range(3):
        fas = as_files[k][di]
        fga = ga_files[k][di]

        # read
        as_pr = np.load(fas+'/test.probs.npy')
        ga_pr = np.load(fga+'/test.probs.npy')
        as_id = open(fas+'/test.ids').read().splitlines()
        ga_id = open(fga+'/test.ids').read().splitlines()

        # compare
        sorted_as = sorted(range(as_pr.shape[0]), key=lambda k:as_id[k])
        sorted_ga = sorted(range(ga_pr.shape[0]), key=lambda k:ga_id[k])
        sorted_gt = sorted(range(gt_lb.shape[0]), key=lambda k:gt_id[k])
        acc_as, acc_ga = 0, 0
        diff = []
        for i in range(as_pr.shape[0]):
            pas = np.argmax(as_pr[sorted_as[i],:])
            pga = np.argmax(ga_pr[sorted_ga[i],:])
            gt = gt_lb[sorted_gt[i]]
            a,g = 0,0
            if pas==gt: a = 1
            if pga==gt: g = 1
            acc_as += a
            acc_ga += g
            if a!=g: diff.append(g)

        # stats
        x2 = sum(diff)
        x1 = len(diff)-x2
        mc_p = binom.cdf(x1,len(diff),0.5) if x1<x2 else binom.cdf(x2,len(diff),0.5)

        # display
        print '%s\t%s\t%.3f\t%.3f\t%.5f' % (k,data_cuts[di],
                float(acc_as)/as_pr.shape[0],float(acc_ga)/as_pr.shape[0],mc_p)
