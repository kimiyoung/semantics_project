import train
import test
import argparse
import os
import numpy as np
import random
import sys

from config import *

# parse arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', dest='model', type=str, default='GAMage',
        help='base model - (GAReader || GAReaderpp || StanfordAR || DeepASReader)')
parser.add_argument('--mode', dest='mode', type=int, default=0,
        help='run mode - (0-train only, 1-test only, 2-val only)')
parser.add_argument('--regularizer', dest='regularizer', type=str, default='l2',
        help='l2 or l1 norm for regularizing word embeddings')
parser.add_argument('--lambda', dest='lambda', type=float, default=0.,
        help='weight of regularization')
parser.add_argument('--nlayers', dest='nlayers', type=int, default=3,
        help='Number of reader layers')
parser.add_argument('--dataset', dest='dataset', type=str, default='wdw',
        help='Dataset - (cnn/questions || dailymail/questions || cbtcn || cbtne || wdw)')
parser.add_argument('--subsample', dest='subsample', type=int, default=-1,
        help='Sample window size around candidates. (-1-no sampling)')
parser.add_argument('--seed', dest='seed', type=int, default=1,
        help='Seed for different experiments with same settings')
parser.add_argument('--train_cut', dest='train_cut', type=float, default=1.0,
        help='Cut training data size by factor (default - no cut)')
parser.add_argument('--gating_fn', dest='gating_fn', type=str, default='T.mul',
        help='Gating function (T.mul || Tsum || Tconcat)')
parser.add_argument('--concat', dest='concat', action='store_true')
parser.add_argument('--no-concat', dest='concat', action='store_false')
parser.set_defaults(concat=False)
parser.set_defaults(coref=False)
args = parser.parse_args()
cmd = vars(args)
params = get_params(cmd['dataset'])
params.update(cmd)

np.random.seed(params['seed'])
random.seed(params['seed'])

# number of coref
#params['num_coref'] = int(open(params['dataset']+'/num_coref.txt').read().split('\t')[0])

# save directory
w2v_filename = params['word2vec'].split('/')[-1].split('.')[0] if params['word2vec'] else 'None'
save_path = ('crfreader_experiments_babi-v2/'+params['model']+'/'+params['dataset'].split('/')[0]+
        '/m'+
        '_lr%.4f'%LEARNING_RATE+
        '_bsize%d'%BATCH_SIZE+
        '_anneal%d'%ANNEAL+
        '_stop%d'%int(STOPPING)+
        #'reg%s'%params['regularizer']+
        #'%.3f'%params['lambda']+
        '_nhid%d'%params['nhidden']+'_nlayers%d'%params['nlayers']+
        '_dropout%.1f'%params['dropout']+'_%s'%w2v_filename+'_chardim%d'%params['char_dim']+
        '_train%d'%params['train_emb']+
        #'_subsample%d'%params['subsample']+
        '_seed%d'%params['seed']+'_use-feat%d'%params['use_feat']+
        #'_traincut%.1f'%params['train_cut']+'_gf%s'%params['gating_fn']+
        #'_rdims%s'%'.'.join([str(rd) for rd in params['relation_dims']])+'/')
        '_concat%d'%int(params['concat'])+
        '_chains%d'%params['max_chains']+
        '_rdims%d'%params['relation_dims'])
if not os.path.exists(save_path): os.makedirs(save_path)
#else: sys.exit()

# train
if params['mode']==0:
    train.main(save_path, params)
# test
elif params['mode']==1:
    test.main(save_path, params)
elif params['mode']==2:
    test.main(save_path, params, mode='validation')
