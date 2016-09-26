import train
import test
import argparse
import os

# parse arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', dest='model', type=str, default='GAReaderpp_prior',
        help='base model - (GAReader || GAReaderpp_prior)')
parser.add_argument('--mode', dest='mode', type=int, default=0,
        help='run mode - (0-train+test, 1-train only, 2-test only)')
parser.add_argument('--regularizer', dest='regularizer', type=str, default='l2',
        help='l2 or l1 norm for regularizing word embeddings')
parser.add_argument('--lambda', dest='lambda', type=float, default=0.,
        help='weight of regularization')
parser.add_argument('--nhidden', dest='nhidden', type=int, default=256,
        help='GRU hiddent state size')
parser.add_argument('--nlayers', dest='nlayers', type=int, default=2,
        help='Number of reader layers')
parser.add_argument('--dropout', dest='dropout', type=float, default=0.3,
        help='Dropout rate')
parser.add_argument('--word2vec', dest='word2vec', type=str, default=None,
        help='File with word embeddings. Should have header with number and size of embeddings.')
parser.add_argument('--dataset', dest='dataset', type=str, default='wdw',
        help='Dataset - (cnn/questions || dailymail/questions || cbtcn || cbtne || wdw)')
parser.add_argument('--train_emb', dest='train_emb', type=int, default=0,
        help='Tune word embeddings - (0-No, 1-Yes)')
parser.add_argument('--subsample', dest='subsample', type=int, default=-1,
        help='Sample window size around candidates. (-1-no sampling)')
args = parser.parse_args()
params=vars(args)

w2v_filename = params['word2vec'].split('/')[-1].split('.')[0] if params['word2vec'] else 'None'
save_path = ('experiments/'+params['model']+'/'+params['dataset'].split('/')[0]+
        '/reg%s'%params['regularizer']+
        '%.3f'%params['lambda']+'_nhid%d'%params['nhidden']+'_nlayers%d'%params['nlayers']+
        '_dropout%.1f'%params['dropout']+'_%s'%w2v_filename+
        '_train%d'%params['train_emb']+'_subsample%d'%params['subsample']+'/')

if not os.path.exists(save_path): os.makedirs(save_path)

# train
if params['mode']<2:
    train.main(save_path, params['regularizer'], params['lambda'], params['nhidden'],
            params['dropout'], params['word2vec'], params['dataset'], params['nlayers'],
            params['train_emb'], params['subsample'], params['model'])

# test
if params['mode']!=1:
    test.main(save_path, params['regularizer'], params['lambda'], params['nhidden'],
            params['dropout'], params['word2vec'], params['dataset'], params['nlayers'],
            params['train_emb'], params['subsample'], params['model'])
