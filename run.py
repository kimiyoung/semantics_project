import train
import test
import argparse

# parse arguments
parser = argparse.ArgumentParser()
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
parser.add_argument('--word2vec', dest='word2vec', type=str, default='word2vec_embed.txt',
        help='File with word embeddings. Should have header with number and size of embeddings.')
parser.add_argument('--dataset', dest='dataset', type=str, default='wdw',
        help='Dataset - (cnn/questions || dailymail/questions || cbtcn || cbtne || wdw)')
parser.add_argument('--train_emb', dest='train_emb', type=int, default=0,
        help='Tune word embeddings - (0-No, 1-Yes)')
parser.add_argument('--subsample', dest='subsample', type=int, default=-1,
        help='Sample window size around candidates. (-1-no sampling)')
args = parser.parse_args()
params=vars(args)

save_path = ('exp_gapp_pr/'+params['dataset'].split('/')[0]+'/reg%s'%params['regularizer']+
        '%.3f'%params['lambda']+'_nhid%d'%params['nhidden']+'_nlayers%d'%params['nlayers']+
        '_dropout%.1f'%params['dropout']+'_%s'%params['word2vec'].split('/')[-1].split('.')[0]+
        '_train%d'%params['train_emb']+'_subsample%d'%params['subsample']+'/')

# train
train.main(save_path, params['regularizer'], params['lambda'], params['nhidden'],
        params['dropout'], params['word2vec'], params['dataset'], params['nlayers'],
        params['train_emb'], params['subsample'])

# test
test.main(save_path, params['regularizer'], params['lambda'], params['nhidden'],
        params['dropout'], params['word2vec'], params['dataset'], params['nlayers'],
        params['train_emb'], params['subsample'])
