# Minibatch Size
BATCH_SIZE = 64
# Gradient clip threshold
GRAD_CLIP = 10
# Learning rate
LEARNING_RATE = 0.0005
# Maximum number of steps in BPTT
GRAD_STEPS = -1
# Number of epochs for training
NUM_EPOCHS = 10
# do validation every VALIDATION_FREQ iterations
VALIDATION_FREQ = 500
# maximum word length for character model
MAX_WORD_LEN = 10
# annealing every x epochs
ANNEAL = 1
# early stopping
STOPPING = True

# dataset params
def get_params(dataset):
    if dataset=='cbtcn':
        return cbtcn_params
    elif dataset=='wdw' or dataset=='wdw_relaxed':
        return wdw_params
    elif dataset=='cnn':
        return cnn_params
    elif dataset=='dailymail':
        return dailymail_params
    elif dataset=='cbtne':
        return cbtne_params
    elif dataset=='lambada':
        return lambada_params
    elif dataset.startswith('babi-orig-1k-') or dataset.startswith('babi-pron-1k-'):
        return babi_params
    elif dataset=='babi-3-1k-mix' or dataset=='babi-3-10k-mix':
        return babimix_params
    elif dataset=='lambada_debug' or dataset=='wdw_debug':
        return debug_params
    else:
        raise ValueError("Dataset %s not found"%dataset)

cbtcn_params = {
        'nhidden'   :   128,
        'char_dim'  :   25,
        'dropout'   :   0.4,
        'word2vec'  :   'word2vec/word2vec_glove.txt',
        'train_emb' :   0,
        'use_feat'  :   0,
        }

wdw_params = {
        'nhidden'   :   128,
        'char_dim'  :   0,
        'dropout'   :   0.3,
        'word2vec'  :   'word2vec/word2vec_glove.txt',
        'train_emb' :   0,
        'use_feat'  :   1,
        #'num_coref' :   76,
        #'coref_dim' :   0,
        'max_chains' :   5,
        'num_relations' :   2,
        'relation_dims' :   64,
        }

cnn_params = {
        'nhidden'   :   240,
        'char_dim'  :   0,
        'dropout'   :   0.2,
        'word2vec'  :   'word2vec/word2vec_glove.txt',
        'train_emb' :   1,
        'use_feat'  :   0,
        'num_coref' :   100,
        'coref_dim' :   16,
        }

dailymail_params = {
        'nhidden'   :   256,
        'char_dim'  :   0,
        'dropout'   :   0.1,
        'word2vec'  :   'word2vec/word2vec_glove.txt',
        'train_emb' :   0,
        'use_feat'  :   1,
        }

cbtne_params = {
        'nhidden'   :   128,
        'char_dim'  :   25,
        'dropout'   :   0.4,
        'word2vec'  :   'word2vec/word2vec_glove.txt',
        'train_emb' :   0,
        'use_feat'  :   1,
        }

lambada_params = {
        'nhidden'   :   256,
        'char_dim'  :   0,
        'dropout'   :   0.2,
        'word2vec'  :   'word2vec/word2vec_glove.txt',
        'train_emb' :   1,
        'use_feat'  :   0,
        #'num_coref' :   14,
        #'coref_dim' :   0,
        'max_chains'    :   16,
        'num_relations' :   10,
        'relation_dims' :   25,
        }

babi_params = {
        'nhidden'   :   64,
        'char_dim'  :   0,
        'dropout'   :   0.1,
        'word2vec'  :   None,
        'train_emb' :   1,
        'use_feat'  :   0,
        #'coref_dim' :   0,
        'max_chains'    :   14,
        'num_relations' :   2,
        'relation_dims' :   32,
        }

babimix_params = {
        'nhidden'   :   32,
        'char_dim'  :   0,
        'dropout'   :   0.1,
        'word2vec'  :   None,
        'train_emb' :   1,
        'use_feat'  :   0,
        'num_coref' :   20,
        'coref_dim' :   32,
        }

debug_params = {
        'nhidden'   :   8,
        'char_dim'  :   0,
        'dropout'   :   0.2,
        'word2vec'  :   None,
        'train_emb' :   1,
        'use_feat'  :   0,
        'max_chains' :   25,
        'num_relations' :   64,
        'relation_dims' :   2,
        }
