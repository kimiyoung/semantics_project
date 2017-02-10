# Minibatch Size
BATCH_SIZE = 32
# Gradient clip threshold
GRAD_CLIP = 10
# Learning rate
LEARNING_RATE = 0.01
# Maximum number of steps in BPTT
GRAD_STEPS = -1
# Number of epochs for training
NUM_EPOCHS = 30
# do validation every VALIDATION_FREQ iterations
VALIDATION_FREQ = 100
# maximum word length for character model
MAX_WORD_LEN = 10
# annealing every x epochs
ANNEAL = 4
# early stopping
STOPPING = False

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
    elif dataset=='babi' or dataset=='babi-3-1k-pcrf' or dataset=='babi-3-1k-orig':
        return babi_params
    elif dataset=='babi-clean':
        return babiclean_params
    elif dataset=='debug' or dataset=='wdw_debug':
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
        'char_dim'  :   25,
        'dropout'   :   0.3,
        'word2vec'  :   'word2vec/word2vec_glove.txt',
        'train_emb' :   0,
        'use_feat'  :   1,
        'num_coref' :   76,
        'coref_dim' :   0,
        }

cnn_params = {
        'nhidden'   :   256,
        'char_dim'  :   0,
        'dropout'   :   0.2,
        'word2vec'  :   'word2vec/word2vec_glove.txt',
        'train_emb' :   0,
        'use_feat'  :   1,
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
        'nhidden'   :   224,
        'char_dim'  :   0,
        'dropout'   :   0.2,
        'word2vec'  :   'word2vec/word2vec_glove.txt',
        'train_emb' :   1,
        'use_feat'  :   0,
        'num_coref' :   12,
        'coref_dim' :   32,
        }

babi_params = {
        'nhidden'   :   32,
        'char_dim'  :   0,
        'dropout'   :   0.1,
        'word2vec'  :   None,
        'train_emb' :   1,
        'use_feat'  :   0,
        'num_coref' :   4,
        'coref_dim' :   32,
        }

babiclean_params = {
        'nhidden'   :   112,
        'char_dim'  :   0,
        'dropout'   :   0.3,
        'word2vec'  :   None,
        'train_emb' :   1,
        'use_feat'  :   0,
        'num_coref' :   4,
        'coref_dim' :   16,
        }

debug_params = {
        'nhidden'   :   8,
        'char_dim'  :   0,
        'dropout'   :   0.2,
        'word2vec'  :   None,
        'train_emb' :   1,
        'use_feat'  :   0,
        'num_coref' :   76,
        'coref_dim' :   2,
        }
