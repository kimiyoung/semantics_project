# Minibatch Size
BATCH_SIZE = 120
# Gradient clip threshold
GRAD_CLIP = 10
# Learning rate
LEARNING_RATE = 0.001
# Number of RNN hidden units
NUM_HIDDEN = 128
# Maximum number of steps in BPTT
GRAD_STEPS = -1
# Number of epochs for training
NUM_EPOCHS = 10
# Dimension of word embedding
EMBED_DIM = 64
# Whether allow skip connection from input to the 2nd layer
SKIP_CONNECT = True
# do validation every VALIDATION_FREQ iterations
VALIDATION_FREQ = 100
# Dropout rate
DROPOUT_RATE = 0.05
# file of word2vec embeddings for initialization
WORD2VEC_PATH = None
# context before or after query (only relevant for Unidirectional)
MODE = 'qca'
# dataset
DATASET = 'cnn/questions'
# num layers
NUM_LAYER = 3
# are there subsets of candidates?
if DATASET=='dailymail/questions' or DATASET=='cnn/questions':
    CANDIDATE_SUBSET = False
elif DATASET=='cbtcn' or DATASET=='cbtne':
    CANDIDATE_SUBSET = True
# l2 regularization
REGULARIZATION=0.008
