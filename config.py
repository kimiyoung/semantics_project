# Minibatch Size
BATCH_SIZE = 32
# Gradient clip threshold
GRAD_CLIP = 10
# Learning rate
LEARNING_RATE = 5e-5
# Number of RNN hidden units
NUM_HIDDEN = 128
# Maximum number of steps in BPTT
GRAD_STEPS = -1
# Number of epochs for training
NUM_EPOCHS = 50
# Dimension of word embedding
EMBED_DIM = 128
# Whether allow skip connection from input to the 2nd layer
SKIP_CONNECT = True
# do validation every VALIDATION_FREQ iterations
VALIDATION_FREQ = 300
# Dropout rate
DROPOUT_RATE = 0.1
# file of word2vec embeddings for initialization
WORD2VEC_PATH = "word2vec_embed.txt"

