# Minibatch Size
BATCH_SIZE = 32
# Gradient clip threshold
GRAD_CLIP = 50
# Learning rate
LEARNING_RATE = 5e-4
# Number of RNN hidden units
NUM_HIDDEN = 128
# Maximum number of steps in BPTT
GRAD_STEPS = -1
# Special symbols indicating the starting/ending points of a query
SYMB_BEGIN, SYMB_END = "@begin", "@end"
# Number of epochs for training
NUM_EPOCHS = 20
# Dimension of word embedding
EMBED_DIM = 128
# Whether allow skip connection from input to the 2nd layer
SKIP_CONNECT = True
