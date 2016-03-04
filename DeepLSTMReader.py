import numpy as np
import theano
import theano.tensor as T
import lasagne
import time

from config import *
from MiniBatchLoader import MiniBatchLoader

EMBED_DIM=128

if __name__ == '__main__':

    train_batch_loader = MiniBatchLoader("cnn/questions/training", "vocab.txt")
    val_batch_loader = MiniBatchLoader("cnn/questions/validation", "vocab.txt")
    vocab_size = len(train_batch_loader.dictionary)

    print("building network ...")
    input_values = T.itensor3()
    l_in = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=input_values)

    l_in_embedded = lasagne.layers.EmbeddingLayer(l_in, input_size=vocab_size, output_size=EMBED_DIM)

    l_mask = lasagne.layers.InputLayer(shape=(BATCH_SIZE, MAX_DOC_LEN + MAX_QRY_LEN))

    l_forward_1 = lasagne.layers.LSTMLayer(
        l_in_embedded, NUM_HIDDEN, grad_clipping=GRAD_CLIP,
        mask_input=l_mask,
        nonlinearity=lasagne.nonlinearities.tanh,
        gradient_steps=GRAD_STEPS)

    # XXX: might need a vertical dropout layer in between
    l_forward_2 = lasagne.layers.LSTMLayer(
        l_forward_1, NUM_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh,
        gradient_steps=GRAD_STEPS)

    l_forward_slice = lasagne.layers.SliceLayer(l_forward_2, -1, 1)
    l_out = lasagne.layers.DenseLayer(l_forward_slice, num_units=vocab_size,
            W=lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)

    target_values = T.ivector('target_output')
    network_output = lasagne.layers.get_output(l_out)
    cost = T.nnet.categorical_crossentropy(network_output, target_values).mean()
    all_params = lasagne.layers.get_all_params(l_out, trainable=True)

    print("computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)

    print("compiling functions ...")
    train = theano.function([l_in.input_var, target_values, l_mask.input_var], cost, updates=updates, allow_input_downcast=True)
    compute_cost = theano.function([l_in.input_var, target_values, l_mask.input_var], cost, allow_input_downcast=True)

    # use a small batch for validation
    d_val, q_val, a_val, m_val = val_batch_loader.next()
    x_val = np.concatenate([d_val, q_val], axis=1)

    print("training ...")
    for _ in xrange(NUM_EPOCHS):

        # d, q, a, m: document, query, answer, mask
        d_train, q_train, a_train, m_train = train_batch_loader.next()
        x_train = np.concatenate([d_train, q_train], axis=1)
        
        cost_train = train(x_train, a_train, m_train)
        cost_val = compute_cost(x_val, a_val, m_val)

        print "train_cost=%.5e val_cost=%.5e" % (cost_train, cost_val)

