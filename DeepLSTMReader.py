import numpy as np
import theano
import theano.tensor as T
import lasagne.layers as L
import lasagne
import time

from config import *
from MiniBatchLoader import MiniBatchLoader

def build_lstm_reader(vocab_size, input_var=T.itensor3(), mask_var=T.tensor3(), skip_connect=True):
    # the input layer
    l_in = L.InputLayer(shape=(None, None, 1), input_var=input_var)
    # the mask layer
    l_mask = L.InputLayer(shape=(None, None), input_var=mask_var)
    # the lookup table of word embeddings
    l_embed = L.EmbeddingLayer(l_in, vocab_size, EMBED_DIM)

    # the 1st lstm layer
    l_fwd_1 = L.LSTMLayer(l_embed, NUM_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=l_mask,
            gradient_steps=GRAD_STEPS, precompute_input=True)

    # the 2nd lstm layer
    if skip_connect:
        # construct skip connection from the lookup table to the 2nd layer
        batch_size, seq_len, _ = input_var.shape
        # concatenate the last dimension of l_fwd_1 and embed
        l_fwd_1_shp = L.ReshapeLayer(l_fwd_1, (-1, NUM_HIDDEN))
        l_embed_shp = L.ReshapeLayer(l_embed, (-1, EMBED_DIM))
        to_next_layer = L.ReshapeLayer(L.concat([l_fwd_1_shp, l_embed_shp], axis=1),
                (batch_size, seq_len, NUM_HIDDEN+EMBED_DIM)) 
    else:
        to_next_layer = l_fwd_1

    l_fwd_2 = L.LSTMLayer(to_next_layer, NUM_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=l_mask,
            gradient_steps=GRAD_STEPS, precompute_input=True)

    # slice final states of both lstm layers
    l_fwd_1_slice = L.SliceLayer(l_fwd_1, -1, 1)
    l_fwd_2_slice = L.SliceLayer(l_fwd_2, -1, 1)

    # g will be used to score the words based on their embeddings
    g = L.DenseLayer(L.concat([l_fwd_1_slice, l_fwd_2_slice], axis=1), num_units=EMBED_DIM)
    # W is shared with the embedding layer
    l_out = L.DenseLayer(g, num_units=vocab_size, W=l_embed.W.T, nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

if __name__ == '__main__':

    print("loading training files (may take up to 2 min) ...")
    # loading all files in the training set can be slow. I'll optimize it when I have time.
    # alternatively, we may just store the train_batch_loader in disk via serialization.
    train_batch_loader = MiniBatchLoader("cnn/questions/training", "vocab.txt")

    print("loading validation files ...")
    val_batch_loader = MiniBatchLoader("cnn/questions/validation", "vocab.txt")
    vocab_size = train_batch_loader.vocab_size

    print("building network ...")
    input_var, target_var, mask_var = T.itensor3('input'), T.ivector('target'), T.matrix('mask')
    network = build_lstm_reader(vocab_size, input_var, mask_var, skip_connect=SKIP_CONNECT)

    print("computing updates ...")
    prediction = L.get_output(network)
    loss = T.nnet.categorical_crossentropy(prediction, target_var).mean()
    params = L.get_all_params(network, trainable=True)
    updates = lasagne.updates.adagrad(loss, params, learning_rate=LEARNING_RATE)

    print("compiling functions ...")
    acc = lasagne.objectives.categorical_accuracy(prediction, target_var).mean()
    train_fn = theano.function([input_var, target_var, mask_var], [loss, acc], updates=updates)
    val_fn = theano.function([input_var, target_var, mask_var], [loss, acc])

    print("training ...")
    # pick a small batch for validation
    d_val, q_val, a_val, md_val, mq_val, _ = val_batch_loader.next()
    x_val = np.concatenate([q_val, d_val], axis=1)
    m_val = np.concatenate([mq_val, md_val], axis=1)

    for epoch in xrange(NUM_EPOCHS):
        estart = time.time()
        for d, q, a, md, mq, _ in train_batch_loader:
            x = np.concatenate([q, d], axis=1)
            m = np.concatenate([mq, md], axis=1)
            loss, acc = train_fn(x, a, m)
            print "TRAIN loss=%.4e acc=%.4f Time Elapsed %.1f" % (loss, acc, time.time()-estart)

        loss_val, acc_val = val_fn(x_val, a_val, m_val)
        print "Epoch %d TRAIN loss=%.4e acc=%.4f VAL loss=%.4e acc=%.4f" % (epoch, loss, acc, loss_val, acc_val)

