import numpy as np
import theano
import theano.tensor as T
import lasagne.layers as L
import lasagne
import time

from config import *
from MiniBatchLoader import MiniBatchLoader

def build_lstm_reader(vocab_size, input_var=T.itensor3(), mask_var=T.tensor3()):
    # the input layer
    l_in = L.InputLayer(shape=(BATCH_SIZE, MAX_DOC_LEN + MAX_QRY_LEN, 1), input_var=input_var)
    # the mask layer
    l_mask = L.InputLayer(shape=(BATCH_SIZE, MAX_DOC_LEN + MAX_QRY_LEN), input_var=mask_var)
    # the lookup table for word embeddings
    l_embed = L.EmbeddingLayer(l_in, input_size=vocab_size, output_size=EMBED_DIM)
    # the 1st lstm layer
    l_fwd_1 = L.LSTMLayer(l_embed, NUM_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=l_mask, gradient_steps=GRAD_STEPS)
    # the 2nd lstm layer
    # NOTE: we probably need a vertical dropout layer in between
    # NOTE: not sure if mask should be explicitly applied to the 2nd layer
    l_fwd_2 = L.LSTMLayer(l_fwd_1, NUM_HIDDEN, grad_clipping=GRAD_CLIP, gradient_steps=GRAD_STEPS)
    # slice final states of both lstm layers
    l_fwd_1_slice = L.SliceLayer(l_fwd_1, -1, 1)
    l_fwd_2_slice = L.SliceLayer(l_fwd_2, -1, 1)
    # g will be used to score the words based on their embeddings
    g = L.DenseLayer(L.concat([l_fwd_1_slice, l_fwd_2_slice], axis=1), num_units=EMBED_DIM)
    # W is shared with the embedding layer
    l_out = L.DenseLayer(g, num_units=vocab_size, W = l_embed.W.T, nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

if __name__ == '__main__':

    train_batch_loader = MiniBatchLoader("cnn/questions/training", "vocab.txt")
    val_batch_loader = MiniBatchLoader("cnn/questions/validation", "vocab.txt")
    vocab_size = train_batch_loader.vocab_size

    print("building network ...")
    input_var, target_var, mask_var = T.itensor3('input'), T.ivector('target'), T.matrix('mask')
    network = build_lstm_reader(vocab_size, input_var, mask_var)

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
    d_val, q_val, a_val, m_val = val_batch_loader.next()
    x_val = np.concatenate([d_val, q_val], axis=1)

    for epoch in xrange(NUM_EPOCHS):
        estart = time.time()
        for d_train, q_train, a_train, m_train in train_batch_loader:
            # d, q, a, m: document, query, answer, mask
            x_train = np.concatenate([d_train, q_train], axis=1)
            
            loss_train, acc_train = train_fn(x_train, a_train, m_train)

            print "TRAIN loss=%.4e acc=%.4f Time Elapsed %.1f" % (
                    loss_train, acc_train, time.time()-estart)

        loss_val, acc_val = val_fn(x_val, a_val, m_val)
        print "Epoch %d TRAIN loss=%.4e acc=%.4f VAL loss=%.4e acc=%.4f" % (
                epoch, loss_train, acc_train, loss_val, acc_val)

