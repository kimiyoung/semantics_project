import numpy as np
import theano
import theano.tensor as T
import lasagne.layers as L
import lasagne
import time

from collections import OrderedDict

from config import *
from MiniBatchLoader import MiniBatchLoader

SCALE = 0.1

def init_params(vocab_size):
    params = OrderedDict()

    # lookup table
    params['W_embed'] = theano.shared(np.random.normal(loc=0., scale=SCALE, 
            size=(vocab_size,EMBED_DIM)).astype('float32'), name='W_embed')

    def LSTMparams(params, index, skip_connect=False):
        if skip_connect:
            in_dim = EMBED_DIM+NUM_HIDDEN
        else:
            in_dim = EMBED_DIM

        params['W_lstm%d_xi'%index] = theano.shared(np.random.normal(loc=0., scale=SCALE, 
                size=(in_dim,NUM_HIDDEN)).astype('float32'), name='W_lstm%d_xi'%index)
        params['W_lstm%d_hi'%index] = theano.shared(np.random.normal(loc=0., scale=SCALE, 
                size=(NUM_HIDDEN,NUM_HIDDEN)).astype('float32'), name='W_lstm%d_hi'%index)
        params['W_lstm%d_ci'%index] = theano.shared(np.random.normal(loc=0., scale=SCALE, 
                size=(NUM_HIDDEN,NUM_HIDDEN)).astype('float32'), name='W_lstm%d_ci'%index)
        params['b_lstm%d_i'%index] = theano.shared(np.zeros((NUM_HIDDEN)).astype('float32'), 
                name='b_lstm%d_i'%index)

        params['W_lstm%d_xf'%index] = theano.shared(np.random.normal(loc=0., scale=SCALE, 
                size=(in_dim,NUM_HIDDEN)).astype('float32'), name='W_lstm%d_xf'%index)
        params['W_lstm%d_hf'%index] = theano.shared(np.random.normal(loc=0., scale=SCALE, 
                size=(NUM_HIDDEN,NUM_HIDDEN)).astype('float32'), name='W_lstm%d_hf'%index)
        params['W_lstm%d_cf'%index] = theano.shared(np.random.normal(loc=0., scale=SCALE, 
                size=(NUM_HIDDEN,NUM_HIDDEN)).astype('float32'), name='W_lstm%d_cf'%index)
        params['b_lstm%d_f'%index] = theano.shared(np.zeros((NUM_HIDDEN)).astype('float32'), 
                name='b_lstm%d_f'%index)

        params['W_lstm%d_xo'%index] = theano.shared(np.random.normal(loc=0., scale=SCALE, 
                size=(in_dim,NUM_HIDDEN)).astype('float32'), name='W_lstm%d_xo'%index)
        params['W_lstm%d_ho'%index] = theano.shared(np.random.normal(loc=0., scale=SCALE, 
                size=(NUM_HIDDEN,NUM_HIDDEN)).astype('float32'), name='W_lstm%d_ho'%index)
        params['W_lstm%d_co'%index] = theano.shared(np.random.normal(loc=0., scale=SCALE, 
                size=(NUM_HIDDEN,NUM_HIDDEN)).astype('float32'), name='W_lstm%d_co'%index)
        params['b_lstm%d_o'%index] = theano.shared(np.zeros((NUM_HIDDEN)).astype('float32'), 
                name='b_lstm%d_o'%index)

        params['W_lstm%d_xc'%index] = theano.shared(np.random.normal(loc=0., scale=SCALE, 
                size=(in_dim,NUM_HIDDEN)).astype('float32'), name='W_lstm%d_xc'%index)
        params['W_lstm%d_hc'%index] = theano.shared(np.random.normal(loc=0., scale=SCALE, 
                size=(NUM_HIDDEN,NUM_HIDDEN)).astype('float32'), name='W_lstm%d_hc'%index)
        params['b_lstm%d_c'%index] = theano.shared(np.zeros((NUM_HIDDEN)).astype('float32'), 
                name='b_lstm%d_c'%index)

        return params

    # LSTM layers
    params = LSTMparams(params, 1)
    params = LSTMparams(params, 2, skip_connect=SKIP_CONNECT)

    # dense layer
    params['W_dense'] = theano.shared(np.random.normal(loc=0., scale=SCALE, 
            size=(2*NUM_HIDDEN,EMBED_DIM)).astype('float32'), name='W_dense')
    params['b_dense'] = theano.shared(np.zeros((EMBED_DIM,)).astype('float32'), 
            name='b_dense')

    return params

def build_lstm_reader(params, vocab_size, input_var=T.itensor3(), mask_var=T.tensor3(), skip_connect=True):
    # the input layer
    l_in = L.InputLayer(shape=(None, None, 1), input_var=input_var)
    # the mask layer
    l_mask = L.InputLayer(shape=(None, None), input_var=mask_var)
    # the lookup table of word embeddings
    l_embed = L.EmbeddingLayer(l_in, vocab_size, EMBED_DIM)

    # the 1st lstm layer
    in_gate = L.Gate(W_in=params['W_lstm1_xi'], W_hid=params['W_lstm1_hi'], 
            W_cell=params['W_lstm1_ci'], b=params['b_lstm1_i'], 
            nonlinearity=lasagne.nonlinearities.sigmoid)
    forget_gate = L.Gate(W_in=params['W_lstm1_xf'], W_hid=params['W_lstm1_hf'], 
            W_cell=params['W_lstm1_cf'], b=params['b_lstm1_f'], 
            nonlinearity=lasagne.nonlinearities.sigmoid)
    out_gate = L.Gate(W_in=params['W_lstm1_xo'], W_hid=params['W_lstm1_ho'], 
            W_cell=params['W_lstm1_co'], b=params['b_lstm1_o'], 
            nonlinearity=lasagne.nonlinearities.sigmoid)
    cell_gate = L.Gate(W_in=params['W_lstm1_xc'], W_hid=params['W_lstm1_hc'], 
            W_cell=None, b=params['b_lstm1_c'], 
            nonlinearity=lasagne.nonlinearities.tanh)
    l_fwd_1 = L.LSTMLayer(l_embed, NUM_HIDDEN, ingate=in_gate, forgetgate=forget_gate,
            cell=cell_gate, outgate=out_gate, peepholes=True, grad_clipping=GRAD_CLIP, 
            mask_input=l_mask, gradient_steps=GRAD_STEPS, precompute_input=True)

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

    # the 2nd lstm layer
    in_gate = L.Gate(W_in=params['W_lstm2_xi'], W_hid=params['W_lstm2_hi'], 
            W_cell=params['W_lstm2_ci'], b=params['b_lstm2_i'], 
            nonlinearity=lasagne.nonlinearities.sigmoid)
    forget_gate = L.Gate(W_in=params['W_lstm2_xf'], W_hid=params['W_lstm2_hf'], 
            W_cell=params['W_lstm2_cf'], b=params['b_lstm2_f'], 
            nonlinearity=lasagne.nonlinearities.sigmoid)
    out_gate = L.Gate(W_in=params['W_lstm2_xo'], W_hid=params['W_lstm2_ho'], 
            W_cell=params['W_lstm2_co'], b=params['b_lstm2_o'], 
            nonlinearity=lasagne.nonlinearities.sigmoid)
    cell_gate = L.Gate(W_in=params['W_lstm2_xc'], W_hid=params['W_lstm2_hc'], 
            W_cell=None, b=params['b_lstm2_c'], 
            nonlinearity=lasagne.nonlinearities.tanh)
    l_fwd_2 = L.LSTMLayer(to_next_layer, NUM_HIDDEN, ingate=in_gate, forgetgate=forget_gate,
            cell=cell_gate, outgate=out_gate, peepholes=True, grad_clipping=GRAD_CLIP, 
            mask_input=l_mask, gradient_steps=GRAD_STEPS, precompute_input=True)

    # slice final states of both lstm layers
    l_fwd_1_slice = L.SliceLayer(l_fwd_1, -1, 1)
    l_fwd_2_slice = L.SliceLayer(l_fwd_2, -1, 1)

    # g will be used to score the words based on their embeddings
    g = L.DenseLayer(L.concat([l_fwd_1_slice, l_fwd_2_slice], axis=1), num_units=EMBED_DIM, 
            W=params['W_dense'], b=params['b_dense'], nonlinearity=None)
    # W is shared with the embedding layer
    l_out = L.DenseLayer(g, num_units=vocab_size, W=params['W_embed'].T, b=None, 
            nonlinearity=lasagne.nonlinearities.softmax)
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
    params = init_params(vocab_size)
    input_var, target_var, mask_var = T.itensor3('input'), T.ivector('target'), T.matrix('mask')
    network = build_lstm_reader(params, vocab_size, input_var, mask_var, skip_connect=SKIP_CONNECT)

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

