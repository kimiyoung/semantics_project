import theano.tensor as T
import lasagne.layers as L
import lasagne
from config import *

# to be finished

def build_model(vocab_size, num_entities, input_var=T.itensor3(), mask_var=T.tensor3(),
        W_init=lasagne.init.Normal()):
    # the input layer
    l_in = L.InputLayer(shape=(None, None, 1), input_var=input_var)
    # the mask layer
    l_mask = L.InputLayer(shape=(None, None), input_var=mask_var)
    # the lookup table for word embeddings
    l_embed = L.EmbeddingLayer(l_in, input_size=vocab_size, output_size=EMBED_DIM, W=W_init)
    # the forward lstm layer
    l_fwd = L.LSTMLayer(l_embed, NUM_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=l_mask,
            gradient_steps=GRAD_STEPS, precompute_input=True)
    # the backward lstm layer
    l_bkd = L.LSTMLayer(l_embed, NUM_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=l_mask,
            gradient_steps=GRAD_STEPS, precompute_input=True, backwards=True)
    # slice final states of both directions
    l_fwd_slice = L.SliceLayer(l_fwd, -1, 1)
    l_bkd_slice = L.SliceLayer(l_bkd, 0, 1)
    # g will be used to score the words based on their embeddings
    g = L.DenseLayer(L.ElemwiseSumLayer([l_fwd_slice, l_bkd_slice]), num_units=EMBED_DIM, nonlinearity=lasagne.nonlinearities.tanh)

    # W is shared with the embedding layer
    l_out = L.DenseLayer(g, num_units=vocab_size, W=l_embed.W.T, nonlinearity=lasagne.nonlinearities.softmax)
    return l_out
