import theano.tensor as T
import lasagne.layers as L
import lasagne
from config import *

# to be finished

def build_model(vocab_size, doc_var, qry_var, doc_mask_var, qry_mask_var, W_init=lasagne.init.Normal()):

    l_doc_in = L.InputLayer(shape=(None, None, 1), input_var=doc_var)
    l_qry_in = L.InputLayer(shape=(None, None, 1), input_var=qry_var)

    l_doc_embed = L.EmbeddingLayer(l_doc_in, vocab_size, EMBED_DIM, W=W_init)
    l_qry_embed = L.EmbeddingLayer(l_qry_in, vocab_size, EMBED_DIM, W=l_doc_embed.W)

    l_doc_mask = L.InputLayer(shape=(None, None), input_var=doc_mask_var)
    l_qry_mask = L.InputLayer(shape=(None, None), input_var=qry_mask_var)

    l_doc_fwd = L.LSTMLayer(l_doc_embed, NUM_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=l_doc_mask, gradient_steps=GRAD_STEPS, precompute_input=True)
    l_doc_bkd = L.LSTMLayer(l_doc_embed, NUM_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=l_doc_mask, gradient_steps=GRAD_STEPS, precompute_input=True, backwards=True)
    l_qry_fwd = L.LSTMLayer(l_qry_embed, NUM_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=l_qry_mask, gradient_steps=GRAD_STEPS, precompute_input=True)
    l_qry_bkd = L.LSTMLayer(l_qry_embed, NUM_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=l_qry_mask, gradient_steps=GRAD_STEPS, precompute_input=True, backwards=True)

    l_doc_fwd_slice = L.SliceLayer(l_doc_fwd, -1, 1)
    l_doc_bkd_slice = L.SliceLayer(l_doc_bkd, 0, 1)
    l_qry_fwd_slice = L.SliceLayer(l_qry_fwd, -1, 1)
    l_qry_bkd_slice = L.SliceLayer(l_qry_bkd, 0, 1)

    r = L.DenseLayer(L.ElemwiseSumLayer([l_doc_fwd_slice, l_doc_bkd_slice]), num_units=NUM_HIDDEN, nonlinearity=lasagne.nonlinearities.tanh)
    u = L.DenseLayer(L.ElemwiseSumLayer([l_qry_fwd_slice, l_qry_bkd_slice]), num_units=NUM_HIDDEN, nonlinearity=lasagne.nonlinearities.tanh)

    g = L.DenseLayer(L.concat([r,u], axis=1), num_units=EMBED_DIM, W=lasagne.init.GlorotNormal(), nonlinearity=lasagne.nonlinearities.tanh)

    l_out = L.DenseLayer(g, num_units=vocab_size, W=l_doc_embed.W.T, nonlinearity=lasagne.nonlinearities.softmax, b=None)

    return l_out

