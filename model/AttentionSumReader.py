import theano
import theano.tensor as T
import lasagne.layers as L
import lasagne
import numpy as np
import cPickle as pickle
from config import *

class Model:

    def __init__(self, vocab_size, W_init=lasagne.init.GlorotNormal()):

        doc_var, query_var, docmask_var, qmask_var, target_var = T.itensor3('doc'), T.itensor3('quer'), T.imatrix('doc_mask'), T.imatrix('q_mask'), T.ivector('ans')

        self.network = self.build_network(vocab_size, doc_var, query_var, docmask_var, qmask_var, W_init)
        predicted_probs = L.get_output(self.network)
        predicted_probs_val = L.get_output(self.network, deterministic=True)

        loss_fn = T.nnet.categorical_crossentropy(predicted_probs, target_var).mean()
        eval_fn = lasagne.objectives.categorical_accuracy(predicted_probs, target_var).mean()

        loss_fn_val = T.nnet.categorical_crossentropy(predicted_probs_val, target_var).mean()
        eval_fn_val = lasagne.objectives.categorical_accuracy(predicted_probs_val, target_var).mean()

        params = L.get_all_params(self.network, trainable=True)
        
        updates = lasagne.updates.rmsprop(loss_fn, params, rho=0.95, learning_rate=LEARNING_RATE)
        updates_with_momentum = lasagne.updates.apply_momentum(updates, params=params)

        self.train_fn = theano.function([input_var, target_var, mask_var], [loss_fn, eval_fn, predicted_probs], updates=updates_with_momentum)
        self.validate_fn = theano.function([input_var, target_var, mask_var], [loss_fn_val, eval_fn_val, predicted_probs_val])

    def train(self, d, q, a, m_d, m_q):
        x = np.concatenate([d, q], axis=1)
        m = np.concatenate([m_d, m_q], axis=1)
        return self.train_fn(x, a, m)

    def validate(self, d, q, a, m_d, m_q):
        x = np.concatenate([d, q], axis=1)
        m = np.concatenate([m_d, m_q], axis=1)
        return self.validate_fn(x, a, m)

    def build_network(self, vocab_size, doc_var, query_var, docmask_var, qmask_var, W_init):

        l_docin = L.InputLayer(shape=(None,None,1), input_var=doc_var)
        l_qin = L.InputLayer(shape=(None,None,1), input_var=query_var)
        l_docmask = L.InputLayer(shape=(None,None), input_var=docmask_var)
        l_qmask = L.InputLayer(shape=(None,None), input_var=qmask_var)
        l_docembed = L.EmbeddingLayer(l_docin, input_size=vocab_size, output_size=EMBED_DIM, W=W_init)
        l_qembed = L.EmbeddingLayer(l_qin, input_size=vocab_size, output_size=EMBED_DIM, W=W_init)

        l_fwd_doc = L.GRULayer(l_docembed, NUM_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=l_mask, gradient_steps=GRAD_STEPS, precompute_input=True)
        l_bkd_doc = L.GRULayer(l_docembed, NUM_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=l_mask, gradient_steps=GRAD_STEPS, precompute_input=True, backwards=True)

        l_doc = L.concat([l_fwd_doc, l_bkd_doc], axis=2)

        l_fwd_q = L.GRULayer(l_qembed, NUM_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=l_mask, gradient_steps=GRAD_STEPS, precompute_input=True)
        l_bkd_q = L.GRULayer(l_qembed, NUM_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=l_mask, gradient_steps=GRAD_STEPS, precompute_input=True, backwards=True)

        l_fwd_q_slice = L.SliceLayer(l_fwd_q, -1, 1)
        l_bkd_q_slice = L.SliceLayer(l_bkd_q, 0, 1)
        l_q = L.ConcatLayer([l_fwd_q_slice, l_bkd_q_slice])

        d = L.get_output(l_doc) # B x N x D
        q = L.get_output(l_q) # B x D
        q_res = T.tile(q, (1,

        p = 

        l_fwd_2_slice = L.SliceLayer(l_fwd_2, -1, 1)
        l_bkd_2_slice = L.SliceLayer(l_bkd_2, 0, 1)
        y_2 = L.ElemwiseSumLayer([l_fwd_2_slice, l_bkd_2_slice])

        y = L.concat([y_1, y_2], axis=1)
        g = L.DenseLayer(y, num_units=EMBED_DIM, nonlinearity=lasagne.nonlinearities.tanh)
        l_out = L.DenseLayer(g, num_units=vocab_size, W=l_embed.W.T, nonlinearity=lasagne.nonlinearities.softmax)

        return l_out

    def load_model(self, load_path):
        with open(load_path, 'r') as f:
            data = pickle.load(f)
        L.set_all_param_values(self.network, data)

    def save_model(self, save_path):
        data = L.get_all_param_values(self.network)
        with open(save_path, 'w') as f:
            pickle.dump(data, f)

