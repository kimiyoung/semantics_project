import theano
import theano.tensor as T
import lasagne.layers as L
import lasagne
import numpy as np
from config import *

class Model:

    def __init__(self, vocab_size, W_init=lasagne.init.GlorotNormal()):

        input_var, mask_var, target_var = T.itensor3('dq_pair'), T.imatrix('dq_mask'), T.ivector('ans')

        self.network = self.build_network(vocab_size, input_var, mask_var, W_init)
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
        x, m = self.mix_document_quary(d, q, m_d, m_q)
        return self.train_fn(x, a, m)

    def mix_document_quary(self, d, q, m_d, m_q):

        period_ix = 4091 # XXX

        n, nd, _ = d.shape
        _, nq, _ = q.shape

        x = np.zeros(shape=(n,nd+nq,1))
        m = np.zeros(shape=(n,nd+nq))

        for i, doc in enumerate(d):
            # positions where periods occur
            positions = [p for p, w in enumerate(doc) if w[0] == period_ix]
            if len(positions) == 0:
                k = 0
            else:
                # randomly pick a position to insert
                k = np.random.choice(positions) + 1
            x[i] = np.concatenate((d[i,:k,:], q[i], d[i,k:,:]), axis=0)
            m[i] = np.concatenate((m_d[i,:k], m_q[i], m_d[i,k:]), axis=0)

        return x.astype('int32'), m.astype('int32')

    def validate(self, d, q, a, m_d, m_q):
        x = np.concatenate([d, q], axis=1)
        m = np.concatenate([m_d, m_q], axis=1)
        return self.validate_fn(x, a, m)

    def build_network(self, vocab_size, input_var, mask_var, W_init):

        l_in = L.InputLayer(shape=(None,None,1), input_var=input_var)
        l_mask = L.InputLayer(shape=(None,None), input_var=mask_var)
        l_embed = L.EmbeddingLayer(l_in, input_size=vocab_size, output_size=EMBED_DIM, W=W_init)

        l_fwd_1 = L.LSTMLayer(l_embed, NUM_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=l_mask, gradient_steps=GRAD_STEPS, precompute_input=True)
        l_bkd_1 = L.LSTMLayer(l_embed, NUM_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=l_mask, gradient_steps=GRAD_STEPS, precompute_input=True, backwards=True)

        l_all_1 = L.concat([l_fwd_1, l_bkd_1], axis=2)

        l_fwd_2 = L.LSTMLayer(l_all_1, NUM_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=l_mask, gradient_steps=GRAD_STEPS, precompute_input=True)
        l_bkd_2 = L.LSTMLayer(l_all_1, NUM_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=l_mask, gradient_steps=GRAD_STEPS, precompute_input=True, backwards=True)

        l_fwd_1_slice = L.SliceLayer(l_fwd_1, -1, 1)
        l_bkd_1_slice = L.SliceLayer(l_bkd_1, 0, 1)
        y_1 = L.ElemwiseSumLayer([l_fwd_1_slice, l_bkd_1_slice])

        l_fwd_2_slice = L.SliceLayer(l_fwd_2, -1, 1)
        l_bkd_2_slice = L.SliceLayer(l_bkd_2, 0, 1)
        y_2 = L.ElemwiseSumLayer([l_fwd_2_slice, l_bkd_2_slice])

        y = L.concat([y_1, y_2], axis=1)
        g = L.DenseLayer(y, num_units=EMBED_DIM, nonlinearity=lasagne.nonlinearities.tanh)
        l_out = L.DenseLayer(g, num_units=vocab_size, W=l_embed.W.T, nonlinearity=lasagne.nonlinearities.softmax)

        return l_out

