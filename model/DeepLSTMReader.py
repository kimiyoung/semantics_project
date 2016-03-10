import theano
import theano.tensor as T
import lasagne.layers as L
import lasagne
import numpy as np
from config import *

class Model:

    def __init__(self, vocab_size, W_init=lasagne.init.GlorotNormal()):

        input_var, mask_var, target_var = T.itensor3('dq_pair'), T.imatrix('dq_mask'), T.ivector('ans')

        network = self.build_network(vocab_size, W_init, input_var, mask_var, skip_connect=SKIP_CONNECT)
        predicted_probs = L.get_output(network)
        predicted_probs_val = L.get_output(network, deterministic=True)

        loss_fn = T.nnet.categorical_crossentropy(predicted_probs, target_var).mean()
        eval_fn = lasagne.objectives.categorical_accuracy(predicted_probs, target_var).mean()

        loss_fn_val = T.nnet.categorical_crossentropy(predicted_probs_val, target_var).mean()
        eval_fn_val = lasagne.objectives.categorical_accuracy(predicted_probs_val, target_var).mean()
        
        all_params = L.get_all_params(network, trainable=True)

        updates = lasagne.updates.rmsprop(loss_fn, all_params, rho=0.95, learning_rate=LEARNING_RATE)
        updates_with_momentum = lasagne.updates.apply_momentum(updates, params=all_params)

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

    def build_network(self, vocab_size, W_init, input_var, mask_var, skip_connect=True):

        l_in = L.InputLayer(shape=(None, None, 1), input_var=input_var)

        l_mask = L.InputLayer(shape=(None, None), input_var=mask_var)

        l_embed = L.EmbeddingLayer(l_in, input_size=vocab_size, output_size=EMBED_DIM, W=W_init)

        l_embed_noise = L.dropout(l_embed, p=DROPOUT_RATE)

        forget_gate_1 = L.Gate(b=lasagne.init.Constant(3))
        forget_gate_2 = L.Gate(b=lasagne.init.Constant(3))

        # NOTE: LSTM layer provided by Lasagne is slightly different from that used in DeepMind's paper.
        # In the paper the cell-to-* weights are not diagonal.
        l_fwd_1 = L.LSTMLayer(l_embed_noise, NUM_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=l_mask,
                gradient_steps=GRAD_STEPS, precompute_input=True, forgetgate=forget_gate_1)

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

        to_next_layer_noise = L.dropout(to_next_layer, p=DROPOUT_RATE)

        l_fwd_2 = L.LSTMLayer(to_next_layer_noise, NUM_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=l_mask,
                gradient_steps=GRAD_STEPS, precompute_input=True, forgetgate=forget_gate_2)

        # slice final states of both lstm layers
        l_fwd_1_slice = L.SliceLayer(l_fwd_1, -1, 1)
        l_fwd_2_slice = L.SliceLayer(l_fwd_2, -1, 1)

        # g will be used to score the words based on their embeddings
        g = L.DenseLayer(L.concat([l_fwd_1_slice, l_fwd_2_slice], axis=1), num_units=EMBED_DIM, W=lasagne.init.GlorotNormal(), nonlinearity=lasagne.nonlinearities.tanh)

        # W is shared with the lookup table
        l_out = L.DenseLayer(g, num_units=vocab_size, W=l_embed.W.T, nonlinearity=lasagne.nonlinearities.softmax, b=None)
        return l_out

