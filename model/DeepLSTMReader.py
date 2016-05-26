import theano
import theano.tensor as T
import lasagne.layers as L
import lasagne
import numpy as np
from config import *
from collections import OrderedDict

SCALE=0.1

class Model:

    def __init__(self, vocab_size, W_init=lasagne.init.GlorotNormal(), pretrained=None):
        
        self.rho = 0.95
        self.learningrate = LEARNING_RATE

        self.input_var, self.mask_var, self.target_var, self.docidx_var, self.docidx_mask_var = T.itensor3('dq_pair'), T.imatrix('dq_mask'), T.ivector('ans'), T.imatrix('doc'), T.imatrix('doc_mask')

        if pretrained is not None:
            self.params = load_params_shared(pretrained)
        else:
            self.params = self.init_params(vocab_size, W_init)

        network = self.build_network(vocab_size, self.input_var, self.mask_var, self.docidx_var, self.docidx_mask_var,
                skip_connect=SKIP_CONNECT)
        self.predicted_probs = L.get_output(network)
        self.predicted_probs_val = L.get_output(network, deterministic=True)
        #self.predicted_probs, self.predicted_probs_val = self.build_network(vocab_size, 
        #        self.input_var, self.mask_var, self.docidx_var, self.docidx_mask_var, skip_connect=SKIP_CONNECT)

        self.loss_fn = T.nnet.categorical_crossentropy(self.predicted_probs, self.target_var).mean()
        self.eval_fn = lasagne.objectives.categorical_accuracy(self.predicted_probs, 
                self.target_var).mean()

        self.loss_fn_val = T.nnet.categorical_crossentropy(self.predicted_probs_val, 
                self.target_var).mean()
        self.eval_fn_val = lasagne.objectives.categorical_accuracy(self.predicted_probs_val, 
                self.target_var).mean()
        
        updates = lasagne.updates.rmsprop(self.loss_fn, self.params.values(), rho=self.rho, 
                learning_rate=self.learningrate)
        updates_with_momentum = lasagne.updates.apply_momentum(updates, params=self.params.values())

        self.train_fn = theano.function([self.input_var, self.target_var, self.mask_var, self.docidx_var, self.docidx_mask_var], 
                [self.loss_fn, self.eval_fn, self.predicted_probs], updates=updates_with_momentum, on_unused_input='warn')
        self.validate_fn = theano.function([self.input_var, self.target_var, self.mask_var, self.docidx_var, self.docidx_mask_var], 
                [self.loss_fn_val, self.eval_fn_val, self.predicted_probs_val], on_unused_input='warn')

    def update_learningrate(self):
        self.learningrate = max(1e-6,self.learningrate/2)

        updates = lasagne.updates.rmsprop(self.loss_fn, self.params.values(), rho=self.rho, 
                learning_rate=self.learningrate)
        updates_with_momentum = lasagne.updates.apply_momentum(updates, params=self.params.values())

        self.train_fn = theano.function([self.input_var, self.target_var, self.mask_var, self.docidx_var, self.docidx_mask_var], 
                [self.loss_fn, self.eval_fn, self.predicted_probs], updates=updates_with_momentum, on_unused_input='warn')

    def init_params(self, vocab_size, W_init):
        params = OrderedDict()

        # lookup
        params['W_emb'] = theano.shared(W_init)

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
                    size=(NUM_HIDDEN)).astype('float32'), name='W_lstm%d_ci'%index)
            params['b_lstm%d_i'%index] = theano.shared(np.zeros((NUM_HIDDEN)).astype('float32'), 
                    name='b_lstm%d_i'%index)

            params['W_lstm%d_xf'%index] = theano.shared(np.random.normal(loc=0., scale=SCALE, 
                    size=(in_dim,NUM_HIDDEN)).astype('float32'), name='W_lstm%d_xf'%index)
            params['W_lstm%d_hf'%index] = theano.shared(np.random.normal(loc=0., scale=SCALE, 
                    size=(NUM_HIDDEN,NUM_HIDDEN)).astype('float32'), name='W_lstm%d_hf'%index)
            params['W_lstm%d_cf'%index] = theano.shared(np.random.normal(loc=0., scale=SCALE, 
                    size=(NUM_HIDDEN)).astype('float32'), name='W_lstm%d_cf'%index)
            params['b_lstm%d_f'%index] = theano.shared(3*np.ones((NUM_HIDDEN)).astype('float32'), 
                    name='b_lstm%d_f'%index)

            params['W_lstm%d_xo'%index] = theano.shared(np.random.normal(loc=0., scale=SCALE, 
                    size=(in_dim,NUM_HIDDEN)).astype('float32'), name='W_lstm%d_xo'%index)
            params['W_lstm%d_ho'%index] = theano.shared(np.random.normal(loc=0., scale=SCALE, 
                    size=(NUM_HIDDEN,NUM_HIDDEN)).astype('float32'), name='W_lstm%d_ho'%index)
            params['W_lstm%d_co'%index] = theano.shared(np.random.normal(loc=0., scale=SCALE, 
                    size=(NUM_HIDDEN)).astype('float32'), name='W_lstm%d_co'%index)
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

    def train(self, d, q, a, m_d, m_q):
        if MODE=='cqa':
            x = np.concatenate([d, q], axis=1)
            m = np.concatenate([m_d, m_q], axis=1)
        elif MODE=='qca':
            x = np.concatenate([q, d], axis=1)
            m = np.concatenate([m_q, m_d], axis=1)
        return self.train_fn(x, a, m, d.reshape(d.shape[:2]), m_d)

    def validate(self, d, q, a, m_d, m_q):
        if MODE=='cqa':
            x = np.concatenate([d, q], axis=1)
            m = np.concatenate([m_d, m_q], axis=1)
        elif MODE=='qca':
            x = np.concatenate([q, d], axis=1)
            m = np.concatenate([m_q, m_d], axis=1)
        return self.validate_fn(x, a, m, d.reshape(d.shape[:2]), m_d)

    def build_network(self, vocab_size, input_var, mask_var, docidx_var, docidx_mask, skip_connect=True):

        l_in = L.InputLayer(shape=(None, None, 1), input_var=input_var)

        l_mask = L.InputLayer(shape=(None, None), input_var=mask_var)

        l_embed = L.EmbeddingLayer(l_in, input_size=vocab_size, output_size=EMBED_DIM, 
                W=self.params['W_emb'])

        l_embed_noise = L.dropout(l_embed, p=DROPOUT_RATE)

        # NOTE: Moved initialization of forget gate biases to init_params
        #forget_gate_1 = L.Gate(b=lasagne.init.Constant(3))
        #forget_gate_2 = L.Gate(b=lasagne.init.Constant(3))

        # NOTE: LSTM layer provided by Lasagne is slightly different from that used in DeepMind's paper.
        # In the paper the cell-to-* weights are not diagonal.
        # the 1st lstm layer
        in_gate = L.Gate(W_in=self.params['W_lstm1_xi'], W_hid=self.params['W_lstm1_hi'], 
                        W_cell=self.params['W_lstm1_ci'], b=self.params['b_lstm1_i'], 
                        nonlinearity=lasagne.nonlinearities.sigmoid)
        forget_gate = L.Gate(W_in=self.params['W_lstm1_xf'], W_hid=self.params['W_lstm1_hf'], 
                        W_cell=self.params['W_lstm1_cf'], b=self.params['b_lstm1_f'], 
                        nonlinearity=lasagne.nonlinearities.sigmoid)
        out_gate = L.Gate(W_in=self.params['W_lstm1_xo'], W_hid=self.params['W_lstm1_ho'], 
                        W_cell=self.params['W_lstm1_co'], b=self.params['b_lstm1_o'], 
                        nonlinearity=lasagne.nonlinearities.sigmoid)
        cell_gate = L.Gate(W_in=self.params['W_lstm1_xc'], W_hid=self.params['W_lstm1_hc'], 
                        W_cell=None, b=self.params['b_lstm1_c'], 
                        nonlinearity=lasagne.nonlinearities.tanh)
        l_fwd_1 = L.LSTMLayer(l_embed_noise, NUM_HIDDEN, ingate=in_gate, forgetgate=forget_gate,
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

        to_next_layer_noise = L.dropout(to_next_layer, p=DROPOUT_RATE)

        in_gate = L.Gate(W_in=self.params['W_lstm2_xi'], W_hid=self.params['W_lstm2_hi'], 
                        W_cell=self.params['W_lstm2_ci'], b=self.params['b_lstm2_i'], 
                        nonlinearity=lasagne.nonlinearities.sigmoid)
        forget_gate = L.Gate(W_in=self.params['W_lstm2_xf'], W_hid=self.params['W_lstm2_hf'], 
                        W_cell=self.params['W_lstm2_cf'], b=self.params['b_lstm2_f'], 
                        nonlinearity=lasagne.nonlinearities.sigmoid)
        out_gate = L.Gate(W_in=self.params['W_lstm2_xo'], W_hid=self.params['W_lstm2_ho'], 
                        W_cell=self.params['W_lstm2_co'], b=self.params['b_lstm2_o'], 
                        nonlinearity=lasagne.nonlinearities.sigmoid)
        cell_gate = L.Gate(W_in=self.params['W_lstm2_xc'], W_hid=self.params['W_lstm2_hc'], 
                        W_cell=None, b=self.params['b_lstm2_c'], 
                        nonlinearity=lasagne.nonlinearities.tanh)
        l_fwd_2 = L.LSTMLayer(to_next_layer_noise, NUM_HIDDEN, ingate=in_gate, forgetgate=forget_gate,
                        cell=cell_gate, outgate=out_gate, peepholes=True, grad_clipping=GRAD_CLIP, 
                        mask_input=l_mask, gradient_steps=GRAD_STEPS, precompute_input=True)

        # slice final states of both lstm layers
        l_fwd_1_slice = L.SliceLayer(l_fwd_1, -1, 1)
        l_fwd_2_slice = L.SliceLayer(l_fwd_2, -1, 1)

        # g will be used to score the words based on their embeddings
        g = L.DenseLayer(L.concat([l_fwd_1_slice, l_fwd_2_slice], axis=1), num_units=EMBED_DIM, 
                W=self.params['W_dense'], b=self.params['b_dense'], 
                nonlinearity=lasagne.nonlinearities.tanh)

        ## get outputs
        #g_out = L.get_output(g) # B x D
        #g_out_val = L.get_output(g, deterministic=True) # B x D

        ## compute softmax probs
        #probs,_ = theano.scan(fn=lambda g,d,dm,W: T.nnet.softmax(T.dot(g,W[d,:].T)*dm),
        #                    outputs_info=None,
        #                    sequences=[g_out,docidx_var,docidx_mask],
        #                    non_sequences=self.params['W_emb'])
        #predicted_probs = probs.reshape(docidx_var.shape) # B x N
        #probs_val,_ = theano.scan(fn=lambda g,d,dm,W: T.nnet.softmax(T.dot(g,W[d,:].T)*dm),
        #                    outputs_info=None,
        #                    sequences=[g_out_val,docidx_var,docidx_mask],
        #                    non_sequences=self.params['W_emb'])
        #predicted_probs_val = probs_val.reshape(docidx_var.shape) # B x N
        #return predicted_probs, predicted_probs_val

        # W is shared with the lookup table
        l_out = L.DenseLayer(g, num_units=vocab_size, W=self.params['W_emb'].T, 
                nonlinearity=lasagne.nonlinearities.softmax, b=None)
        return l_out

    def save_model(self, save_path):
        saveparams = OrderedDict()
        for kk,vv in self.params.iteritems():
            saveparams[kk] = vv.get_value()
        np.savez(save_path,**saveparams)

def load_params_shared(path):
    """
    Load previously saved model
    """
    params = OrderedDict()

    with open(path,'r') as f:
        npzfile = np.load(f)
        for kk, vv in npzfile.iteritems():
            params[kk] = theano.shared(vv, name=kk)

    return params

