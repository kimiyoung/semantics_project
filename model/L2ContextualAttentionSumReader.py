import theano
import theano.tensor as T
import lasagne.layers as L
import lasagne
import numpy as np
import cPickle as pickle
from config import *

class Model:

    def __init__(self, vocab_size, W_init=lasagne.init.GlorotNormal()):

        doc_var, query_var = T.itensor3('doc'), T.itensor3('quer')
        docmask_var, qmask_var = T.imatrix('doc_mask'), T.imatrix('q_mask')
        #doci_var = T.itensor3('doci')
        target_var = T.ivector('ans')

        predicted_probs, predicted_probs_val, self.doc_net, self.q_net = self.build_network(vocab_size, 
		doc_var, query_var, docmask_var, qmask_var, W_init)

        loss_fn = T.nnet.categorical_crossentropy(predicted_probs, target_var).mean()
        eval_fn = lasagne.objectives.categorical_accuracy(predicted_probs, target_var).mean()

        loss_fn_val = T.nnet.categorical_crossentropy(predicted_probs_val, target_var).mean()
        eval_fn_val = lasagne.objectives.categorical_accuracy(predicted_probs_val, target_var).mean()

        params = L.get_all_params(self.doc_net, trainable=True) + L.get_all_params(self.q_net, trainable=True)
        
        updates = lasagne.updates.adam(loss_fn, params, learning_rate=LEARNING_RATE)
        #updates_with_momentum = lasagne.updates.apply_momentum(updates, params=params)

        self.train_fn = theano.function([doc_var, query_var, target_var, docmask_var, qmask_var], 
                [loss_fn, eval_fn, predicted_probs], 
                updates=updates)
        self.validate_fn = theano.function([doc_var, query_var, target_var, docmask_var, qmask_var], 
                [loss_fn_val, eval_fn_val, predicted_probs_val])

    def train(self, d, q, a, m_d, m_q):
        return self.train_fn(d, q, a, m_d, m_q)

    def validate(self, d, q, a, m_d, m_q):
        return self.validate_fn(d, q, a, m_d, m_q)

    def build_network(self, vocab_size, doc_var, query_var, docmask_var, qmask_var, W_init):

        l_docin = L.InputLayer(shape=(None,None,1), input_var=doc_var)
        l_qin = L.InputLayer(shape=(None,None,1), input_var=query_var)
        l_docmask = L.InputLayer(shape=(None,None), input_var=docmask_var)
        l_qmask = L.InputLayer(shape=(None,None), input_var=qmask_var)
        l_docembed = L.EmbeddingLayer(l_docin, input_size=vocab_size, 
                output_size=EMBED_DIM, W=W_init) # B x N x 1 x DE
        l_doce = L.ReshapeLayer(l_docembed, (doc_var.shape[0],doc_var.shape[1],EMBED_DIM)) # B x N x DE
        l_qembed = L.EmbeddingLayer(l_qin, input_size=vocab_size, 
                output_size=EMBED_DIM, W=l_docembed.W)

        l_fwd_q = L.GRULayer(l_qembed, NUM_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=l_qmask, 
                gradient_steps=GRAD_STEPS, precompute_input=True)
        l_bkd_q = L.GRULayer(l_qembed, NUM_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=l_qmask, 
                gradient_steps=GRAD_STEPS, precompute_input=True, backwards=True)

        l_fwd_q_slice = L.SliceLayer(l_fwd_q, -1, 1)
        l_bkd_q_slice = L.SliceLayer(l_bkd_q, 0, 1)
        l_q = L.ConcatLayer([l_fwd_q_slice, l_bkd_q_slice]) # B x 2D
        q = L.get_output(l_q) # B x 2D

        l_fwd_doc_1 = L.GRULayer(l_doce, NUM_HIDDEN, grad_clipping=GRAD_CLIP, 
                mask_input=l_docmask, gradient_steps=GRAD_STEPS, precompute_input=True)
        l_bkd_doc_1 = L.GRULayer(l_doce, NUM_HIDDEN, grad_clipping=GRAD_CLIP, 
                mask_input=l_docmask, gradient_steps=GRAD_STEPS, precompute_input=True, backwards=True)

        l_doc_1 = L.concat([l_fwd_doc_1, l_bkd_doc_1], axis=2)
	l_doc_1 = L.dropout(l_doc_1, p=DROPOUT_RATE)

        l_fwd_q_c = L.GRULayer(l_qembed, NUM_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=l_qmask, 
                gradient_steps=GRAD_STEPS, precompute_input=True)
        l_bkd_q_c = L.GRULayer(l_qembed, NUM_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=l_qmask, 
                gradient_steps=GRAD_STEPS, precompute_input=True, backwards=True)

        l_fwd_q_slice_c = L.SliceLayer(l_fwd_q_c, -1, 1)
        l_bkd_q_slice_c = L.SliceLayer(l_bkd_q_c, 0, 1)
        l_q_c = L.ConcatLayer([l_fwd_q_slice_c, l_bkd_q_slice_c]) # B x DE

        qd = L.get_output(l_q_c)
        q_rep = T.reshape(T.tile(qd,(1,doc_var.shape[1])), 
                (doc_var.shape[0],doc_var.shape[1],2*NUM_HIDDEN)) # B x N x DE

        l_q_rep_in = L.InputLayer(shape=(None,None,2*NUM_HIDDEN), input_var=q_rep)
        l_doc_gru_in = L.ElemwiseMergeLayer([l_doc_1, l_q_rep_in], T.mul)

        l_fwd_doc = L.GRULayer(l_doc_gru_in, NUM_HIDDEN, grad_clipping=GRAD_CLIP, 
                mask_input=l_docmask, gradient_steps=GRAD_STEPS, precompute_input=True)
        l_bkd_doc = L.GRULayer(l_doc_gru_in, NUM_HIDDEN, grad_clipping=GRAD_CLIP, 
                mask_input=l_docmask, gradient_steps=GRAD_STEPS, precompute_input=True, backwards=True)

        l_doc = L.concat([l_fwd_doc, l_bkd_doc], axis=2)

        d = L.get_output(l_doc) # B x N x 2D
        p = T.batched_dot(d,q) # B x N
        pm = T.nnet.softmax(T.set_subtensor(T.alloc(-20.,p.shape[0],p.shape[1])[docmask_var.nonzero()],
                p[docmask_var.nonzero()]))

        index = T.reshape(T.repeat(T.arange(p.shape[0]),p.shape[1]),p.shape)
        final = T.inc_subtensor(T.alloc(0.,p.shape[0],vocab_size)[index,T.flatten(doc_var,outdim=2)],pm)

        dv = L.get_output(l_doc, deterministic=True) # B x N x 2D
        p = T.batched_dot(dv,q) # B x N
        pm = T.nnet.softmax(T.set_subtensor(T.alloc(-20.,p.shape[0],p.shape[1])[docmask_var.nonzero()],
                p[docmask_var.nonzero()]))

        index = T.reshape(T.repeat(T.arange(p.shape[0]),p.shape[1]),p.shape)
        final_v = T.inc_subtensor(T.alloc(0.,p.shape[0],vocab_size)[index,T.flatten(doc_var,outdim=2)],pm)

        return final, final_v, l_doc, [l_q, l_q_c]

    def load_model(self, load_path):
        with open(load_path, 'r') as f:
            data = pickle.load(f)
        L.set_all_param_values([self.doc_net]+self.q_net, data)

    def save_model(self, save_path):
        data = L.get_all_param_values([self.doc_net]+self.q_net)
        with open(save_path, 'w') as f:
            pickle.dump(data, f)

