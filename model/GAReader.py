import theano
import theano.tensor as T
import lasagne.layers as L
import lasagne
import numpy as np
import cPickle as pickle
from config import *
from tools import sub_sample

class Model:

    def __init__(self, K, vocab_size, W_init, regularizer, rlambda, nhidden, embed_dim,
            dropout, train_emb, subsample):
        self.nhidden = nhidden
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.train_emb = train_emb
        self.subsample = subsample
        norm = lasagne.regularization.l2 if regularizer=='l2' else lasagne.regularization.l1
        if W_init is None: W_init = lasagne.init.GlorotNormal().sample((vocab_size, self.embed_dim))

        doc_var, query_var, cand_var = T.itensor3('doc'), T.itensor3('quer'), T.wtensor3('cand')
        docmask_var, qmask_var, candmask_var = T.bmatrix('doc_mask'), T.bmatrix('q_mask'), \
                T.bmatrix('c_mask')
        target_var = T.ivector('ans')

        if rlambda> 0.: W_pert = W_init + lasagne.init.GlorotNormal().sample(W_init.shape)
        else: W_pert = W_init
        predicted_probs, predicted_probs_val, self.doc_net, self.q_net, W_emb = self.build_network(K, 
                vocab_size, doc_var, query_var, cand_var, docmask_var, qmask_var, 
                candmask_var, W_pert)

        loss_fn = T.nnet.categorical_crossentropy(predicted_probs, target_var).mean() + \
                rlambda*norm(W_emb-W_init)
        eval_fn = lasagne.objectives.categorical_accuracy(predicted_probs, target_var).mean()

        loss_fn_val = T.nnet.categorical_crossentropy(predicted_probs_val, target_var).mean() + \
                rlambda*norm(W_emb-W_init)
        eval_fn_val = lasagne.objectives.categorical_accuracy(predicted_probs_val, target_var).mean()

        params = L.get_all_params(self.doc_net, trainable=True) + \
                L.get_all_params(self.q_net, trainable=True)
        
        updates = lasagne.updates.adam(loss_fn, params, learning_rate=LEARNING_RATE)

        self.train_fn = theano.function([doc_var, query_var, cand_var, target_var, docmask_var, \
                qmask_var, candmask_var], 
                [loss_fn, eval_fn, predicted_probs], 
                updates=updates)
        self.validate_fn = theano.function([doc_var, query_var, cand_var, target_var, docmask_var, \
                qmask_var, candmask_var], 
                [loss_fn_val, eval_fn_val, predicted_probs_val])

    def train(self, d, q, c, a, m_d, m_q, m_c):
        if self.subsample!=-1: m_d = sub_sample(m_d, m_c, self.subsample)
        return self.train_fn(d, q, c, a, m_d.astype('int8'), m_q.astype('int8'), m_c.astype('int8'))

    def validate(self, d, q, c, a, m_d, m_q, m_c):
        if self.subsample!=-1: m_d = sub_sample(m_d, m_c, self.subsample)
        return self.validate_fn(d, q, c, a, 
                m_d.astype('int8'), m_q.astype('int8'), m_c.astype('int8'))

    def build_network(self, K, vocab_size, doc_var, query_var, cand_var, docmask_var, 
            qmask_var, candmask_var, W_init):

        l_docin = L.InputLayer(shape=(None,None,1), input_var=doc_var)
        l_qin = L.InputLayer(shape=(None,None,1), input_var=query_var)
        l_docmask = L.InputLayer(shape=(None,None), input_var=docmask_var)
        l_qmask = L.InputLayer(shape=(None,None), input_var=qmask_var)
        l_docembed = L.EmbeddingLayer(l_docin, input_size=vocab_size, 
                output_size=self.embed_dim, W=W_init) # B x N x 1 x DE
        l_doce = L.ReshapeLayer(l_docembed, 
                (doc_var.shape[0],doc_var.shape[1],self.embed_dim)) # B x N x DE
        l_qembed = L.EmbeddingLayer(l_qin, input_size=vocab_size, 
                output_size=self.embed_dim, W=l_docembed.W)

        if self.train_emb==0: l_docembed.params[l_docembed.W].remove('trainable')

        l_fwd_q = L.GRULayer(l_qembed, self.nhidden, grad_clipping=GRAD_CLIP, mask_input=l_qmask, 
                gradient_steps=GRAD_STEPS, precompute_input=True)
        l_bkd_q = L.GRULayer(l_qembed, self.nhidden, grad_clipping=GRAD_CLIP, mask_input=l_qmask, 
                gradient_steps=GRAD_STEPS, precompute_input=True, backwards=True)

        l_fwd_q_slice = L.SliceLayer(l_fwd_q, -1, 1)
        l_bkd_q_slice = L.SliceLayer(l_bkd_q, 0, 1)
        l_q = L.ConcatLayer([l_fwd_q_slice, l_bkd_q_slice]) # B x 2D
        q = L.get_output(l_q) # B x 2D

        l_qs = [l_q]
        for i in range(K-1):
            l_fwd_doc_1 = L.GRULayer(l_doce, self.nhidden, grad_clipping=GRAD_CLIP, 
                    mask_input=l_docmask, gradient_steps=GRAD_STEPS, precompute_input=True)
            l_bkd_doc_1 = L.GRULayer(l_doce, self.nhidden, grad_clipping=GRAD_CLIP, 
                    mask_input=l_docmask, gradient_steps=GRAD_STEPS, 
                    precompute_input=True, backwards=True)

            l_doc_1 = L.concat([l_fwd_doc_1, l_bkd_doc_1], axis=2)

            l_fwd_q_1 = L.GRULayer(l_qembed, self.nhidden, grad_clipping=GRAD_CLIP, mask_input=l_qmask, 
                    gradient_steps=GRAD_STEPS, precompute_input=True)
            l_bkd_q_1 = L.GRULayer(l_qembed, self.nhidden, grad_clipping=GRAD_CLIP, mask_input=l_qmask, 
                    gradient_steps=GRAD_STEPS, precompute_input=True, backwards=True)

            l_fwd_q_slice_1 = L.SliceLayer(l_fwd_q_1, -1, 1)
            l_bkd_q_slice_1 = L.SliceLayer(l_bkd_q_1, 0, 1)
            l_q_c_1 = L.ConcatLayer([l_fwd_q_slice_1, l_bkd_q_slice_1]) # B x DE
            
            l_qs.append(l_q_c_1)

            qd = L.get_output(l_q_c_1)
            q_rep = T.reshape(T.tile(qd,(1,doc_var.shape[1])), 
                    (doc_var.shape[0],doc_var.shape[1],2*self.nhidden)) # B x N x DE

            l_q_rep_in = L.InputLayer(shape=(None,None,2*self.nhidden), input_var=q_rep)
            l_doc_2_in = L.ElemwiseMergeLayer([l_doc_1, l_q_rep_in], T.mul)
            l_doce = L.dropout(l_doc_2_in, p=self.dropout)

        l_fwd_doc = L.GRULayer(l_doce, self.nhidden, grad_clipping=GRAD_CLIP, 
                mask_input=l_docmask, gradient_steps=GRAD_STEPS, precompute_input=True)
        l_bkd_doc = L.GRULayer(l_doce, self.nhidden, grad_clipping=GRAD_CLIP, 
                mask_input=l_docmask, gradient_steps=GRAD_STEPS, 
                precompute_input=True, backwards=True)

        l_doc = L.concat([l_fwd_doc, l_bkd_doc], axis=2)

        d = L.get_output(l_doc) # B x N x 2D
        p = T.batched_dot(d,q) # B x N
        pm = T.nnet.softmax(p)*candmask_var
        pm = pm/pm.sum(axis=1)[:,np.newaxis]
        final = T.batched_dot(pm, cand_var)

        dv = L.get_output(l_doc, deterministic=True) # B x N x 2D
        p = T.batched_dot(dv,q) # B x N
        pm = T.nnet.softmax(p)*candmask_var
        pm = pm/pm.sum(axis=1)[:,np.newaxis]
        final_v = T.batched_dot(pm, cand_var)

        return final, final_v, l_doc, l_qs, l_docembed.W

    def load_model(self, load_path):
        with open(load_path, 'r') as f:
            data = pickle.load(f)
        L.set_all_param_values([self.doc_net]+self.q_net, data)

    def save_model(self, save_path):
        data = L.get_all_param_values([self.doc_net]+self.q_net)
        with open(save_path, 'w') as f:
            pickle.dump(data, f)

