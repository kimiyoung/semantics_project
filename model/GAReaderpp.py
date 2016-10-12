import theano
import theano.tensor as T
import lasagne.layers as L
import lasagne
import numpy as np
import cPickle as pickle
from config import *
from tools import sub_sample
from layers import IndexLayer

def prepare_input(d,q):
    f = np.zeros(d.shape[:2]).astype('int32')
    for i in range(d.shape[0]):
        f[i,:] = np.in1d(d[i,:,0],q[i,:,0])
    return f

class Model:

    def __init__(self, K, vocab_size, num_chars, W_init, regularizer, rlambda, 
            nhidden, embed_dim, dropout, train_emb, subsample, char_dim, use_feat):
        self.nhidden = nhidden
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.train_emb = train_emb
        self.subsample = subsample
        self.char_dim = char_dim
        self.learning_rate = LEARNING_RATE
        self.num_chars = num_chars
        self.use_feat = use_feat

        norm = lasagne.regularization.l2 if regularizer=='l2' else lasagne.regularization.l1
        self.use_chars = self.char_dim!=0
        if W_init is None: W_init = lasagne.init.GlorotNormal().sample((vocab_size, self.embed_dim))

        doc_var, query_var, cand_var = T.itensor3('doc'), T.itensor3('quer'), \
                T.wtensor3('cand')
        docmask_var, qmask_var, candmask_var = T.bmatrix('doc_mask'), T.bmatrix('q_mask'), \
                T.bmatrix('c_mask')
        target_var = T.ivector('ans')
        feat_var = T.imatrix('feat')
        doc_toks, qry_toks= T.imatrix('dchars'), T.imatrix('qchars')
        tok_var, tok_mask = T.imatrix('tok'), T.bmatrix('tok_mask')
        cloze_var = T.ivector('cloze')
        self.inps = [doc_var, doc_toks, query_var, qry_toks, cand_var, target_var, docmask_var,
                qmask_var, tok_var, tok_mask, candmask_var, feat_var, cloze_var]

        if rlambda> 0.: W_pert = W_init + lasagne.init.GlorotNormal().sample(W_init.shape)
        else: W_pert = W_init
        self.predicted_probs, predicted_probs_val, self.doc_net, self.q_net, W_emb, attentions = (
                self.build_network(K, vocab_size, W_pert))

        self.loss_fn = T.nnet.categorical_crossentropy(self.predicted_probs, target_var).mean() + \
                rlambda*norm(W_emb-W_init)
        self.eval_fn = lasagne.objectives.categorical_accuracy(self.predicted_probs, 
                target_var).mean()

        loss_fn_val = T.nnet.categorical_crossentropy(predicted_probs_val, target_var).mean() + \
                rlambda*norm(W_emb-W_init)
        eval_fn_val = lasagne.objectives.categorical_accuracy(predicted_probs_val, 
                target_var).mean()

        self.params = L.get_all_params([self.doc_net]+self.q_net, trainable=True)
        
        updates = lasagne.updates.adam(self.loss_fn, self.params, learning_rate=self.learning_rate)

        self.train_fn = theano.function(self.inps,
                [self.loss_fn, self.eval_fn, self.predicted_probs], 
                updates=updates,
                on_unused_input='warn')
        self.validate_fn = theano.function(self.inps, 
                [loss_fn_val, eval_fn_val, predicted_probs_val]+attentions,
                on_unused_input='warn')

    def anneal(self):
        self.learning_rate /= 2
        updates = lasagne.updates.adam(self.loss_fn, self.params, learning_rate=self.learning_rate)
        self.train_fn = theano.function(self.inps, \
                [self.loss_fn, self.eval_fn, self.predicted_probs], 
                updates=updates,
                on_unused_input='warn')

    def train(self, dw, dt, qw, qt, c, a, m_dw, m_qw, tt, tm, m_c, cl):
        f = prepare_input(dw,qw)
        if self.subsample!=-1: m_dw = sub_sample(m_dw, m_c, self.subsample)
        return self.train_fn(dw, dt, qw, qt, c, a, 
                m_dw.astype('int8'), m_qw.astype('int8'), 
                tt, tm.astype('int8'), 
                m_c.astype('int8'), f, cl)

    def validate(self, dw, dt, qw, qt, c, a, m_dw, m_qw, tt, tm, m_c, cl):
        f = prepare_input(dw,qw)
        if self.subsample!=-1: m_dw = sub_sample(m_dw, m_c, self.subsample)
        return self.validate_fn(dw, dt, qw, qt, c, a, 
                m_dw.astype('int8'), m_qw.astype('int8'), 
                tt, tm.astype('int8'), 
                m_c.astype('int8'), f, cl)

    def build_network(self, K, vocab_size, W_init):

        l_docin = L.InputLayer(shape=(None,None,1), input_var=self.inps[0])
        l_doctokin = L.InputLayer(shape=(None,None), input_var=self.inps[1])
        l_qin = L.InputLayer(shape=(None,None,1), input_var=self.inps[2])
        l_qtokin = L.InputLayer(shape=(None,None), input_var=self.inps[3])
        l_docmask = L.InputLayer(shape=(None,None), input_var=self.inps[6])
        l_qmask = L.InputLayer(shape=(None,None), input_var=self.inps[7])
        l_tokin = L.InputLayer(shape=(None,MAX_WORD_LEN), input_var=self.inps[8])
        l_tokmask = L.InputLayer(shape=(None,MAX_WORD_LEN), input_var=self.inps[9])
        l_featin = L.InputLayer(shape=(None,None), input_var=self.inps[11])

        doc_shp = self.inps[1].shape
        qry_shp = self.inps[3].shape

        l_docembed = L.EmbeddingLayer(l_docin, input_size=vocab_size, 
                output_size=self.embed_dim, W=W_init) # B x N x 1 x DE
        l_doce = L.ReshapeLayer(l_docembed, 
                (doc_shp[0],doc_shp[1],self.embed_dim)) # B x N x DE
        l_qembed = L.EmbeddingLayer(l_qin, input_size=vocab_size, 
                output_size=self.embed_dim, W=l_docembed.W)
        l_qembed = L.ReshapeLayer(l_qembed, 
                (qry_shp[0],qry_shp[1],self.embed_dim)) # B x N x DE
        l_fembed = L.EmbeddingLayer(l_featin, input_size=2, output_size=2) # B x N x 2

        if self.train_emb==0: l_docembed.params[l_docembed.W].remove('trainable')

        # char embeddings
        if self.use_chars:
            l_lookup = L.EmbeddingLayer(l_tokin, self.num_chars, 2*self.char_dim) # T x L x D
            l_fgru = L.GRULayer(l_lookup, self.char_dim, grad_clipping=GRAD_CLIP, 
                    mask_input=l_tokmask, gradient_steps=GRAD_STEPS, precompute_input=True,
                    only_return_final=True)
            l_bgru = L.GRULayer(l_lookup, 2*self.char_dim, grad_clipping=GRAD_CLIP, 
                    mask_input=l_tokmask, gradient_steps=GRAD_STEPS, precompute_input=True, 
                    backwards=True, only_return_final=True) # T x 2D
            l_fwdembed = L.DenseLayer(l_fgru, self.embed_dim/2, nonlinearity=None) # T x DE/2
            l_bckembed = L.DenseLayer(l_bgru, self.embed_dim/2, nonlinearity=None) # T x DE/2
            l_embed = L.ElemwiseSumLayer([l_fwdembed, l_bckembed], coeffs=1)
            l_docchar_embed = IndexLayer([l_doctokin, l_embed]) # B x N x DE/2
            l_qchar_embed = IndexLayer([l_qtokin, l_embed]) # B x Q x DE/2

            l_doce = L.ConcatLayer([l_doce, l_docchar_embed], axis=2)
            l_qembed = L.ConcatLayer([l_qembed, l_qchar_embed], axis=2)

        l_fwd_q = L.GRULayer(l_qembed, self.nhidden, grad_clipping=GRAD_CLIP, mask_input=l_qmask, 
                gradient_steps=GRAD_STEPS, precompute_input=True, only_return_final=False)
        l_bkd_q = L.GRULayer(l_qembed, self.nhidden, grad_clipping=GRAD_CLIP, mask_input=l_qmask, 
                gradient_steps=GRAD_STEPS, precompute_input=True, backwards=True, 
                only_return_final=False)

        l_q = L.ConcatLayer([l_fwd_q, l_bkd_q], axis=2) # B x Q x 2D
        q = L.get_output(l_q) # B x Q x 2D
        q = q[T.arange(q.shape[0]),self.inps[12],:] # B x 2D

        l_qs = [l_q]
        attentions = []
        for i in range(K-1):
            l_fwd_doc_1 = L.GRULayer(l_doce, self.nhidden, grad_clipping=GRAD_CLIP, 
                    mask_input=l_docmask, gradient_steps=GRAD_STEPS, precompute_input=True)
            l_bkd_doc_1 = L.GRULayer(l_doce, self.nhidden, grad_clipping=GRAD_CLIP, 
                    mask_input=l_docmask, gradient_steps=GRAD_STEPS, precompute_input=True, \
                            backwards=True)

            l_doc_1 = L.concat([l_fwd_doc_1, l_bkd_doc_1], axis=2) # B x N x DE

            l_fwd_q_1 = L.GRULayer(l_qembed, self.nhidden, grad_clipping=GRAD_CLIP, 
                    mask_input=l_qmask, 
                    gradient_steps=GRAD_STEPS, precompute_input=True)
            l_bkd_q_1 = L.GRULayer(l_qembed, self.nhidden, grad_clipping=GRAD_CLIP, 
                    mask_input=l_qmask, 
                    gradient_steps=GRAD_STEPS, precompute_input=True, backwards=True)

            l_q_c_1 = L.ConcatLayer([l_fwd_q_1, l_bkd_q_1], axis=2) # B x Q x DE
            l_qs.append(l_q_c_1)

            qd = L.get_output(l_q_c_1) # B x Q x DE
            dd = L.get_output(l_doc_1) # B x N x DE
            M = T.batched_dot(dd, qd.dimshuffle((0,2,1))) # B x N x Q
            alphas = T.nnet.softmax(T.reshape(M, (M.shape[0]*M.shape[1],M.shape[2])))
            alphas_r = T.reshape(alphas, (M.shape[0],M.shape[1],M.shape[2]))* \
                    self.inps[7][:,np.newaxis,:] # B x N x Q
            alphas_r = alphas_r/alphas_r.sum(axis=2)[:,:,np.newaxis] # B x N x Q
            attentions.append(alphas_r)
            q_rep = T.batched_dot(alphas_r, qd) # B x N x DE

            l_q_rep_in = L.InputLayer(shape=(None,None,2*self.nhidden), input_var=q_rep)
            l_doc_2_in = L.ElemwiseMergeLayer([l_doc_1, l_q_rep_in], T.mul)
            l_doce = L.dropout(l_doc_2_in, p=self.dropout) # B x N x DE

        if self.use_feat: l_doce = L.ConcatLayer([l_doce, l_fembed], axis=2) # B x N x DE+2
        l_fwd_doc = L.GRULayer(l_doce, self.nhidden, grad_clipping=GRAD_CLIP, 
                mask_input=l_docmask, gradient_steps=GRAD_STEPS, precompute_input=True)
        l_bkd_doc = L.GRULayer(l_doce, self.nhidden, grad_clipping=GRAD_CLIP, 
                mask_input=l_docmask, gradient_steps=GRAD_STEPS, precompute_input=True, \
                        backwards=True)

        l_doc = L.concat([l_fwd_doc, l_bkd_doc], axis=2)

        d = L.get_output(l_doc) # B x N x 2D
        p = T.batched_dot(d,q) # B x N
        pm = T.nnet.softmax(p)*self.inps[10]
        pm = pm/pm.sum(axis=1)[:,np.newaxis]
        final = T.batched_dot(pm, self.inps[4])

        dv = L.get_output(l_doc, deterministic=True) # B x N x 2D
        p = T.batched_dot(dv,q) # B x N
        pm = T.nnet.softmax(p)*self.inps[10]
        pm = pm/pm.sum(axis=1)[:,np.newaxis]
        final_v = T.batched_dot(pm, self.inps[4])

        return final, final_v, l_doc, l_qs, l_docembed.W, attentions

    def load_model(self, load_path):
        with open(load_path, 'r') as f:
            data = pickle.load(f)
        L.set_all_param_values([self.doc_net]+self.q_net, data)

    def save_model(self, save_path):
        data = L.get_all_param_values([self.doc_net]+self.q_net)
        with open(save_path, 'w') as f:
            pickle.dump(data, f)

if __name__=="__main__":
    m_d = np.asarray([[1,1,1,1,1,0,0,0],[1,1,1,1,1,1,1,0]]).astype('int32')
    m_c = np.asarray([[1,1,0,0,0,0,0,0],[0,1,0,1,0,0,1,0]]).astype('int32')
    print 'doc mask', m_d
    print 'cand mask', m_c
    print 'new mask (N=1)', sub_sample(m_d, m_c, 1)
    print 'new mask (N=2)', sub_sample(m_d, m_c, 2)
    print 'new mask (N=3)', sub_sample(m_d, m_c, 3)

