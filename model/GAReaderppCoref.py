import theano
import theano.tensor as T
import lasagne.layers as L
import lasagne
import numpy as np
import cPickle as pickle
from config import *
from tools import sub_sample
from layers import *
from lasagne_layers_v2 import *

def prepare_input(d,q):
    f = np.zeros(d.shape[:2]).astype('int32')
    for i in range(d.shape[0]):
        f[i,:] = np.in1d(d[i,:,0],q[i,:,0])
    return f

class Model:

    def __init__(self, params, vocab_size, num_chars, W_init, embed_dim, num_cand, 
            cloze=True, save_attn=False):
        self.nhidden = params['nhidden']
        self.embed_dim = embed_dim
        self.dropout = params['dropout']
        self.train_emb = params['train_emb']
        self.subsample = params['subsample']
        self.char_dim = params['char_dim']
        self.learning_rate = LEARNING_RATE
        self.num_chars = num_chars
        self.use_feat = params['use_feat']
        self.save_attn = save_attn
        self.gating_fn = params['gating_fn']
        self.numcoref = params['num_coref']
        rlambda = params['lambda']
        K = params['nlayers']
        self.corefdim = params['coref_dim']

        norm = (lasagne.regularization.l2 
                if params['regularizer']=='l2' else lasagne.regularization.l1)
        self.use_chars = self.char_dim!=0
        if W_init is None: 
            W_init = lasagne.init.GlorotNormal().sample((vocab_size, self.embed_dim))

        doc_var, query_var = T.itensor3('doc'), T.itensor3('quer')
        cand_var, doccoref_var, qrycoref_var = T.itensor3('cand'), T.imatrix('coref'), \
                T.imatrix('qcoref')
        doccoref_in, qrycoref_in= T.btensor3('corin'), T.btensor3('qcorin')
        docmask_var, qmask_var, candmask_var = T.bmatrix('doc_mask'), T.bmatrix('q_mask'), \
                T.bmatrix('c_mask')
        target_var = T.ivector('ans')
        feat_var = T.imatrix('feat')
        doc_toks, qry_toks= T.imatrix('dchars'), T.imatrix('qchars')
        tok_var, tok_mask = T.imatrix('tok'), T.bmatrix('tok_mask')
        cloze_var = T.ivector('cloze')
        self.inps = [doc_var, doc_toks, query_var, qry_toks, cand_var, target_var, docmask_var,
                qmask_var, tok_var, tok_mask, candmask_var, feat_var, cloze_var, doccoref_var,
                qrycoref_var, doccoref_in, qrycoref_in]

        if rlambda> 0.: W_pert = W_init + lasagne.init.GlorotNormal().sample(W_init.shape)
        else: W_pert = W_init

        self.predicted_probs, predicted_probs_val, \
                doc_probs_val, self.network, W_emb, attentions = (
                self.build_network(K, vocab_size, W_pert, cloze))

        self.loss_fn = T.nnet.categorical_crossentropy(self.predicted_probs, 
                target_var).mean() + rlambda*norm(W_emb-W_init)
        loss_fn_val = T.nnet.categorical_crossentropy(predicted_probs_val, 
                target_var).mean() + rlambda*norm(W_emb-W_init)
        self.eval_fn = lasagne.objectives.categorical_accuracy(self.predicted_probs, 
                target_var).mean()
        eval_fn_val = lasagne.objectives.categorical_accuracy(predicted_probs_val, 
                target_var).mean()

        self.params = L.get_all_params(self.network, trainable=True)
        
        lr_var = T.fscalar(name="learning_rate")
        updates = lasagne.updates.adam(self.loss_fn, self.params, 
                learning_rate=lr_var)

        self.train_fn = theano.function(self.inps+[lr_var],
                [self.loss_fn, self.eval_fn, self.predicted_probs],
                updates=updates,
                on_unused_input='warn')
        self.validate_fn = theano.function(self.inps, 
                [loss_fn_val, eval_fn_val, predicted_probs_val, 
                    doc_probs_val]+attentions,
                on_unused_input='warn')

    def anneal(self):
        self.learning_rate /= 2

    def train(self, dw, dt, qw, qt, c, a, m_dw, m_qw, tt, tm, m_c, cl, crd, crq):
        f = prepare_input(dw,qw)
        crdi = self._prepare_coref(crd)
        crqi = self._prepare_coref(crq)
        if self.subsample!=-1: m_dw = sub_sample(m_dw, m_c, self.subsample)
        return self.train_fn(dw, dt, qw, qt, c, a, 
                m_dw.astype('int8'), m_qw.astype('int8'), 
                tt, tm.astype('int8'), 
                m_c.astype('int8'), f, cl, crd, crq, crdi, crqi, self.learning_rate)

    def validate(self, dw, dt, qw, qt, c, a, m_dw, m_qw, tt, tm, m_c, cl, crd, crq):
        f = prepare_input(dw,qw)
        crdi = self._prepare_coref(crd)
        crqi = self._prepare_coref(crq)
        if self.subsample!=-1: m_dw = sub_sample(m_dw, m_c, self.subsample)
        return self.validate_fn(dw, dt, qw, qt, c, a, 
                m_dw.astype('int8'), m_qw.astype('int8'), 
                tt, tm.astype('int8'), 
                m_c.astype('int8'), f, cl, crd, crq, crdi, crqi)

    def _prepare_coref(self, cd):
        cdi = np.zeros((cd.shape[0], cd.shape[1], self.numcoref+1), dtype='int8')
        for i in np.arange(cd.shape[0]):
            cdi[i, np.arange(cd.shape[1]), cd[i,:]] = 1
        return cdi[:,:,1:]

    def build_network(self, K, vocab_size, W_init, cloze):

        l_docin = L.InputLayer(shape=(None,None,1), input_var=self.inps[0])
        l_doctokin = L.InputLayer(shape=(None,None), input_var=self.inps[1])
        l_qin = L.InputLayer(shape=(None,None,1), input_var=self.inps[2])
        l_qtokin = L.InputLayer(shape=(None,None), input_var=self.inps[3])
        l_docmask = L.InputLayer(shape=(None,None), input_var=self.inps[6])
        l_qmask = L.InputLayer(shape=(None,None), input_var=self.inps[7])
        l_tokin = L.InputLayer(shape=(None,MAX_WORD_LEN), input_var=self.inps[8])
        l_tokmask = L.InputLayer(shape=(None,MAX_WORD_LEN), input_var=self.inps[9])
        l_featin = L.InputLayer(shape=(None,None), input_var=self.inps[11])
        l_coref_doc = L.InputLayer(shape=(None,None), input_var=self.inps[13])
        l_coref_qry = L.InputLayer(shape=(None,None), input_var=self.inps[14])
        l_doccorefin = L.InputLayer(shape=(None,None,self.numcoref), 
                input_var=self.inps[15])
        l_qrycorefin = L.InputLayer(shape=(None,None,self.numcoref), 
                input_var=self.inps[16])

        doc_shp = self.inps[1].shape
        qry_shp = self.inps[3].shape

        l_docembed = L.EmbeddingLayer(l_docin, input_size=vocab_size, 
                output_size=self.embed_dim, W=W_init) # B x N x 1 x DE
        l_doce = L.ReshapeLayer(l_docembed, 
                (doc_shp[0],doc_shp[1],self.embed_dim)) # B x N x DE
        l_qemb = L.EmbeddingLayer(l_qin, input_size=vocab_size, 
                output_size=self.embed_dim, W=l_docembed.W)
        l_qembed = L.ReshapeLayer(l_qemb, 
                (qry_shp[0],qry_shp[1],self.embed_dim)) # B x N x DE
        l_fembed = L.EmbeddingLayer(l_featin, input_size=2, output_size=2) # B x N x 2

        if self.train_emb==0: 
            l_docembed.params[l_docembed.W].remove('trainable')
            l_qemb.params[l_qemb.W].remove('trainable')

        # char embeddings
        if self.use_chars:
            l_lookup = L.EmbeddingLayer(l_tokin, self.num_chars, self.char_dim) # T x L x D
            l_fgru = L.GRULayer(l_lookup, self.char_dim, grad_clipping=GRAD_CLIP, 
                    mask_input=l_tokmask, gradient_steps=GRAD_STEPS, precompute_input=True,
                    only_return_final=True)
            l_bgru = L.GRULayer(l_lookup, self.char_dim, grad_clipping=GRAD_CLIP, 
                    mask_input=l_tokmask, gradient_steps=GRAD_STEPS, precompute_input=True, 
                    backwards=True, only_return_final=True) # T x 2D
            l_fwdembed = L.DenseLayer(l_fgru, self.embed_dim/2, nonlinearity=None) # T x DE/2
            l_bckembed = L.DenseLayer(l_bgru, self.embed_dim/2, nonlinearity=None) # T x DE/2
            l_embed = L.ElemwiseSumLayer([l_fwdembed, l_bckembed], coeffs=1)
            l_docchar_embed = IndexLayer([l_doctokin, l_embed]) # B x N x DE/2
            l_qchar_embed = IndexLayer([l_qtokin, l_embed]) # B x Q x DE/2

            l_doce = L.ConcatLayer([l_doce, l_docchar_embed], axis=2)
            l_qembed = L.ConcatLayer([l_qembed, l_qchar_embed], axis=2)

        attentions = []
        if self.save_attn:
            l_m = PairwiseInteractionLayer([l_doce,l_qembed])
            attentions.append(L.get_output(l_m, deterministic=True))

        # concatenate coref indicators
        l_doce = L.ConcatLayer([l_doce, l_doccorefin], axis=2)
        l_qembed = L.ConcatLayer([l_qembed, l_qrycorefin], axis=2)
        for i in range(K):
            if self.use_feat and i==K-1: 
                l_doce = L.ConcatLayer([l_doce, l_fembed], axis=2) # B x N x DE+2

            # forward
            l_fwd_doc = CorefGRULayer([l_doce,l_coref_doc], 
                self.nhidden, self.corefdim, self.numcoref, grad_clipping=GRAD_CLIP, 
                mask_input=l_docmask, gradient_steps=GRAD_STEPS, precompute_input=True)
            l_fwd_doc_1 = L.SliceLayer(l_fwd_doc, 
                    indices=slice(0,-(self.numcoref+1)), axis=1)
            if self.corefdim>0:
                l_fwd_coref = L.SliceLayer(l_fwd_doc,
                        indices=slice(-(self.numcoref+1),None), axis=1)
                l_fwd_coref = L.SliceLayer(l_fwd_coref,
                        indices=slice(self.nhidden,None), axis=2)
                l_fwd_q = CorefGRULayer([l_qembed,l_coref_qry], 
                        self.nhidden, self.corefdim, self.numcoref,
                        hid_init_slow=l_fwd_coref,
                        grad_clipping=GRAD_CLIP, 
                        mask_input=l_qmask, 
                        gradient_steps=GRAD_STEPS, precompute_input=True)
            else:
                l_fwd_q = CorefGRULayer([l_qembed,l_coref_qry], 
                        self.nhidden, self.corefdim, self.numcoref,
                        grad_clipping=GRAD_CLIP, 
                        mask_input=l_qmask, 
                        gradient_steps=GRAD_STEPS, precompute_input=True)
            l_fwd_q_1 = L.SliceLayer(l_fwd_q, 
                    indices=slice(0,-(self.numcoref+1)), axis=1)

            # backward
            l_bkd_q = CorefGRULayer([l_qembed,l_coref_qry], 
                    self.nhidden, self.corefdim, self.numcoref,
                    grad_clipping=GRAD_CLIP, 
                    mask_input=l_qmask, 
                    gradient_steps=GRAD_STEPS, precompute_input=True, backwards=True)
            l_bkd_q_1 = L.SliceLayer(l_bkd_q, 
                    indices=slice(0,-(self.numcoref+1)), axis=1)
            if self.corefdim>0:
                l_bkd_coref = L.SliceLayer(l_bkd_q,
                        indices=slice(-(self.numcoref+1),None), axis=1)
                l_bkd_coref = L.SliceLayer(l_bkd_coref,
                        indices=slice(self.nhidden,None), axis=2)
                l_bkd_doc = CorefGRULayer([l_doce,l_coref_doc],
                    self.nhidden, self.corefdim, self.numcoref, grad_clipping=GRAD_CLIP, 
                    hid_init_slow=l_bkd_coref,
                    mask_input=l_docmask, gradient_steps=GRAD_STEPS, precompute_input=True,
                    backwards=True)
            else:
                l_bkd_doc = CorefGRULayer([l_doce,l_coref_doc],
                    self.nhidden, self.corefdim, self.numcoref, grad_clipping=GRAD_CLIP, 
                    mask_input=l_docmask, gradient_steps=GRAD_STEPS, precompute_input=True,
                    backwards=True)
            l_bkd_doc_1 = L.SliceLayer(l_bkd_doc, 
                    indices=slice(0,-(self.numcoref+1)), axis=1)

            l_doc_1 = L.concat([l_fwd_doc_1, l_bkd_doc_1], axis=2) # B x N x DE
            l_q_c_1 = L.ConcatLayer([l_fwd_q_1, l_bkd_q_1], axis=2) # B x Q x DE

            l_m = PairwiseInteractionLayer([l_doc_1, l_q_c_1])
            if self.save_attn: 
                attentions.append(L.get_output(l_m, deterministic=True))

            if i<K-1:
                l_doc_2_in = GatedAttentionLayer([l_doc_1, l_q_c_1, l_m], 
                        gating_fn=self.gating_fn, 
                        mask_input=self.inps[7])
                l_doce = L.dropout(l_doc_2_in, p=self.dropout) # B x N x DE

        l_ans = AnswerLayer([l_doc_1,l_q_c_1], self.inps[12], cloze=cloze)
        l_prob = AttentionSumLayer(l_ans, self.inps[4], mask_input=self.inps[10])
        final = L.get_output(l_prob)
        final_v = L.get_output(l_prob, deterministic=True)
        doc_probs_v = L.get_output(l_ans, deterministic=True)

        return final, final_v, doc_probs_v, l_prob, l_docembed.W, attentions

    def load_model(self, load_path):
        with open(load_path, 'r') as f:
            data = pickle.load(f)
        L.set_all_param_values(self.network, data)

    def save_model(self, save_path):
        data = L.get_all_param_values(self.network)
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

