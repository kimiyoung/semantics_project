import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
import numpy as np
import cPickle as pkl
from config import *
from layers import *

def prepare_input(d,a,q):
    t = np.zeros(d.shape[:2]).astype('int32')
    f = np.zeros(d.shape[:2]).astype('int8')
    ai = (d==a[:,np.newaxis,np.newaxis]).argmax(axis=1).flatten()
    for i in range(d.shape[0]):
        t[i,:ai[i]] = 0
        t[i,ai[i]] = 1
        t[i,ai[i]+1:] = 2
        f[i,:] = np.in1d(d[i,:,0],q[i,:,0])
    return t,f

class Model:
    def __init__(self, vocab_size, num_classes, W_init=lasagne.init.GlorotNormal()):

        doc_var, query_var = T.imatrix('doc'), T.imatrix('quer')
        docmask_var, qmask_var= T.bmatrix('doc_mask'), T.bmatrix('q_mask')
        target_var = T.imatrix('ans')
        feat_var = T.bmatrix('feat')

        self.inps = [doc_var, query_var, target_var, docmask_var, qmask_var, feat_var]
        loss, self.params, test_out = self.build_network(vocab_size, num_classes, \
                W_init, *self.inps)

        loss = loss + REGULARIZATION*lasagne.regularization.apply_penalty(self.params, \
                lasagne.regularization.l2)
        updates = lasagne.updates.rmsprop(loss, self.params, learning_rate=LEARNING_RATE, \
                rho=0.95, epsilon=0.0001)
        self.train_fn = theano.function(self.inps, loss, updates=updates)
        self.validate_fn = theano.function(self.inps, test_out, on_unused_input='warn')

    def train(self, d, q, a, m_d, m_q, m_c):
        t,f = prepare_input(d,a,q)
        return self.train_fn(d[:,:,0],q[:,:,0],t,m_d.astype('int8'),m_q.astype('int8'),f), 0., None

    def validate(self, d, q, a, m_d, m_q, m_c):
        t,f = prepare_input(d,a,q)
        preds = self.validate_fn(d[:,:,0],q[:,:,0],t,m_d.astype('int8'),m_q.astype('int8'),f)
        ans = d[np.arange(d.shape[0]),(preds==1).argmax(axis=1),0]
        acc = float(np.sum(ans==a))/d.shape[0]
        return 0., acc, None

    def build_network(self, V, C, W, dv, qv, tv, dmv, qmv, fv):
        # inputs
        l_docin = L.InputLayer(shape=(None,None), input_var=dv)
        l_qin = L.InputLayer(shape=(None,None), input_var=qv)
        l_docmask = L.InputLayer(shape=(None,None), input_var=dmv)
        l_qmask = L.InputLayer(shape=(None,None), input_var=qmv)
        l_featin = L.InputLayer(shape=(None,None), input_var=fv)
        l_docembed = L.EmbeddingLayer(l_docin, input_size=V, 
                output_size=EMBED_DIM, W=W) # B x N x DE
        l_qembed = L.EmbeddingLayer(l_qin, input_size=V, 
                output_size=EMBED_DIM, W=l_docembed.W) # B x Q x DE
        l_fembed = L.EmbeddingLayer(l_featin, input_size=2, output_size=2) # B x N x 2

        # question lstm
        l_q_lstm = L.LSTMLayer(l_qembed, NUM_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=l_qmask, \
                gradient_steps=GRAD_STEPS, precompute_input=True) # B x Q x D
        l_q_lstm = L.dropout(l_q_lstm, p=DROPOUT_RATE)
        l_q_att_in = L.ReshapeLayer(l_q_lstm, (qv.shape[0]*qv.shape[1],NUM_HIDDEN)) # BQ x D
        l_q_att_1 = L.DenseLayer(l_q_att_in, NUM_HIDDEN, b=None, \
                nonlinearity=lasagne.nonlinearities.tanh) # BQ x D
        l_q_att_2 = L.DenseLayer(l_q_att_1, 1, b=None, nonlinearity=None) # BQ x 1
        l_q_att_out = L.ReshapeLayer(l_q_att_2, (qv.shape[0], qv.shape[1])) # B x Q
        q = L.get_output(l_q_lstm)
        alphas = T.nnet.softmax(L.get_output(l_q_att_out))*qmv # B x Q
        alphas = alphas/alphas.sum(axis=1)[:,np.newaxis]
        rq = (alphas[:,:,np.newaxis]*q).sum(axis=1) # B x D

        # evidence lstm
        rq_tiled = T.reshape(T.tile(rq, (1,dv.shape[1])), (dv.shape[0],dv.shape[1],NUM_HIDDEN))
        l_rq_in = L.InputLayer(shape=(None,None,NUM_HIDDEN), input_var=rq_tiled) # B x N x D
        l_ev = L.ConcatLayer([l_docembed, l_rq_in, l_fembed], axis=2) # B x N x (DE+D+2)
        l_ev_lstm1 = L.LSTMLayer(l_ev, NUM_HIDDEN, grad_clipping=GRAD_CLIP, mask_input=l_docmask, \
                gradient_steps=GRAD_STEPS, precompute_input=True) # B x N x D
        l_ev_lstm1 = L.dropout(l_ev_lstm1, p=DROPOUT_RATE)
        l_ev_lstm2 = L.LSTMLayer(l_ev_lstm1, NUM_HIDDEN, grad_clipping=GRAD_CLIP, \
                mask_input=l_docmask, gradient_steps=GRAD_STEPS, precompute_input=True, \
                backwards=True) # B x N x D
        l_ev_lstm2 = L.dropout(l_ev_lstm2, p=DROPOUT_RATE)
        l_ev_lstm3 = L.LSTMLayer(L.ConcatLayer([l_ev_lstm1,l_ev_lstm2], axis=2), NUM_HIDDEN, \
                grad_clipping=GRAD_CLIP, mask_input=l_docmask, gradient_steps=GRAD_STEPS, \
                precompute_input=True) # B x N x D
        l_ev_lstm3 = L.dropout(l_ev_lstm3, p=DROPOUT_RATE)

        # crf
        l_class_in = L.ReshapeLayer(l_ev_lstm3, (dv.shape[0]*dv.shape[1],NUM_HIDDEN)) # BN x D
        l_class = L.DenseLayer(l_class_in, C, b=None, nonlinearity=None) # BN x C
        l_crf_in = L.ReshapeLayer(l_class, (dv.shape[0],dv.shape[1],C)) # B x N x C
        l_crf = CRFLayer(l_crf_in, C, mask_input=dmv, label_input=tv, normalize=False, \
                end_points=True) # 1
        l_crfdecode = CRFDecodeLayer(l_crf_in, C, W_sim=l_crf.W_sim, \
                W_end_points=l_crf.W_end_points, mask_input=dmv) # B x N

        # params
        params = L.get_all_params(l_crf, trainable=True) + \
                L.get_all_params(l_q_att_out, trainable=True)
        self.e_net = l_crf
        self.q_net = l_q_att_out

        return L.get_output(l_crf), params, L.get_output(l_crfdecode, deterministic=True)

    def load_model(self, load_path):
        with open(load_path, 'r') as f:
            data = pkl.load(f)
        L.set_all_param_values([self.e_net,self.q_net], data)

    def save_model(self, save_path):
        data = L.get_all_param_values([self.e_net,self.q_net])
        with open(save_path, 'w') as f:
            pkl.dump(data, f)

if __name__=='__main__':
    m = Model(5,3)
    d = np.asarray([[1,0,1,3],[2,2,1,4]]).astype('int32')
    q = np.asarray([[2,1],[3,1]]).astype('int32')
    dm = np.asarray([[1,1,1,0],[1,1,0,0]]).astype('int32')
    qm = np.asarray([[1,1],[1,0]]).astype('int32')
    a = np.asarray([1,1]).astype('int32')
    for i in range(10):
        print 'loss ', m.train(d,q,a,dm,qm,[])[0]
        print 'acc ', m.validate(d,q,a,dm,qm,[])[1]
