import tensorflow as tf
import numpy as np
from tools import sub_sample
from config import *
from tflayers import *

EPS = 1e-7

def prepare_input(d,q):
    f = np.zeros(d.shape[:2]).astype('int32')
    for i in range(d.shape[0]):
        f[i,:] = np.in1d(d[i,:,0],q[i,:,0])
    return f

class Model:

    def __init__(self, params, vocab_size, num_chars, W_init, embed_dim, num_cand,
            cloze=True):
        self.nhidden = params['nhidden']
        self.embed_dim = embed_dim
        self.dropout = params['dropout']
        self.train_emb = params['train_emb']
        self.subsample = params['subsample']
        self.char_dim = params['char_dim']
        self.learning_rate = LEARNING_RATE
        self.num_chars = num_chars
        self.use_feat = params['use_feat']
        seed = params['seed']
        K = params['nlayers']

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(seed)
            # placeholders
            self.doc = tf.placeholder(tf.int32, shape=(None, None))
            self.qry = tf.placeholder(tf.int32, shape=(None, None))
            self.cand = tf.placeholder(tf.int32, shape=(None, None, num_cand))
            self.dmask = tf.placeholder(tf.int32, shape=(None, None))
            self.qmask = tf.placeholder(tf.int32, shape=(None, None))
            self.cmask = tf.placeholder(tf.int32, shape=(None, None))
            self.ans = tf.placeholder(tf.int32, shape=(None))
            self.feat = tf.placeholder(tf.int32, shape=(None, None))
            self.cloze = tf.placeholder(tf.int32, shape=(None))
            self.keep_prob = tf.placeholder(tf.float32)
            self.lrate = tf.placeholder(tf.float32)
            
            # variables
            if W_init is None:
                W_init = tf.random_normal((vocab_size, self.embed_dim), 
                        mean=0.0, stddev=glorot(vocab_size, self.embed_dim), 
                        dtype=tf.float32)
            self.Wemb = tf.Variable(W_init, trainable=bool(self.train_emb))
            self.Femb = tf.Variable(tf.random_normal((2,2), mean=0.0, stddev=glorot(2,2), 
                dtype=tf.float32))

            # network
            # embeddings
            doc_emb = tf.nn.embedding_lookup(self.Wemb, self.doc) # B x N x De
            qry_emb = tf.nn.embedding_lookup(self.Wemb, self.qry) # B x Q x De
            fea_emb = tf.nn.embedding_lookup(self.Femb, self.feat) # B x N x 2
            # layers
            for i in range(K):
                # doc
                indim = self.embed_dim if i==0 else 2*self.nhidden
                if self.use_feat and i==K-1:
                    doc_emb = tf.concat([doc_emb, fea_emb], axis=2) # B x N x (De+2)
                    indim += 2
                fgru = GRU(indim, self.nhidden, "docfgru%d"%i)
                bgru = GRU(indim, self.nhidden, "docbgru%d"%i, reverse=True)
                fout = fgru.compute(None, doc_emb, self.dmask) # B x N x Dh
                bout = bgru.compute(None, doc_emb, self.dmask) # B x N x Dh
                doc_emb = tf.concat([fout, bout], axis=2) # B x N x 2Dh
                # qry
                indim = self.embed_dim if i==0 else 2*self.nhidden
                fgru = GRU(indim, self.nhidden, "qryfgru%d"%i)
                bgru = GRU(indim, self.nhidden, "qrybgru%d"%i, reverse=True)
                fout = fgru.compute(None, qry_emb, self.qmask) # B x Q x Dh
                bout = bgru.compute(None, qry_emb, self.qmask) # B x Q x Dh
                qry_emb = tf.concat([fout, bout], axis=2) # B x Q x 2Dh
                # gated attention
                if i<K-1:
                    qshuf = tf.transpose(qry_emb, perm=(0,2,1)) # B x 2Dh x Q
                    M = batched_matmul(doc_emb, qshuf) # B x N x Q
                    alphas = tf.nn.softmax(M)*tf.expand_dims(tf.to_float(self.qmask), axis=1)
                    alphas = alphas/tf.reduce_sum(alphas, axis=2, keep_dims=True) # B x N x Q
                    gating = batched_matmul(alphas, qry_emb) # B x N x 2Dh
                    doc_emb = doc_emb*gating # B x N x 2Dh
                    doc_emb = tf.nn.dropout(doc_emb, self.keep_prob) 
            # attention sum
            if cloze:
                cl = tf.expand_dims(tf.one_hot(self.cloze, tf.shape(self.qry)[1]), axis=2) # B x Q x 1
                q = tf.reduce_sum(cl*qry_emb, axis=1) # B x 2Dh
            else:
                mid = self.nhidden
                q = tf.concat([qry_emb[:,-1,:mid], qry_emb[:,0,mid:]], axis=1) # B x 2Dh
            p = tf.reduce_sum(doc_emb*tf.expand_dims(q, axis=1), axis=2) # B x N
            probs = tf.nn.softmax(p) # B x N
            probm = probs*tf.to_float(self.cmask) + EPS
            probm = probm/tf.reduce_sum(probm, axis=1, keep_dims=True) # B x N
            self.probc = tf.reduce_sum(tf.expand_dims(probm, axis=2)*tf.to_float(self.cand), 
                    axis=1) # B x C

            # loss
            t1hot = tf.one_hot(self.ans, num_cand) # B x C
            self.loss = -tf.reduce_mean(tf.reduce_sum(
                tf.to_float(t1hot)*tf.log(self.probc+EPS), axis=1))
            self.acc = tf.reduce_mean(tf.cast(
                tf.equal(tf.cast(tf.argmax(self.probc,axis=1),tf.int32),self.ans), tf.float32))

            # ops
            opt = tf.train.AdamOptimizer(learning_rate=self.lrate)
            grads = opt.compute_gradients(self.loss)
            grads_clipped = [(tf.clip_by_value(gg, -GRAD_CLIP, GRAD_CLIP),var) 
                    for gg,var in grads if gg is not None]
            self.train_op = opt.apply_gradients(grads_clipped)

            # bells and whistles
            self.session = tf.Session()
            self.saver = tf.train.Saver()
            self.session.run(tf.global_variables_initializer())
            self.doc_rep = doc_emb
            self.qry_rep = qry_emb
            self.doc_probs = probs

    def anneal(self):
        self.learning_rate /= 2

    def train(self, dw, dt, qw, qt, c, a, m_dw, m_qw, tt, tm, m_c, cl, crd, crq):
        f = prepare_input(dw,qw)
        loss, acc, probs, _ = self.session.run([self.loss, self.acc, self.probc, self.train_op], 
                feed_dict = {
                    self.doc : dw[:,:,0],
                    self.qry : qw[:,:,0],
                    self.cand : c,
                    self.dmask : m_dw,
                    self.qmask : m_qw,
                    self.cmask : m_c,
                    self.ans : a,
                    self.feat : f,
                    self.cloze : cl,
                    self.keep_prob : 1.-self.dropout,
                    self.lrate : self.learning_rate,
                    })
        return loss, acc, probs

    def validate(self, dw, dt, qw, qt, c, a, m_dw, m_qw, tt, tm, m_c, cl, crd, crq):
        f = prepare_input(dw,qw)
        loss, acc, probs, drep, qrep, dprobs = self.session.run(
                [self.loss, self.acc, self.probc, self.doc_rep, self.qry_rep, self.doc_probs], 
                feed_dict = {
                    self.doc : dw[:,:,0],
                    self.qry : qw[:,:,0],
                    self.cand : c,
                    self.dmask : m_dw,
                    self.qmask : m_qw,
                    self.cmask : m_c,
                    self.ans : a,
                    self.feat : f,
                    self.cloze : cl,
                    self.keep_prob : 1.,
                    })
        return loss, acc, probs, drep, qrep, dprobs
    
    def save_model(self, save_path):
        base_path = save_path.rsplit('/',1)[0]
        self.saver.save(self.session, base_path+'model')

    def load_model(self, load_path):
        base_path = load_path.rsplit('/',1)[0]
        new_saver = tf.train.import_meta_graph(base_path+'model.meta')
        new_saver.restore(self.session, tf.train.latest_checkpoint(base_path))
