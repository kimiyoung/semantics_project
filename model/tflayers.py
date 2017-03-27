import tensorflow as tf
import numpy as np
import cPickle as pkl

def batched_matmul(x,y):
    # x : B x M x N, y : B x N x Q
    fn = lambda x: tf.matmul(x[0], x[1])
    return tf.map_fn(fn, [x,y], dtype=tf.float32) # B x M x Q

def glorot(d1,d2):
    return np.sqrt(6./(d1+d2))

class GRU(object):
    def __init__(self, idim, odim, name, reverse=False):
        def _gate_params(insize, outsize, name):
            gate = {}
            gate["W"] = tf.Variable(tf.random_normal((insize,outsize), 
                mean=0.0, stddev=glorot(insize,outsize)),
                name="W"+name, dtype=tf.float32)
            gate["U"] = tf.Variable(tf.random_normal((outsize,outsize), 
                mean=0.0, stddev=glorot(outsize,outsize)),
                name="U"+name, dtype=tf.float32)
            gate["b"] = tf.Variable(tf.zeros((outsize,)), 
                name="U"+name, dtype=tf.float32)
            return gate

        self.resetgate = _gate_params(idim, odim, "r"+name)
        self.updategate = _gate_params(idim, odim, "r"+name)
        self.hiddengate = _gate_params(idim, odim, "r"+name)
        self.reverse = reverse
        self.out_dim = odim

    @staticmethod
    def _gru_cell(prev, inp, rgate, ugate, hgate):
        r = tf.sigmoid(tf.matmul(inp,rgate["W"]) + tf.matmul(prev,rgate["U"]) + 
                rgate["b"])
        z = tf.sigmoid(tf.matmul(inp,ugate["W"]) + tf.matmul(prev,ugate["U"]) + 
                ugate["b"])
        ht = tf.tanh(tf.matmul(inp,hgate["W"]) + r*tf.matmul(prev,hgate["U"]) + 
                hgate["b"])
        h = (1.-z)*prev + z*ht
        return h

    def _step_gru(self, prev, inps):
        # prev : B x Do
        # inps : (B x Di, B)
        elems, mask = inps[0], inps[1]
        new = self._gru_cell(prev, elems, 
                self.resetgate, self.updategate, self.hiddengate)
        newmasked = tf.expand_dims((1.-tf.to_float(mask)),axis=1)*prev + \
                tf.expand_dims(tf.to_float(mask),axis=1)*new
        return newmasked

    def compute(self, init, inp, mask):
        # init : B x Do
        # inp : B x N x Di
        # mask : B x N
        if self.reverse:
            inp = tf.reverse(inp, [1])
            mask = tf.reverse(mask, [1])
        if init is None:
            init = tf.zeros((tf.shape(inp)[0],self.out_dim), dtype=tf.float32)
        inpre = tf.transpose(inp, perm=(1,0,2)) # N x B x Di
        maskre = tf.transpose(mask, perm=(1,0)) # N x B
        outs = tf.transpose(tf.scan(self._step_gru, (inpre, maskre), 
                initializer=init), perm=(1,0,2)) # B x N x Do
        if self.reverse:
            outs = tf.reverse(outs, [1])
        return outs

class MageRNN(object):
    """
    MAGE RNN model which takes as input five tensors:
    X: B x N x Din (batch of sequences)
    M: B x N (masks for sequences)
    Ei: B x N x C (one-hot mask for incoming edges at each timestep)
    Eo: B x N x C (one-hot mask for outgoing edges at each timestep)
    Ri: B x N x C (index for incoming relation types at each timestep)
    Ro: B x N x C (index for outgoing relation types at each timestep)
    Q: B x Dout (optionally, query vectors for modulating relation attention)
    C is the maximum number of chains.
    Indexes in Ri/o go from 0 to M-1, where M is the number of relation types

    During each recurrent update, the relevant hidden states are fetched from the memory
    based on R/E, and combined through an attention mechanism. Multiple edges of the same type
    are averaged together.

    During construction provide the parameters for initializing the variables.
        "num_relations" :   total number of relations,
        "output_dim"    :   output dimensionality for each relation type,
        "input_dim"     :   input dimensionality
        "max_chains"    :   maximum number of chains

    compute() takes the three placeholders for the tensors described above, and optionally the
    initializer for the recurrence, and outputs
    another placeholder for the output of the layer.
    """

    def __init__(self, num_relations, input_dim, output_dim, max_chains):
        self.num_relations = num_relations
        self.rdims = output_dim
        self.input_dim = input_dim
        self.output_dim = self.rdims*self.num_relations
        self.max_chains = max_chains

        # initialize gates
        def _gate_params(name):
            gate = {}
            gate["W"] = tf.Variable(tf.random_normal((self.input_dim,self.output_dim), 
                mean=0.0, stddev=glorot(self.input_dim,self.output_dim)),
                name="W"+name, dtype=tf.float32)
            gate["U"] = tf.Variable(tf.random_normal((self.num_relations,self.rdims,self.output_dim),
                mean=0.0, stddev=glorot(self.rdims,self.output_dim)),
                name="U"+name, dtype=tf.float32)
            gate["b"] = tf.Variable(tf.zeros((self.output_dim,)), 
                name="b"+name, dtype=tf.float32)
            return gate
        self.resetgate = _gate_params("r")
        self.updategate = _gate_params("u")
        self.hiddengate = _gate_params("h")

        # initialize attention params
        self.Rkeys = tf.Variable(tf.random_normal((self.num_relations,self.rdims),
            mean=0.0, stddev=0.1),
            name="Rkeys", dtype=tf.float32) # R x Dr
        self.Watt = tf.Variable(tf.random_normal((self.rdims,self.input_dim),
            mean=0.0, stddev=glorot(self.rdims,self.input_dim)),
            name="Watt", dtype=tf.float32) # Dr x Din
        self.Uatt = tf.Variable(tf.random_normal((self.rdims,self.input_dim),
            mean=0.0, stddev=glorot(self.rdims,self.input_dim)),
            name="Uatt", dtype=tf.float32) # Dr x Din

    def compute(self, X, M, Ei, Eo, Ri, Ro, init=None):
        # reshape for scan
        Xre = tf.transpose(X, perm=(1,0,2))
        Mre = tf.transpose(M, perm=(1,0))
        Eire = tf.transpose(Ei, perm=(1,0,2))
        Eore = tf.transpose(Eo, perm=(1,0,2))
        Rire = tf.transpose(Ri, perm=(1,0,2))
        Rore = tf.transpose(Ro, perm=(1,0,2))

        # update
        if init is None: init = tf.zeros((tf.shape(X)[0], self.output_dim), dtype=tf.float32)
        mem_init = tf.zeros((tf.shape(X)[0], self.max_chains, self.rdims), dtype=tf.float32)
        outs, mems = tf.scan(self._step, (Xre, Mre, Eire, Eore, Rire, Rore), 
                initializer=(init,mem_init)) # N x B x Dout

        return tf.transpose(outs, perm=(1,0,2)), tf.transpose(mems, perm=(1,0,2,3))

    def _mem_read(self, e):
        # e : B x R x C
        N = tf.shape(self.memory)[1]
        e1hot = tf.one_hot(e, N, axis=1) # B x N x R x C
        e_r = tf.expand_dims(e1hot, axis=4) # B x N x R x C x 1
        m_r = tf.expand_dims(self.memory, axis=3) # B x N x R x 1 x Dr
        return tf.reduce_sum(e_r*m_r, axis=[1,3]) # B x R x Dr

    def _mem_write(self, m, i):
        # m : B x Dout
        m_r = tf.reshape(m, [tf.shape(m)[0],self.num_relations, self.rdims]) # B x R x Dr
        s1 = tf.slice(self.memory, [0,0,0,0], [-1,i+1,-1,-1]) # B x (i+1) x R x Dr
        s2 = tf.slice(self.memory, [0,i+2,0,0], [-1,-1,-1,-1]) # B x (N-i-1) x R x Dr
        self.memory.assign(tf.concat([s1,tf.expand_dims(m_r,axis=1),s2], axis=1))
        self.memory.set_shape([None,None,self.num_relations, self.rdims])

    def _attention(self, x, c, e, r):
        EPS = 1e-7
        # x : B x Din, m : B, c : B x C x Dr, e : B x C, r : B x C
        rd = tf.nn.embedding_lookup(self.Rkeys, r) # B x C x Dr
        v = tf.tanh(tf.tensordot(c, self.Watt, [[2],[0]]) + 
                tf.tensordot(rd, self.Uatt, [[2],[0]])) # B x C x Din
        actvs = tf.reduce_sum(tf.expand_dims(x, axis=1)*v, axis=2) # B x C
        alphas = tf.nn.softmax(actvs) # B x C
        alphas_m = alphas*tf.to_float(e) + EPS # mask
        return alphas_m/tf.reduce_sum(alphas_m, keep_dims=True)

    @staticmethod
    def _getU(r, gate):
        # r : B x C, gate["U"] : R x Dr x Dout
        return tf.gather(gate["U"], r) # B x C x Dr x Dout

    def _hid_to_hid(self, h, r, gate):
        U = self._getU(r, gate) # B x C x Dr x Dout
        h_r = tf.expand_dims(h, axis=3) # B x C x Dr x 1
        return tf.reduce_sum(h_r*U, axis=[1,2]) # B x Dout

    def _hid_prev(self, x, c, e, r):
        alphas = self._attention(x, c, e, r) # B x C
        h = tf.expand_dims(alphas, axis=2)*c # B x C x Dr
        return h

    def _step(self, prev, inps):
        hprev, mprev = prev[0], prev[1] # hprev : B x Dout, mprev : B x C x Dr
        x, m, ei, eo, ri, ro = inps[0], inps[1], inps[2], inps[3], \
                inps[4], inps[5] # x : B x Din, m : B, ei/o : B x C, ri/o : B x C

        hnew = self._gru_cell(x, mprev, ei, ri, self.resetgate, self.updategate,
                self.hiddengate) # B x Dout
        hnew_r = tf.reshape(hnew, 
                [tf.shape(x)[0], self.num_relations, self.rdims]) # B x R x Dr
        ro1hot = tf.one_hot(ro, self.num_relations, axis=2) # B x C x R
        mnew = tf.reduce_sum(
                tf.expand_dims(hnew_r, axis=1)*tf.expand_dims(ro1hot, axis=3), 
                axis=[2]) # B x C x Dr
        hnew.set_shape([None,self.output_dim])

        m_r = tf.to_float(tf.expand_dims(m, axis=1)) # B x 1
        hnew = (1.-m_r)*hprev + m_r*hnew

        eo_r = tf.expand_dims(m_r*tf.to_float(eo), axis=2) # B x C x 1
        mnew = (1.-eo_r)*mprev + eo_r*mnew

        return hnew, mnew

    def _gru_cell(self, x, c, e, ri, rgate, ugate, hgate):
        prev = self._hid_prev(x, c, e, ri) # B x C x Dr
        r = tf.sigmoid(tf.matmul(x,rgate["W"]) + \
                self._hid_to_hid(prev, ri, rgate) + \
                rgate["b"])
        z = tf.sigmoid(tf.matmul(x,ugate["W"]) + \
                self._hid_to_hid(prev, ri, ugate) + \
                ugate["b"])
        ht = tf.tanh(tf.matmul(x,hgate["W"]) + \
                r*self._hid_to_hid(prev, ri, hgate) + \
                hgate["b"])
        hp = tf.tile(tf.reduce_sum(prev, axis=1), [1,self.num_relations]) # B x Dout
        h = (1.-z)*hp + z*ht
        return h

def test_magernn():
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(0)
        mage = MageRNN(2,2,2,2)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            def _get_gate(gate):
                return [gate["W"].eval(), gate["U"].eval(), gate["b"].eval()]

            X = tf.constant([[[1,0],[0,1],[0,0]]], dtype=tf.float32)
            M = tf.constant([[1,1,0]], dtype=tf.int32)
            Ei = tf.constant([[[0,0],[1,1],[1,0]]], dtype=tf.int32)
            Eo = tf.constant([[[1,1],[1,0],[0,0]]], dtype=tf.int32)
            Ri = tf.constant([[[0,0],[0,1],[0,0]]], dtype=tf.int32)
            Ro = tf.constant([[[0,1],[0,0],[0,0]]], dtype=tf.int32)
            O, MEM = mage.compute(X, M, Ei, Eo, Ri, Ro)
            var = [_get_gate(mage.resetgate), _get_gate(mage.updategate), _get_gate(mage.hiddengate),
                    mage.Rkeys.eval(), mage.Watt.eval(), mage.Uatt.eval()]
            pkl.dump(var, open('tmp_params.p','wb'))
            print sess.run([O,MEM])

if __name__=="__main__":
    test_magernn()
