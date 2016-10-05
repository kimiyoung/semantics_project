import numpy as np
import theano
import lasagne
import theano.tensor as T
import lasagne.layers as L

CHAR_DIM = 50
NUM_CHARS = 77
MAX_WORD_LEN = 10
BATCH = 32
SEQ1_LEN = 2048
SEQ2_LEN = 226
EMBED_DIM = 100

class Model:

    def __init__(self, num_chars, char_dim, max_word_len, embed_dim):
        self.num_chars = num_chars
        self.char_dim = char_dim
        self.max_word_len = max_word_len
        self.embed_dim = embed_dim

        chars1, chars2 = T.itensor3(), T.itensor3()
        mask1, mask2 = T.btensor3(), T.btensor3()
        self.inps = [chars1, chars2, mask1, mask2]
        l_e1, l_e2 = self.build_network()

        self.fn = theano.function(self.inps, [L.get_output(l_e1), L.get_output(l_e2)])

    def build_network(self):
        l_char1_in = L.InputLayer(shape=(None, None, self.max_word_len), 
                input_var=self.inps[0])
        l_char2_in = L.InputLayer(shape=(None, None, self.max_word_len), 
                input_var=self.inps[1])
        l_mask1_in = L.InputLayer(shape=(None, None, self.max_word_len), 
                input_var=self.inps[2])
        l_mask2_in = L.InputLayer(shape=(None, None, self.max_word_len), 
                input_var=self.inps[3])
        l_char_in = L.ConcatLayer([l_char1_in, l_char2_in], axis=1) # B x (ND+NQ) x L
        l_char_mask = L.ConcatLayer([l_mask1_in, l_mask2_in], axis=1)
        shp = (self.inps[0].shape[0], self.inps[0].shape[1]+self.inps[1].shape[1], 
                self.inps[1].shape[2])
        l_index_reshaped = L.ReshapeLayer(l_char_in, (shp[0]*shp[1],shp[2])) # BN x L
        l_mask_reshaped = L.ReshapeLayer(l_char_mask, (shp[0]*shp[1],shp[2])) # BN x L
        l_lookup = L.EmbeddingLayer(l_index_reshaped, self.num_chars, self.char_dim) # BN x L x D
        l_fgru = L.GRULayer(l_lookup, 2*self.char_dim, grad_clipping=10, 
                gradient_steps=-1, precompute_input=True,
                only_return_final=True, mask_input=l_mask_reshaped)
        l_bgru = L.GRULayer(l_lookup, 2*self.char_dim, grad_clipping=10, 
                gradient_steps=-1, precompute_input=True, 
                backwards=True, only_return_final=True,
                mask_input=l_mask_reshaped) # BN x 2D
        l_fwdembed = L.DenseLayer(l_fgru, self.embed_dim/2, nonlinearity=None) # BN x DE
        l_bckembed = L.DenseLayer(l_bgru, self.embed_dim/2, nonlinearity=None) # BN x DE
        l_embed = L.ElemwiseSumLayer([l_fwdembed, l_bckembed], coeffs=1)
        l_char_embed = L.ReshapeLayer(l_embed, (shp[0],shp[1],self.embed_dim/2))
        l_embed1 = L.SliceLayer(l_char_embed, slice(0,self.inps[0].shape[1]), axis=1)
        l_embed2 = L.SliceLayer(l_char_embed, slice(-self.inps[1].shape[1],None), axis=1)
        return l_embed1, l_embed2

def get_batch(num_chars, shp1, shp2):
    return (np.random.randint(low=0, high=num_chars, size=shp1, dtype='int32'),
            np.random.randint(low=0, high=num_chars, size=shp2, dtype='int32'),
            np.random.randint(low=0, high=2, size=shp1, dtype='int8'),
            np.random.randint(low=0, high=2, size=shp2, dtype='int8'))

if __name__=='__main__':
    m = Model(NUM_CHARS, CHAR_DIM, MAX_WORD_LEN, EMBED_DIM)
    for ite in range(100):
        c1, c2, m1, m2 = get_batch(NUM_CHARS, 
                (BATCH, SEQ1_LEN, MAX_WORD_LEN), 
                (BATCH, SEQ2_LEN, MAX_WORD_LEN))
        out = m.fn(c1,c2,m1,m2)
        print "Iteration {}".format(ite)
