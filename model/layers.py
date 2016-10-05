import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L

def theano_logsumexp(x, axis=None):
    xmax = x.max(axis=axis, keepdims=True)
    xmax_ = x.max(axis=axis)
    return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

class IndexLayer(L.MergeLayer):
    """
    Layer which takes two inputs: a tensor D with arbitrary shape, and integer values,
    and a 2D lookup tensor whose rows are indices in D. Returns the first tensor with
    its each value replaced by the lookup from second tensor. This is similar to 
    EmbeddingLayer, but the lookup matrix is not a parameter, and can be of arbitrary
    size
    """

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0] + (input_shapes[1][-1],)

    def get_output_for(self, inputs, **kwargs):
        return inputs[1][inputs[0]]

class CRFLayer(L.Layer):

    def __init__(self, incoming, num_classes, W_sim = lasagne.init.GlorotUniform(), mask_input = None, label_input = None, normalize = False, end_points = False, **kwargs):

        super(CRFLayer, self).__init__(incoming, **kwargs)
        self.num_classes = num_classes
        self.W_sim = self.add_param(W_sim, (num_classes, num_classes))
        self.mask_input = mask_input
        self.label_input = label_input
        self.normalize = normalize
        self.end_points = end_points
        if end_points:
            self.W_end_points = self.add_param(lasagne.init.GlorotUniform(), (2, num_classes))
        else:
            self.W_end_points = None

    def get_output_shape_for(self, input_shape):
        return (1, )

    def get_output_for(self, input, **kwargs):
        def norm_fn(f, mask, label, previous, W_sim):
            # f: batch * class, mask: batch, label: batch, previous: batch * class, W_sim: class * class
            # previous: batch * class

            next = previous.dimshuffle(0, 1, 'x') + f.dimshuffle(0, 'x', 1) + W_sim.dimshuffle('x', 0, 1) # batch * class * class
            next = theano_logsumexp(next, axis = 1) # batch * class
            mask = mask.dimshuffle(0, 'x')
            next = previous * (1.0 - mask) + next * mask
            return next

        f = input # batch * time * class
        if self.end_points:
            for i in range(self.num_classes):
                f = T.inc_subtensor(f[:, 0, i], self.W_end_points[0, i])
                f = T.inc_subtensor(f[:, -1, i], self.W_end_points[1, i])

        initial = f[:, 0, :]
        outputs, _ = theano.scan(fn = norm_fn, \
         sequences = [f.dimshuffle(1, 0, 2)[1: ], self.mask_input.dimshuffle(1, 0)[1: ], self.label_input.dimshuffle(1, 0)[1:]], \
         outputs_info = initial, non_sequences = [self.W_sim], strict = True)
        norm = T.sum(theano_logsumexp(outputs[-1], axis = 1))

        f_pot = (f.reshape((-1, f.shape[-1]))[T.arange(f.shape[0] * f.shape[1]), self.label_input.flatten()] * self.mask_input.flatten()).sum()

        labels = self.label_input # batch * time
        shift_labels = T.roll(labels, -1, axis = 1)
        mask = self.mask_input # batch * time
        shift_mask = T.roll(mask, -1, axis = 1)

        g_pot = (self.W_sim[labels.flatten(), shift_labels.flatten()] * mask.flatten() * shift_mask.flatten()).sum()

        return - (f_pot + g_pot - norm) / f.shape[0] if self.normalize else - (f_pot + g_pot - norm)

class CRFDecodeLayer(L.Layer):

    def __init__(self, incoming, num_classes, W_sim = lasagne.init.GlorotUniform(), end_points = False, W_end_points = None, mask_input = None, **kwargs):
        super(CRFDecodeLayer, self).__init__(incoming, **kwargs)
        self.W_sim = self.add_param(W_sim, (num_classes, num_classes))
        self.mask_input = mask_input
        self.num_classes = num_classes
        self.end_points = end_points
        if self.end_points:
            self.W_end_points = self.add_param(W_end_points, (2, num_classes))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1])

    def get_output_for(self, input, **kwargs):
        def max_fn(f, mask, prev_score, prev_back, W_sim):
            next_score = prev_score.dimshuffle(0, 1, 'x') + f.dimshuffle(0, 'x', 1) + W_sim.dimshuffle('x', 0, 1)
            next_back = T.argmax(next_score, axis = 1)
            next_score = T.max(next_score, axis = 1)
            mask = mask.dimshuffle(0, 'x')
            next_score = next_score * mask + prev_score * (1.0 - mask)
            next_back = next_back * mask + prev_back * (1.0 - mask)
            next_back = T.cast(next_back, 'int32')
            return [next_score, next_back]

        def produce_fn(back, mask, prev_py):
            # back: batch * class, prev_py: batch, mask: batch
            next_py = back[T.arange(prev_py.shape[0]), prev_py]
            next_py = mask * next_py + (1.0 - mask) * prev_py
            next_py = T.cast(next_py, 'int32')
            return next_py

        f = input

        if self.end_points:
            for i in range(self.num_classes):
                f = T.inc_subtensor(f[:, 0, i], self.W_end_points[0, i])
                f = T.inc_subtensor(f[:, -1, i], self.W_end_points[1, i])

        init_score, init_back = f[:, 0, :], T.zeros_like(f[:, 0, :], dtype = 'int32')
        ([scores, backs], _) = theano.scan(fn = max_fn, \
            sequences = [f.dimshuffle(1, 0, 2)[1: ], self.mask_input.dimshuffle(1, 0)[1: ]], \
            outputs_info = [init_score, init_back], non_sequences = [self.W_sim], strict = True)

        init_py = T.argmax(scores[-1], axis = 1)
        init_py = T.cast(init_py, 'int32')
        # init_py: batch, backs: time * batch * class
        pys, _ = theano.scan(fn = produce_fn, \
            sequences = [backs, self.mask_input.dimshuffle(1, 0)[1:]], outputs_info = [init_py], go_backwards = True)
        # pys: (rev_time - 1) * batch
        pys = pys.dimshuffle(1, 0)[:, :: -1]
        # pys : batch * (time - 1)
        return T.concatenate([pys, init_py.dimshuffle(0, 'x')], axis = 1)

