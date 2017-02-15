import time
import numpy as np
import theano
import theano.tensor as T
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan

from lasagne.layers.base import MergeLayer, Layer
from lasagne.layers.input import InputLayer
from lasagne.layers.dense import DenseLayer
from lasagne.layers.recurrent import Gate, GRULayer
from lasagne.layers import helper, get_output, get_all_params, SliceLayer
from lasagne.updates import sgd

class CorefGRULayer(MergeLayer):
    r"""
    lasagne.layers.recurrent.GRULayer(incomings, num_units,
    resetgate=lasagne.layers.Gate(W_cell=None),
    updategate=lasagne.layers.Gate(W_cell=None),
    hidden_update=lasagne.layers.Gate(
    W_cell=None, lasagne.nonlinearities.tanh),
    hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=False,
    gradient_steps=-1, grad_clipping=0, unroll_scan=False,
    precompute_input=True, mask_input=None, only_return_final=False, **kwargs)

    Gated Recurrent Unit (GRU) Layer with explicit Coref states

    """
    def __init__(self, incomings, num_units, num_coref,
                 resetgate=Gate(W_cell=None),
                 updategate=Gate(W_cell=None),
                 hidden_update=Gate(W_cell=None,
                                    nonlinearity=nonlinearities.tanh),
                 hid_init=init.Constant(0.),
                 hid_init_slow=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 propagate_nocoref=False,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have four 
        # inputs - the layer input, the mask, the corefs, and the initial hidden states
        # We will provide the layer input and corefs as incomings, and a mask input
        # or initial hidden state can be provided separately
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        self.hid_init_slow_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1
        if isinstance(hid_init_slow, Layer):
            incomings.append(hid_init_slow)
            self.hid_init_slow_incoming_index = len(incomings)-1

        # Initialize parent layer
        super(CorefGRULayer, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
        self.num_units = num_units
        self.num_coref = num_coref
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final
        self.propagate_nocoref = propagate_nocoref

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    # hidden state will be concatenation of local and coref states
                    self.add_param(gate.W_hid, (2*num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in all parameters from gates
        (self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')

        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')

        # Initialize hidden state
        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)
        # slow hidden state (add one for no coref)
        if isinstance(hid_init_slow, Layer):
            self.hid_init_slow = hid_init_slow
        else:
            self.hid_init_slow = self.add_param(
                hid_init_slow, (1,self.num_coref+1, self.num_units), name="hid_init_slow",
                trainable=learn_init, regularizable=False)

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable

        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        coref = inputs[1]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        hid_init_slow = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.hid_init_slow_incoming_index > 0:
            hid_init_slow = inputs[self.hid_init_slow_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        coref = coref.dimshuffle(1, 0)
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)

        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0)

        if self.precompute_input:
            # precompute_input inputs*W. W_in is (n_features, 3*num_units).
            # input is then (n_batch, n_time_steps, 3*num_units).
            input = T.dot(input, W_in_stacked) + b_stacked

        # At each call to scan, input_n will be (n_time_steps, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        def step(input_n, coref_n, hid_prev, hid_previous_slow, *args):
            nb = hid_previous_slow.shape[0]
            # select the current slow hidden state
            ent_previous = hid_previous_slow[T.arange(nb), coref_n, :]
            # concatenate with fast hid state
            hid_previous = T.concatenate([hid_prev, ent_previous], axis=1)

            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            hid_input = T.dot(hid_previous, W_hid_stacked)

            if self.grad_clipping:
                input_n = theano.gradient.grad_clip(
                    input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

            if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # Reset and update gates
            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)

            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
            hidden_update_in = slice_w(input_n, 2)
            hidden_update_hid = slice_w(hid_input, 2)
            hidden_update = hidden_update_in + resetgate*hidden_update_hid
            if self.grad_clipping:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hid(hidden_update)

            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid = (1 - updategate)*hid_prev + updategate*hidden_update

            # slice slow states
            hid_slow = hid.copy()
            if not self.propagate_nocoref:
                hid_slow = T.switch(coref_n[:,None], hid_slow, T.zeros((nb,self.num_units)))
            hid_new_slow = T.set_subtensor(
                    hid_previous_slow[T.arange(nb),coref_n,:], 
                    hid_slow)

            return hid, hid_new_slow

        def step_masked(input_n, coref_n, mask_n, 
                hid_previous, hid_previous_slow, *args):
            hid, hid_new_slow = step(input_n, coref_n, 
                hid_previous, hid_previous_slow, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            hid = T.switch(mask_n, hid, hid_previous)
            hid_slow = T.switch(mask_n, hid_new_slow.flatten(ndim=2), 
                    hid_previous_slow.flatten(ndim=2))
            hid_slow = hid_slow.reshape(hid_previous_slow.shape)

            return [hid, hid_slow]

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, coref, mask]
            step_fun = step_masked
        else:
            sequences = [input, coref]
            step_fun = step

        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)
        if not isinstance(self.hid_init_slow, Layer):
            hid_init_slow = T.tile(self.hid_init_slow, (num_batch, 1, 1))

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            hid_states = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init, hid_init_slow],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            hid_states = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init, hid_init_slow],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        hid_out = hid_states[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        # concatenate coref memory to the usual output
        hid_out = T.concatenate([hid_out, hid_states[1][-1]], axis=1)

        return hid_out

if __name__=="__main__":
    # hyperparameters
    d_in = 2
    d = 2
    N = 5

    # set parameters (copy for r, z, h gates)
    W = np.asarray([[1.,0.],
        [0.,1.]]).astype('float32').transpose()
    U = np.asarray([[1.,0.,0.,0.],
        [0.,0.,0.,1.]]).astype('float32').transpose()
    b = np.asarray([0.,1.]).astype('float32')

    # construct gates
    r = Gate(W_in=W, W_hid=U, b=b, W_cell=None)
    u = Gate(W_in=W, W_hid=U, b=b, W_cell=None)
    h = Gate(W_in=W, W_hid=U, b=b, W_cell=None, 
            nonlinearity=nonlinearities.tanh)

    # set inputs
    X = np.asarray([[1.,1.],[0.,0.],[1.,0.],[0.,1.],
        [2.,2.]]).astype('float32')
    Y = np.asarray([1,0,1,2,1]).astype('int32')
    M = np.asarray([1,1,1,0,0]).astype('int32')
    hi = np.asarray([[[0,0],
        [0,1],
        [1,0]]]).astype('float32')

    # set output
    H = np.asarray([[0.55,0.94],
        [0.42,0.81],
        [0.78,0.93],
        [0.78,0.93],
        [0.78,0.93]]).astype('float32')
    HF = np.asarray([[0,0.78,1.],
        [0,0.93,0.]]).astype('float32').transpose()

    # build network (use batch size of 1)
    x = T.ftensor3()
    y = T.imatrix()
    m = T.imatrix()
    hs = T.ftensor3()
    l_in_x = InputLayer(shape=(1,N,d_in), input_var=x)
    l_in_y = InputLayer(shape=(1,N), input_var=y)
    l_in_m = InputLayer(shape=(1,N), input_var=m)
    l_in_h = InputLayer(shape=(1,3,d), input_var=hs)
    l_cgru = CorefGRULayer([l_in_x,l_in_y], 2, 2,
            resetgate=r,
            updategate=u,
            hidden_update=h,
            hid_init_slow=l_in_h,
            mask_input = l_in_m)
    l_hid = SliceLayer(l_cgru, indices=slice(0,-3), axis=1)
    l_hidslow = SliceLayer(l_cgru, indices=slice(-3,None), axis=1)
    l_gru = GRULayer(l_in_x, 4, mask_input = l_in_m)

    # test
    f = theano.function([x,y,m,hs], [get_output(l_hid),get_output(l_hidslow)], 
            on_unused_input='ignore')
    print("Expected output:")
    print(H)
    print("Actual output:")
    out, hf = f(X[np.newaxis,:,:],Y[np.newaxis,:],M[np.newaxis,:],hi)
    print(out)
    if np.isclose(H[np.newaxis,:], out, rtol=0., atol=0.02).all():
        print("Test passed!")
    else:
        print("Test failed :(")
    print("Expected final coref state:")
    print(HF)
    print("Actual final coref state:")
    print(hf)

    # time
    cgru = get_output(l_cgru)
    gru = get_output(l_gru)
    loss_cgru = T.sum(cgru)
    loss_gru = T.sum(gru)
    params_cgru = get_all_params(l_cgru, trainable=True)
    params_gru = get_all_params(l_gru, trainable=True)
    updates_cgru = sgd(loss_cgru, params_cgru, learning_rate=0.1)
    updates_gru = sgd(loss_gru, params_gru, learning_rate=0.1)
    f_cgru = theano.function([x,y,m,hs], loss_cgru, updates=updates_cgru)
    f_gru = theano.function([x,y,m], loss_gru, updates=updates_gru, on_unused_input='ignore')
    tst = time.time()
    for i in range(100):
        out = f_cgru(X[np.newaxis,:,:],Y[np.newaxis,:],M[np.newaxis,:],hi)
    print "Coref GRU elapsed time = %.3f" % (time.time()-tst)
    tst = time.time()
    for i in range(100):
        out = f_gru(X[np.newaxis,:,:],Y[np.newaxis,:],M[np.newaxis,:])
    print "Coref GRU elapsed time = %.3f" % (time.time()-tst)
