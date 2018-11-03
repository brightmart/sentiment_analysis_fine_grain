# coding=utf8
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.layers.merge import _Merge


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        input_shape = K.int_shape(x)

        features_dim = self.features_dim
        # step_dim = self.step_dim
        step_dim = input_shape[1]

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b[:input_shape[1]]

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    	# print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
    	return input_shape[0], self.features_dim
# end Attention


class JoinAttention(_Merge):
    def __init__(self, step_dim, hid_size,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism according to other vector.
        Supports Masking.
        # Input shape, list of
            2D tensor with shape: `(samples, features_1)`.
            3D tensor with shape: `(samples, steps, features_2)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            en = LSTM(64, return_sequences=False)(input)
            de = LSTM(64, return_sequences=True)(input2)
            output = JoinAttention(64, 20)([en, de])
        """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.hid_size = hid_size
        super(JoinAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A merge layer [JoinAttention] should be called '
                             'on a list of inputs.')
        if len(input_shape) != 2:
            raise ValueError('A merge layer [JoinAttention] should be called '
                             'on a list of 2 inputs. '
                             'Got ' + str(len(input_shape)) + ' inputs.')
        if len(input_shape[0]) != 2 or len(input_shape[1]) != 3:
            raise ValueError('A merge layer [JoinAttention] should be called '
                             'on a list of 2 inputs with first ndim 2 and second one ndim 3. '
                             'Got ' + str(len(input_shape)) + ' inputs.')

        self.W_en1 = self.add_weight((input_shape[0][-1], self.hid_size),
                                 initializer=self.init,
                                 name='{}_W0'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.W_en2 = self.add_weight((input_shape[1][-1], self.hid_size),
                                 initializer=self.init,
                                 name='{}_W1'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.W_de = self.add_weight((self.hid_size,),
                                 initializer=self.init,
                                 name='{}_W2'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        if self.bias:
            self.b_en1 = self.add_weight((self.hid_size,),
                                     initializer='zero',
                                     name='{}_b0'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
            self.b_en2 = self.add_weight((self.hid_size,),
                                     initializer='zero',
                                     name='{}_b1'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
            self.b_de = self.add_weight((input_shape[1][1],),
                                     initializer='zero',
                                     name='{}_b2'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b_en1 = None
            self.b_en2 = None
            self.b_de = None

        self._reshape_required = False
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape[1][0], input_shape[1][-1]

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, inputs, mask=None):
        en = inputs[0]
        de = inputs[1]
        de_shape = K.int_shape(de)
        step_dim = de_shape[1]

        hid_en = K.dot(en, self.W_en1)
        hid_de = K.dot(de, self.W_en2)
        if self.bias:
            hid_en += self.b_en1
            hid_de += self.b_en2
        hid = K.tanh(K.expand_dims(hid_en, axis=1) + hid_de)
        eij = K.reshape(K.dot(hid, K.reshape(self.W_de, (self.hid_size, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b_de[:step_dim]

        a = K.exp(eij - K.max(eij, axis=-1, keepdims=True))

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask[1], K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = de * a
        return K.sum(weighted_input, axis=1)
# end JoinAttention
