import tensorflow as tf


class GAT(tf.keras.layers.Layer):
    """
    Graph Attention Layer according to https://arxiv.org/pdf/1710.10903.pdf
    """
    def __init__(self, output_dim, use_bias, activation, num_heads=4, kernel_regularizer=None, **kwargs):
        super(GAT, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)

        self.num_heads = num_heads
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.attentions = [Attention(self.output_dim, self.kernel_regularizer) for _ in range(num_heads)]

    def build(self, input_shape):
        super(GAT, self).build(input_shape)
        input_dim = input_shape[0][-1]
        kernel_shape = (self.num_heads, input_dim, self.output_dim)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer='glorot_uniform',
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.num_heads*self.output_dim,),
                                        initializer='glorot_uniform',
                                        name='bias',
                                        trainable=True)
        else:
            self.bias = None

    def call(self, inputs):
        attn_output = [self.attentions[k](inputs) for k in range(self.num_heads)]
        output = [tf.keras.backend.batch_dot(attn, inputs[0]) for attn in attn_output]
        output = [tf.keras.backend.dot(output[i], self.kernel[i]) for i in range(self.num_heads)]
        output = tf.keras.backend.concatenate(output, axis=-1)
        # output = tf.keras.backend.dot(output, self.kernel)

        if self.use_bias:
            output = tf.keras.backend.bias_add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'use_bias': self.use_bias,
            'activation': self.activation,
            'num_heads': self.num_heads,
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer)
        })
        return config


class Attention(tf.keras.layers.Layer):
    def __init__(self, output_dim, kernel_regularizer=None, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        kernel_shape = (input_dim, self.output_dim)

        self.W = self.add_weight(shape=kernel_shape,
                                 initializer='glorot_uniform',
                                 name='W',
                                 regularizer=self.kernel_regularizer,
                                 trainable=True)

        self.a = self.add_weight(shape=(2, self.output_dim, 1),
                                 initializer='glorot_uniform',
                                 name='a',
                                 regularizer=self.kernel_regularizer,
                                 trainable=True)

    def call(self, inputs):
        X = inputs[0]
        A = inputs[1]

        x_features = tf.keras.backend.dot(X, self.W)  # (N, output_dim)
        att_self = tf.keras.backend.dot(x_features, self.a[0])  # (N, 1)
        att_neighbours = tf.keras.backend.dot(x_features, self.a[1])  # (N, 1)
        att = att_self + tf.transpose(att_neighbours, perm=[0, 2, 1])  # (N, N)
        att = tf.keras.layers.LeakyReLU(alpha=0.2)(att)

        mask = -10e9 * (1.0 - A)
        att_masked = att + mask
        # att_masked = att * A
        dense = tf.keras.layers.Softmax()(att_masked)

        return dense

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer)
        })
        return config


class NoGraphConv(tf.keras.layers.Layer):
    """
    No graph convoluion.
    """
    def __init__(self, output_dim, use_bias, activation, kernel_regularizer=None, **kwargs):
        super(NoGraphConv, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        kernel_shape = (input_dim, self.output_dim)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer='glorot_uniform',
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer='glorot_uniform',
                                        name='bias',
                                        trainable=True)
        else:
            self.bias = None

    def _normalize(self, A):
        n = tf.shape(A)[-1]
        I = tf.cast(tf.eye(n), dtype=A.dtype)[tf.newaxis, :, :]
        return I

    def call(self, inputs):
        output = tf.keras.backend.batch_dot(self._normalize(inputs[1]), inputs[0])
        output = tf.keras.backend.dot(output, self.kernel)

        if self.use_bias:
            output = tf.keras.backend.bias_add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'use_bias': self.use_bias,
            'activation': self.activation,
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer)
        })
        return config


class MultiGraphConv(tf.keras.layers.Layer):
    """
    Graph Convolution Layer according to https://arxiv.org/pdf/1907.05008.pdf
    """

    def __init__(self, output_dim, use_bias, activation, kernel_regularizer=None, **kwargs):
        super(MultiGraphConv, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        kernel_shape = (3*input_dim, self.output_dim)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer='glorot_uniform',
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer='glorot_uniform',
                                        name='bias',
                                        trainable=True)
        else:
            self.bias = None

    def _normalize(self, A, eps=1e-6):
        n = tf.shape(A)[-1]
        A -= tf.linalg.diag(tf.linalg.diag_part(A))
        A_hat = A + tf.cast(tf.eye(n), dtype=A.dtype)[tf.newaxis, :, :]
        deg = tf.reduce_sum(A_hat, axis=2)

        D_symm = tf.linalg.diag(1./(eps + tf.math.sqrt(deg)))
        D_asymm = tf.linalg.diag(1./(eps + deg))

        return [A, tf.matmul(D_asymm, A_hat), tf.matmul(tf.matmul(D_symm, A_hat), D_symm)]

    def call(self, inputs):
        output = [tf.keras.backend.batch_dot(_A, inputs[0]) for _A in self._normalize(inputs[1])]
        output = tf.keras.backend.concatenate(output, axis=-1)
        output = tf.keras.backend.dot(output, self.kernel)

        if self.use_bias:
            output = tf.keras.backend.bias_add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'use_bias': self.use_bias,
            'activation': self.activation,
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer)
        })
        return config


class SAGEConv(tf.keras.layers.Layer):
    """
        GraphSAGE Layer according to https://arxiv.org/abs/1706.02216
    """

    def __init__(self, output_dim, use_bias, activation, kernel_regularizer=None, **kwargs):
        super(SAGEConv, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        kernel_shape = (2*input_dim, self.output_dim)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer='glorot_uniform',
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer='glorot_uniform',
                                        name='bias',
                                        trainable=True)
        else:
            self.bias = None

    def _normalize(self, A, eps=1e-6):
        A -= tf.linalg.diag(tf.linalg.diag_part(A))
        D = tf.linalg.diag(1./(eps + tf.reduce_sum(A, axis=2)))
        return tf.matmul(D, A)

    def call(self, inputs):
        output = tf.keras.backend.batch_dot(self._normalize(inputs[1]), inputs[0])
        output = tf.keras.backend.concatenate([output, inputs[0]], axis=-1)
        output = tf.keras.backend.dot(output, self.kernel)

        if self.use_bias:
            output = tf.keras.backend.bias_add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'use_bias': self.use_bias,
            'activation': self.activation,
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer)
        })
        return config


class GraphConv(tf.keras.layers.Layer):
    """
         Graph Convolution Layer according to (T. Kipf and M. Welling, ICLR 2017)
    """

    def __init__(self, output_dim, use_bias, activation, kernel_regularizer=None, **kwargs):
        super(GraphConv, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        kernel_shape = (input_dim, self.output_dim)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer='glorot_uniform',
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer='glorot_uniform',
                                        name='bias',
                                        trainable=True)
        else:
            self.bias = None

    def _normalize(self, A, eps=1e-6):
        n = tf.shape(A)[-1]
        A -= tf.linalg.diag(tf.linalg.diag_part(A))
        A_hat = A + tf.cast(tf.eye(n), dtype=A.dtype)[tf.newaxis, :, :]
        D_hat = tf.linalg.diag(1./(eps + tf.math.sqrt(tf.reduce_sum(A_hat, axis=2))))
        return tf.matmul(tf.matmul(D_hat, A_hat), D_hat)

    def call(self, inputs):
        output = tf.keras.backend.batch_dot(self._normalize(inputs[1]), inputs[0])
        output = tf.keras.backend.dot(output, self.kernel)

        if self.use_bias:
            output = tf.keras.backend.bias_add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'use_bias': self.use_bias,
            'activation': self.activation,
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer)
        })
        return config


class ChebConv(tf.keras.layers.Layer):
    """
        ChebNet graph convolution according to https://arxiv.org/abs/1606.09375
    """

    def __init__(self, output_dim, use_bias, activation, K=4, kernel_regularizer=None, **kwargs):
        super(ChebConv, self).__init__(**kwargs)

        self.K = K
        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        kernel_shape = (self.K*input_dim, self.output_dim)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer='glorot_uniform',
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer='glorot_uniform',
                                        name='bias',
                                        trainable=True)
        else:
            self.bias = None

    def _normalize(self, A, eps=1e-6):
        A -= tf.linalg.diag(tf.linalg.diag_part(A))
        D = tf.linalg.diag(1./(eps + tf.math.sqrt(tf.reduce_sum(A, axis=2))))
        return tf.matmul(tf.matmul(D, A), D)

    def call(self, inputs):
        L = self._normalize(inputs[1])

        Xt = [inputs[0]]
        Xt.append(tf.keras.backend.batch_dot(L, inputs[0]))
        for k in range(2, self.K):
            Xt.append(2*tf.keras.backend.batch_dot(L, Xt[k-1]) - Xt[k-2])
        output = tf.keras.backend.concatenate(Xt, axis=-1)
        output = tf.keras.backend.dot(output, self.kernel)

        if self.use_bias:
            output = tf.keras.backend.bias_add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'use_bias': self.use_bias,
            'activation': self.activation,
            'K': self.K,
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer)
        })
        return config


class FuncPredictor(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(FuncPredictor, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.output_layer = tf.keras.layers.Dense(2*output_dim)
        self.reshape = tf.keras.layers.Reshape(target_shape=(output_dim, 2))
        self.softmax = tf.keras.layers.Softmax(axis=-1, name='labels')

    def call(self, x):
        x = self.output_layer(x)
        x = self.reshape(x)
        out = self.softmax(x)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
        })
        return config


class SumPooling(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super(SumPooling, self).__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        x_pool = tf.reduce_sum(x, axis=self.axis)
        return x_pool

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'axis': self.axis,
        })
        return config
