from tensorflow.keras import activations
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


class GraphCNN(Layer):

    def __init__(self,
                 output_dim,
                 activation=None,
                 use_bias=True):
        super(GraphCNN, self).__init__()

        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        kernel_shape = (input_dim, self.output_dim)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer='glorot_uniform',
                                      name='kernel',
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer='glorot_uniform',
                                        name='bias')
        else:
            self.bias = None

    def call(self, inputs):
        output = K.batch_dot(inputs[1], inputs[0])
        output = K.dot(output, self.kernel)

        if self.use_bias:
            output = K.bias_add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)
        return output
