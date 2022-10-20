import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense
from tensorflow.python.eager import context
from tensorflow.python.ops import (
    array_ops,
    gen_math_ops,
    math_ops,
    sparse_ops,
    standard_ops,
)


def l2normalize(v, eps=1e-12):
    return v / (tf.norm(v) + eps)


class ConvSN2D(tf.keras.layers.Conv2D):
    def __init__(self, filters, kernel_size, power_iterations=1, datatype=tf.float32, **kwargs):
        super(ConvSN2D, self).__init__(filters, kernel_size, **kwargs)
        self.power_iterations = power_iterations
        self.datatype = datatype

    def build(self, input_shape):
        super(ConvSN2D, self).build(input_shape)

        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1

        self.u = self.add_weight(
            self.name + "_u",
            shape=tuple([1, self.kernel.shape.as_list()[-1]]),
            initializer=tf.initializers.RandomNormal(0, 1),
            trainable=False,
            dtype=self.dtype,
        )

    def compute_spectral_norm(self, W, new_u, W_shape):
        for _ in range(self.power_iterations):

            new_v = l2normalize(tf.matmul(new_u, tf.transpose(W)))
            new_u = l2normalize(tf.matmul(new_v, W))

        sigma = tf.matmul(tf.matmul(new_v, W), tf.transpose(new_u))
        W_bar = W / sigma

        with tf.control_dependencies([self.u.assign(new_u)]):
            W_bar = tf.reshape(W_bar, W_shape)

        return W_bar

    def call(self, inputs):
        W_shape = self.kernel.shape.as_list()
        W_reshaped = tf.reshape(self.kernel, (-1, W_shape[-1]))
        new_kernel = self.compute_spectral_norm(W_reshaped, self.u, W_shape)
        outputs = self._convolution_op(inputs, new_kernel)

        if self.use_bias:
            if self.data_format == "channels_first":
                outputs = tf.nn.bias_add(outputs, self.bias, data_format="NCHW")
            else:
                outputs = tf.nn.bias_add(outputs, self.bias, data_format="NHWC")
        if self.activation is not None:
            return self.activation(outputs)

        return outputs


class DenseSN(Dense):
    def __init__(self, datatype=tf.float32, **kwargs):
        super(DenseSN, self).__init__(**kwargs)
        self.datatype = datatype

    def build(self, input_shape):
        super(DenseSN, self).build(input_shape)

        self.u = self.add_weight(
            self.name + "_u",
            shape=tuple([1, self.kernel.shape.as_list()[-1]]),
            initializer=tf.initializers.RandomNormal(0, 1),
            trainable=False,
            dtype=self.datatype,
        )

    def compute_spectral_norm(self, W, new_u, W_shape):
        new_v = l2normalize(tf.matmul(new_u, tf.transpose(W)))
        new_u = l2normalize(tf.matmul(new_v, W))
        sigma = tf.matmul(tf.matmul(new_v, W), tf.transpose(new_u))
        W_bar = W / sigma
        with tf.control_dependencies([self.u.assign(new_u)]):
            W_bar = tf.reshape(W_bar, W_shape)
        return W_bar

    def call(self, inputs):
        W_shape = self.kernel.shape.as_list()
        W_reshaped = tf.reshape(self.kernel, (-1, W_shape[-1]))
        new_kernel = self.compute_spectral_norm(W_reshaped, self.u, W_shape)
        rank = len(inputs.shape)
        if rank > 2:
            outputs = standard_ops.tensordot(inputs, new_kernel, [[rank - 1], [0]])
            if not context.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            inputs = math_ops.cast(inputs, self._compute_dtype)
            if K.is_sparse(inputs):
                outputs = sparse_ops.sparse_tensor_dense_matmul(inputs, new_kernel)
            else:
                outputs = gen_math_ops.mat_mul(inputs, new_kernel)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class AddNoise(tf.keras.layers.Layer):
    def __init__(self, datatype=tf.float32, **kwargs):
        super(AddNoise, self).__init__(**kwargs)
        self.datatype = datatype

    def build(self, input_shape):
        self.b = self.add_weight(
            shape=[
                1,
            ],
            initializer=tf.keras.initializers.zeros(),
            trainable=True,
            name="noise_weight",
        )

    def call(self, inputs):
        rand = tf.random.normal(
            [tf.shape(inputs)[0], inputs.shape[1], inputs.shape[2], 1],
            mean=0.0,
            stddev=1.0,
            dtype=self.datatype,
        )
        output = inputs + self.b * rand
        return output


class PosEnc(tf.keras.layers.Layer):
    def __init__(self, datatype=tf.float32, **kwargs):
        super(PosEnc, self).__init__(**kwargs)
        self.datatype = datatype

    def call(self, inputs):
        pos = tf.repeat(
            tf.reshape(tf.range(inputs.shape[-3], dtype=tf.int32), [1, -1, 1, 1]),
            inputs.shape[-2],
            -2,
        )
        pos = tf.cast(tf.repeat(pos, tf.shape(inputs)[0], 0), self.dtype) / tf.cast(inputs.shape[-3], self.datatype)
        return tf.concat([inputs, pos], -1)  # [bs,1,hop,2]


def flatten_hw(x, data_format="channels_last"):
    if data_format == "channels_last":
        x = tf.transpose(x, perm=[0, 3, 1, 2])  # Convert to `channels_first`

    old_shape = tf.shape(x)
