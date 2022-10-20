import tensorflow as tf


def mae(x, y):
    return tf.reduce_mean(tf.abs(x - y))


def mse(x, y):
    return tf.reduce_mean((x - y) ** 2)


def d_loss_f(fake):
    return tf.reduce_mean(tf.maximum(1 + fake, 0))


def d_loss_r(real):
    return tf.reduce_mean(tf.maximum(1 - real, 0))


def g_loss_f(fake):
    return tf.reduce_mean(-fake)


def g_loss_r(real):
    return tf.reduce_mean(real)


def spec_conv(real, fake):
    diff = tf.math.sqrt(tf.math.reduce_sum((real - fake) ** 2, [-2, -1]))
    den = tf.math.sqrt(tf.math.reduce_sum(real ** 2, [-2, -1]))
    return tf.reduce_mean(diff / den)


def log_norm(real, fake):
    return tf.reduce_mean(tf.math.log(tf.math.reduce_sum(tf.abs(real - fake), [-2, -1])))


def msesum(x, y):
    return tf.reduce_mean(tf.math.reduce_sum((x - y) ** 2, -1, keepdims=True) + 1e-7)
