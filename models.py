import shutil
import os
from os import path as ospath
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.layer_utils import count_params
from huggingface_hub import hf_hub_download

from layers import AddNoise


class Models_functions:
    def __init__(self, args):

        self.args = args

        if self.args.mixed_precision:
            self.mixed_precision = tf.keras.mixed_precision
            self.policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(self.policy)
        self.init = tf.keras.initializers.he_uniform()

    def conv_util(
        self, inp, filters, kernel_size=(1, 3), strides=(1, 1), noise=False, upsample=False, padding="same", bnorm=True
    ):

        x = inp

        bias = True
        if bnorm:
            bias = False

        if upsample:
            x = tf.keras.layers.Conv2DTranspose(
                filters,
                kernel_size=kernel_size,
                strides=strides,
                activation="linear",
                padding=padding,
                kernel_initializer=self.init,
                use_bias=bias,
            )(x)
        else:
            x = tf.keras.layers.Conv2D(
                filters,
                kernel_size=kernel_size,
                strides=strides,
                activation="linear",
                padding=padding,
                kernel_initializer=self.init,
                use_bias=bias,
            )(x)

        if noise:
            x = AddNoise(self.args.datatype)(x)

        if bnorm:
            x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.activations.swish(x)

        return x

    def pixel_shuffle(self, x, factor=2):
        bs_dim, h_dim, w_dim, c_dim = tf.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]
        x = tf.reshape(x, [bs_dim, h_dim, w_dim, c_dim // factor, factor])
        x = tf.transpose(x, [0, 1, 2, 4, 3])
        return tf.reshape(x, [bs_dim, h_dim, w_dim * factor, c_dim // factor])

    def adain(self, x, emb, name):
        emb = tf.keras.layers.Conv2D(
            x.shape[-1],
            kernel_size=(1, 1),
            strides=1,
            activation="linear",
            padding="same",
            kernel_initializer=self.init,
            use_bias=True,
            name=name,
        )(emb)
        x = x / (tf.math.reduce_std(x, -2, keepdims=True) + 1e-5)
        return x * emb

    def conv_util_gen(
        self,
        inp,
        filters,
        kernel_size=(1, 9),
        strides=(1, 1),
        noise=False,
        upsample=False,
        emb=None,
        se1=None,
        name="0",
    ):

        x = inp

        if upsample:
            x = tf.keras.layers.Conv2DTranspose(
                filters,
                kernel_size=kernel_size,
                strides=strides,
                activation="linear",
                padding="same",
                kernel_initializer=self.init,
                use_bias=True,
                name=name + "c",
            )(x)
        else:
            x = tf.keras.layers.Conv2D(
                filters,
                kernel_size=kernel_size,
                strides=strides,
                activation="linear",
                padding="same",
                kernel_initializer=self.init,
                use_bias=True,
                name=name + "c",
            )(x)

        if noise:
            x = AddNoise(self.args.datatype, name=name + "r")(x)

        if emb is not None:
            x = self.adain(x, emb, name=name + "ai")
        else:
            x = tf.keras.layers.BatchNormalization(name=name + "bn")(x)

        x = tf.keras.activations.swish(x)

        return x

    def res_block_disc(self, inp, filters, kernel_size=(1, 3), kernel_size_2=None, strides=(1, 1), name="0"):

        if kernel_size_2 is None:
            kernel_size_2 = kernel_size

        x = tf.keras.layers.Conv2D(
            inp.shape[-1],
            kernel_size=kernel_size_2,
            strides=1,
            activation="linear",
            padding="same",
            kernel_initializer=self.init,
            name=name + "c0",
        )(inp)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.math.sqrt(tf.cast(0.5, self.args.datatype)) * x
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            activation="linear",
            padding="same",
            kernel_initializer=self.init,
            name=name + "c1",
        )(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.math.sqrt(tf.cast(0.5, self.args.datatype)) * x

        if strides != (1, 1):
            inp = tf.keras.layers.AveragePooling2D(strides, padding="same")(inp)

        if inp.shape[-1] != filters:
            inp = tf.keras.layers.Conv2D(
                filters,
                kernel_size=1,
                strides=1,
                activation="linear",
                padding="same",
                kernel_initializer=self.init,
                use_bias=False,
                name=name + "c3",
            )(inp)

        return x + inp

    def build_encoder2(self):

        inpf = tf.keras.layers.Input((1, self.args.shape, self.args.hop // 4))

        inpfls = tf.split(inpf, 8, -2)
        inpb = tf.concat(inpfls, 0)

        g0 = self.conv_util(inpb, self.args.hop, kernel_size=(1, 3), strides=(1, 1), padding="same", bnorm=False)
        g1 = self.conv_util(
            g0, self.args.hop + self.args.hop // 2, kernel_size=(1, 3), strides=(1, 2), padding="valid", bnorm=False
        )
        g2 = self.conv_util(
            g1, self.args.hop + self.args.hop // 2, kernel_size=(1, 3), strides=(1, 1), padding="same", bnorm=False
        )
        g3 = self.conv_util(g2, self.args.hop * 2, kernel_size=(1, 3), strides=(1, 2), padding="valid", bnorm=False)
        g4 = self.conv_util(g3, self.args.hop * 2, kernel_size=(1, 3), strides=(1, 1), padding="same", bnorm=False)
        g5 = self.conv_util(g4, self.args.hop * 3, kernel_size=(1, 3), strides=(1, 1), padding="valid", bnorm=False)
        g5 = self.conv_util(g5, self.args.hop * 3, kernel_size=(1, 1), strides=(1, 1), padding="valid", bnorm=False)

        g = tf.keras.layers.Conv2D(
            self.args.latdepth,
            kernel_size=(1, 1),
            strides=1,
            padding="valid",
            kernel_initializer=self.init,
            name="cbottle",
            activation="tanh",
        )(g5)

        gls = tf.split(g, 8, 0)
        g = tf.concat(gls, -2)
        gls = tf.split(g, 2, -2)
        g = tf.concat(gls, 0)

        gf = tf.cast(g, tf.float32)

        return tf.keras.Model(inpf, gf, name="ENC2")

    def build_decoder2(self):

        inpf = tf.keras.layers.Input((1, self.args.shape // 32, self.args.latdepth))

        g = inpf

        g = self.conv_util(
            g, self.args.hop * 3, kernel_size=(1, 3), strides=(1, 1), upsample=False, noise=True, bnorm=False
        )
        g = self.conv_util(
            g,
            self.args.hop * 2 + self.args.hop // 2,
            kernel_size=(1, 4),
            strides=(1, 2),
            upsample=True,
            noise=True,
            bnorm=False,
        )
        g = self.conv_util(
            g,
            self.args.hop * 2 + self.args.hop // 2,
            kernel_size=(1, 3),
            strides=(1, 1),
            upsample=False,
            noise=True,
            bnorm=False,
        )
        g = self.conv_util(
            g, self.args.hop * 2, kernel_size=(1, 4), strides=(1, 2), upsample=True, noise=True, bnorm=False
        )
        g = self.conv_util(
            g, self.args.hop * 2, kernel_size=(1, 3), strides=(1, 1), upsample=False, noise=True, bnorm=False
        )
        g = self.conv_util(
            g,
            self.args.hop + self.args.hop // 2,
            kernel_size=(1, 4),
            strides=(1, 2),
            upsample=True,
            noise=True,
            bnorm=False,
        )
        g = self.conv_util(g, self.args.hop, kernel_size=(1, 4), strides=(1, 2), upsample=True, noise=True, bnorm=False)

        gf = tf.keras.layers.Conv2D(
            self.args.hop // 4, kernel_size=(1, 1), strides=1, padding="same", kernel_initializer=self.init, name="cout"
        )(g)

        gfls = tf.split(gf, 2, 0)
        gf = tf.concat(gfls, -2)

        gf = tf.cast(gf, tf.float32)

        return tf.keras.Model(inpf, gf, name="DEC2")

    def build_encoder(self):

        dim = ((4 * self.args.hop) // 2) + 1

        inpf = tf.keras.layers.Input((dim, self.args.shape, 1))

        ginp = tf.transpose(inpf, [0, 3, 2, 1])

        g0 = self.conv_util(ginp, self.args.hop * 4, kernel_size=(1, 1), strides=(1, 1), padding="valid", bnorm=False)
        g1 = self.conv_util(g0, self.args.hop * 4, kernel_size=(1, 1), strides=(1, 1), padding="valid", bnorm=False)
        g2 = self.conv_util(g1, self.args.hop * 4, kernel_size=(1, 1), strides=(1, 1), padding="valid", bnorm=False)
        g4 = self.conv_util(g2, self.args.hop * 4, kernel_size=(1, 1), strides=(1, 1), padding="valid", bnorm=False)
        g5 = self.conv_util(g4, self.args.hop * 4, kernel_size=(1, 1), strides=(1, 1), padding="valid", bnorm=False)

        g = tf.keras.layers.Conv2D(
            self.args.hop // 4, kernel_size=(1, 1), strides=1, padding="valid", kernel_initializer=self.init
        )(g5)

        g = tf.keras.activations.tanh(g)

        gls = tf.split(g, 2, -2)
        g = tf.concat(gls, 0)

        gf = tf.cast(g, tf.float32)

        return tf.keras.Model(inpf, gf, name="ENC")

    def build_decoder(self):

        dim = ((4 * self.args.hop) // 2) + 1

        inpf = tf.keras.layers.Input((1, self.args.shape // 2, self.args.hop // 4))

        g = inpf

        g0 = self.conv_util(g, self.args.hop * 3, kernel_size=(1, 3), strides=(1, 1), noise=True, bnorm=False)
        g1 = self.conv_util(g0, self.args.hop * 3, kernel_size=(1, 3), strides=(1, 2), noise=True, bnorm=False)
        g2 = self.conv_util(g1, self.args.hop * 2, kernel_size=(1, 3), strides=(1, 2), noise=True, bnorm=False)
        g3 = self.conv_util(g2, self.args.hop, kernel_size=(1, 3), strides=(1, 2), noise=True, bnorm=False)
        g = self.conv_util(g3, self.args.hop, kernel_size=(1, 3), strides=(1, 2), noise=True, bnorm=False)

        g33 = self.conv_util(
            g, self.args.hop, kernel_size=(1, 4), strides=(1, 2), upsample=True, noise=True, bnorm=False
        )
        g22 = self.conv_util(
            g3, self.args.hop * 2, kernel_size=(1, 4), strides=(1, 2), upsample=True, noise=True, bnorm=False
        )
        g11 = self.conv_util(
            g22 + g2, self.args.hop * 3, kernel_size=(1, 4), strides=(1, 2), upsample=True, noise=True, bnorm=False
        )
        g00 = self.conv_util(
            g11 + g1, self.args.hop * 3, kernel_size=(1, 4), strides=(1, 2), upsample=True, noise=True, bnorm=False
        )

        g = tf.keras.layers.Conv2D(
            dim, kernel_size=(1, 1), strides=(1, 1), kernel_initializer=self.init, padding="same"
        )(g00 + g0)
        gf = tf.clip_by_value(g, -1.0, 1.0)

        g = self.conv_util(
            g22, self.args.hop * 3, kernel_size=(1, 4), strides=(1, 2), upsample=True, noise=True, bnorm=False
        )
        g = self.conv_util(
            g + g11, self.args.hop * 3, kernel_size=(1, 4), strides=(1, 2), upsample=True, noise=True, bnorm=False
        )
        g = tf.keras.layers.Conv2D(
            dim, kernel_size=(1, 1), strides=(1, 1), kernel_initializer=self.init, padding="same"
        )(g + g00)
        pf = tf.clip_by_value(g, -1.0, 1.0)

        gfls = tf.split(gf, self.args.shape // self.args.window, 0)
        gf = tf.concat(gfls, -2)

        pfls = tf.split(pf, self.args.shape // self.args.window, 0)
        pf = tf.concat(pfls, -2)

        s = tf.transpose(gf, [0, 2, 3, 1])
        p = tf.transpose(pf, [0, 2, 3, 1])

        s = tf.cast(tf.squeeze(s, -1), tf.float32)
        p = tf.cast(tf.squeeze(p, -1), tf.float32)

        return tf.keras.Model(inpf, [s, p], name="DEC")

    def build_critic(self):

        sinp = tf.keras.layers.Input(shape=(1, self.args.latlen, self.args.latdepth * 2))

        sf = tf.keras.layers.Conv2D(
            self.args.base_channels * 3,
            kernel_size=(1, 4),
            strides=(1, 2),
            activation="linear",
            padding="same",
            kernel_initializer=self.init,
            name="1c",
        )(sinp)
        sf = tf.keras.layers.LeakyReLU(0.2)(sf)

        sf = self.res_block_disc(sf, self.args.base_channels * 4, kernel_size=(1, 4), strides=(1, 2), name="2")

        sf = self.res_block_disc(sf, self.args.base_channels * 5, kernel_size=(1, 4), strides=(1, 2), name="3")

        sf = self.res_block_disc(sf, self.args.base_channels * 6, kernel_size=(1, 4), strides=(1, 2), name="4")

        sf = self.res_block_disc(sf, self.args.base_channels * 7, kernel_size=(1, 4), strides=(1, 2), name="5")

        if not self.args.small:
            sf = self.res_block_disc(
                sf, self.args.base_channels * 7, kernel_size=(1, 4), strides=(1, 2), kernel_size_2=(1, 1), name="6"
            )

        sf = tf.keras.layers.Conv2D(
            self.args.base_channels * 7,
            kernel_size=(1, 3),
            strides=(1, 1),
            activation="linear",
            padding="same",
            kernel_initializer=self.init,
            name="7c",
        )(sf)
        sf = tf.keras.layers.LeakyReLU(0.2)(sf)

        gf = tf.keras.layers.Dense(1, activation="linear", use_bias=True, kernel_initializer=self.init, name="7d")(
            tf.keras.layers.Flatten()(sf)
        )

        gf = tf.cast(gf, tf.float32)

        return tf.keras.Model(sinp, gf, name="C")

    def build_generator(self):

        dim = self.args.latdepth * 2

        inpf = tf.keras.layers.Input((self.args.latlen, self.args.latdepth * 2))

        inpfls = tf.split(inpf, 2, -2)
        inpb = tf.concat(inpfls, 0)

        inpg = tf.reduce_mean(inpb, -2)
        inp1 = tf.keras.layers.AveragePooling2D((1, 2), padding="valid")(tf.expand_dims(inpb, -3))
        inp2 = tf.keras.layers.AveragePooling2D((1, 2), padding="valid")(inp1)
        inp3 = tf.keras.layers.AveragePooling2D((1, 2), padding="valid")(inp2)
        inp4 = tf.keras.layers.AveragePooling2D((1, 2), padding="valid")(inp3)
        inp5 = tf.keras.layers.AveragePooling2D((1, 2), padding="valid")(inp4)
        if not self.args.small:
            inp6 = tf.keras.layers.AveragePooling2D((1, 2), padding="valid")(inp5)

        if not self.args.small:
            g = tf.keras.layers.Dense(
                4 * (self.args.base_channels * 7),
                activation="linear",
                use_bias=True,
                kernel_initializer=self.init,
                name="00d",
            )(tf.keras.layers.Flatten()(inp6))
            g = tf.keras.layers.Reshape((1, 4, self.args.base_channels * 7))(g)
            g = AddNoise(self.args.datatype, name="00n")(g)
            g = self.adain(g, inp5, name="00ai")
            g = tf.keras.activations.swish(g)
        else:
            g = tf.keras.layers.Dense(
                4 * (self.args.base_channels * 7),
                activation="linear",
                use_bias=True,
                kernel_initializer=self.init,
                name="00d",
            )(tf.keras.layers.Flatten()(inp5))
            g = tf.keras.layers.Reshape((1, 4, self.args.base_channels * 7))(g)
            g = AddNoise(self.args.datatype, name="00n")(g)
            g = self.adain(g, inp4, name="00ai")
            g = tf.keras.activations.swish(g)

        if not self.args.small:
            g1 = self.conv_util_gen(
                g,
                self.args.base_channels * 6,
                kernel_size=(1, 4),
                strides=(1, 2),
                upsample=True,
                noise=True,
                emb=inp4,
                name="0",
            )
            g1 = tf.math.sqrt(tf.cast(0.5, self.args.datatype)) * g1
            g1 = self.conv_util_gen(
                g1,
                self.args.base_channels * 6,
                kernel_size=(1, 4),
                strides=(1, 1),
                upsample=False,
                noise=True,
                emb=inp4,
                name="1",
            )
            g1 = tf.math.sqrt(tf.cast(0.5, self.args.datatype)) * g1
            g1 = g1 + tf.keras.layers.Conv2D(
                g1.shape[-1],
                kernel_size=(1, 1),
                strides=1,
                activation="linear",
                padding="same",
                kernel_initializer=self.init,
                use_bias=True,
                name="res1c",
            )(self.pixel_shuffle(g))
        else:
            g1 = self.conv_util_gen(
                g,
                self.args.base_channels * 6,
                kernel_size=(1, 1),
                strides=(1, 1),
                upsample=False,
                noise=True,
                emb=inp4,
                name="0_small",
            )
            g1 = tf.math.sqrt(tf.cast(0.5, self.args.datatype)) * g1
            g1 = self.conv_util_gen(
                g1,
                self.args.base_channels * 6,
                kernel_size=(1, 1),
                strides=(1, 1),
                upsample=False,
                noise=True,
                emb=inp4,
                name="1_small",
            )
            g1 = tf.math.sqrt(tf.cast(0.5, self.args.datatype)) * g1
            g1 = g1 + tf.keras.layers.Conv2D(
                g1.shape[-1],
                kernel_size=(1, 1),
                strides=1,
                activation="linear",
                padding="same",
                kernel_initializer=self.init,
                use_bias=True,
                name="res1c_small",
            )(g)

        g2 = self.conv_util_gen(
            g1,
            self.args.base_channels * 5,
            kernel_size=(1, 4),
            strides=(1, 2),
            upsample=True,
            noise=True,
            emb=inp3,
            name="2",
        )
        g2 = tf.math.sqrt(tf.cast(0.5, self.args.datatype)) * g2
        g2 = self.conv_util_gen(
            g2,
            self.args.base_channels * 5,
            kernel_size=(1, 4),
            strides=(1, 1),
            upsample=False,
            noise=True,
            emb=inp3,
            name="3",
        )
        g2 = tf.math.sqrt(tf.cast(0.5, self.args.datatype)) * g2
        g2 = g2 + tf.keras.layers.Conv2D(
            g2.shape[-1],
            kernel_size=(1, 1),
            strides=1,
            activation="linear",
            padding="same",
            kernel_initializer=self.init,
            use_bias=True,
            name="res2c",
        )(self.pixel_shuffle(g1))

        g3 = self.conv_util_gen(
            g2,
            self.args.base_channels * 4,
            kernel_size=(1, 4),
            strides=(1, 2),
            upsample=True,
            noise=True,
            emb=inp2,
            name="4",
        )
        g3 = tf.math.sqrt(tf.cast(0.5, self.args.datatype)) * g3
        g3 = self.conv_util_gen(
            g3,
            self.args.base_channels * 4,
            kernel_size=(1, 4),
            strides=(1, 1),
            upsample=False,
            noise=True,
            emb=inp2,
            name="5",
        )
        g3 = tf.math.sqrt(tf.cast(0.5, self.args.datatype)) * g3
        g3 = g3 + tf.keras.layers.Conv2D(
            g3.shape[-1],
            kernel_size=(1, 1),
            strides=1,
            activation="linear",
            padding="same",
            kernel_initializer=self.init,
            use_bias=True,
            name="res3c",
        )(self.pixel_shuffle(g2))

        g4 = self.conv_util_gen(
            g3,
            self.args.base_channels * 3,
            kernel_size=(1, 4),
            strides=(1, 2),
            upsample=True,
            noise=True,
            emb=inp1,
            name="6",
        )
        g4 = tf.math.sqrt(tf.cast(0.5, self.args.datatype)) * g4
        g4 = self.conv_util_gen(
            g4,
            self.args.base_channels * 3,
            kernel_size=(1, 4),
            strides=(1, 1),
            upsample=False,
            noise=True,
            emb=inp1,
            name="7",
        )
        g4 = tf.math.sqrt(tf.cast(0.5, self.args.datatype)) * g4
        g4 = g4 + tf.keras.layers.Conv2D(
            g4.shape[-1],
            kernel_size=(1, 1),
            strides=1,
            activation="linear",
            padding="same",
            kernel_initializer=self.init,
            use_bias=True,
            name="res4c",
        )(self.pixel_shuffle(g3))

        g5 = self.conv_util_gen(
            g4,
            self.args.base_channels * 2,
            kernel_size=(1, 4),
            strides=(1, 2),
            upsample=True,
            noise=True,
            emb=tf.expand_dims(tf.cast(inpb, dtype=self.args.datatype), -3),
            name="8",
        )

        gf = tf.keras.layers.Conv2D(
            dim, kernel_size=(1, 4), strides=(1, 1), kernel_initializer=self.init, padding="same", name="9c"
        )(g5)

        gfls = tf.split(gf, 2, 0)
        gf = tf.concat(gfls, -2)

        gf = tf.cast(gf, tf.float32)

        return tf.keras.Model(inpf, gf, name="GEN")

    # Load past models from path to resume training or test
    def load(self, path, load_dec=False):
        gen = self.build_generator()
        critic = self.build_critic()
        enc = self.build_encoder()
        dec = self.build_decoder()
        enc2 = self.build_encoder2()
        dec2 = self.build_decoder2()
        gen_ema = self.build_generator()

        switch = tf.Variable(-1.0, dtype=tf.float32)

        if self.args.mixed_precision:
            opt_disc = self.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(0.0001, 0.5))
            opt_dec = self.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(0.0001, 0.5))
        else:
            opt_disc = tf.keras.optimizers.Adam(0.0001, 0.9)
            opt_dec = tf.keras.optimizers.Adam(0.0001, 0.9)

        if load_dec:
            dec.load_weights(self.args.dec_path + "/dec.h5")
            dec2.load_weights(self.args.dec_path + "/dec2.h5")
            enc.load_weights(self.args.dec_path + "/enc.h5")
            enc2.load_weights(self.args.dec_path + "/enc2.h5")

        else:
            grad_vars = critic.trainable_weights
            zero_grads = [tf.zeros_like(w) for w in grad_vars]
            opt_disc.apply_gradients(zip(zero_grads, grad_vars))

            grad_vars = gen.trainable_variables
            zero_grads = [tf.zeros_like(w) for w in grad_vars]
            opt_dec.apply_gradients(zip(zero_grads, grad_vars))

            if not self.args.testing:
                opt_disc.set_weights(np.load(path + "/opt_disc.npy", allow_pickle=True))
                opt_dec.set_weights(np.load(path + "/opt_dec.npy", allow_pickle=True))
                critic.load_weights(path + "/critic.h5")
                gen.load_weights(path + "/gen.h5")
                switch = tf.Variable(float(np.load(path + "/switch.npy", allow_pickle=True)), dtype=tf.float32)
                # enc.load_weights(self.args.dec_path + "/enc.h5")
                # enc2.load_weights(self.args.dec_path + "/enc2.h5")
            gen_ema.load_weights(path + "/gen_ema.h5")
            dec.load_weights(self.args.dec_path + "/dec.h5")
            dec2.load_weights(self.args.dec_path + "/dec2.h5")
            enc.load_weights(self.args.dec_path + "/enc.h5")
            enc2.load_weights(self.args.dec_path + "/enc2.h5")

        return (
            critic,
            gen,
            enc,
            dec,
            enc2,
            dec2,
            gen_ema,
            [opt_dec, opt_disc],
            switch,
        )

    def build(self):
        gen = self.build_generator()
        critic = self.build_critic()
        enc = self.build_encoder()
        dec = self.build_decoder()
        enc2 = self.build_encoder2()
        dec2 = self.build_decoder2()
        gen_ema = self.build_generator()

        switch = tf.Variable(-1.0, dtype=tf.float32)

        gen_ema = tf.keras.models.clone_model(gen)
        gen_ema.set_weights(gen.get_weights())

        if self.args.mixed_precision:
            opt_disc = self.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(0.0001, 0.5))
            opt_dec = self.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(0.0001, 0.5))
        else:
            opt_disc = tf.keras.optimizers.Adam(0.0001, 0.5)
            opt_dec = tf.keras.optimizers.Adam(0.0001, 0.5)

        return (
            critic,
            gen,
            enc,
            dec,
            enc2,
            dec2,
            gen_ema,
            [opt_dec, opt_disc],
            switch,
        )

    def get_networks(self):
        if self.args.load_path != "None":
            (
                critic,
                gen,
                enc,
                dec,
                enc2,
                dec2,
                gen_ema,
                [opt_dec, opt_disc],
                switch,
            ) = self.load(self.args.load_path, load_dec=False)
            print(f"Networks loaded from {self.args.load_path}")
        else:
            (
                critic,
                gen,
                enc,
                dec,
                enc2,
                dec2,
                gen_ema,
                [opt_dec, opt_disc],
                switch,
            ) = self.load(self.args.dec_path, load_dec=True)
            print(f"Encoders/Decoders loaded from {self.args.dec_path}")
            print(f"Networks initialized")

        return (critic, gen, enc, dec, enc2, dec2, gen_ema, [opt_dec, opt_disc], switch)

    def initialize_networks(self):

        (critic, gen, enc, dec, enc2, dec2, gen_ema, [opt_dec, opt_disc], switch) = self.get_networks()

        print(f"Critic params: {count_params(critic.trainable_variables)}")
        print(f"Generator params: {count_params(gen.trainable_variables)}")

        return (critic, gen, enc, dec, enc2, dec2, gen_ema, [opt_dec, opt_disc], switch)

    def download_networks(self):

        print("Checking if models are already available...")

        model_names = ["enc.h5", "enc2.h5", "dec.h5", "dec2.h5"]
        for n in model_names:
            if not ospath.exists(self.args.base_path + "/ae/" + n):
                cached_path = hf_hub_download(repo_id="marcop/musika_ae", filename=n)
                os.makedirs(self.args.base_path + "/ae", exist_ok=True)
                shutil.copy(cached_path, self.args.base_path + "/ae")

        model_names = ["critic.h5", "gen.h5", "gen_ema.h5", "opt_dec.npy", "opt_disc.npy", "switch.npy"]
        for n in model_names:
            if not ospath.exists(self.args.base_path + "/techno/" + n):
                cached_path = hf_hub_download(repo_id="marcop/musika_techno", filename=n)
                os.makedirs(self.args.base_path + "/techno", exist_ok=True)
                shutil.copy(cached_path, self.args.base_path + "/techno")
        for n in model_names:
            if not ospath.exists(self.args.base_path + "/misc/" + n):
                cached_path = hf_hub_download(repo_id="marcop/musika_misc", filename=n)
                os.makedirs(self.args.base_path + "/misc", exist_ok=True)
                shutil.copy(cached_path, self.args.base_path + "/misc")
        for n in model_names:
            if not ospath.exists(self.args.base_path + "/misc_small/" + n):
                cached_path = hf_hub_download(repo_id="marcop/musika_misc_small", filename=n)
                os.makedirs(self.args.base_path + "/misc_small", exist_ok=True)
                shutil.copy(cached_path, self.args.base_path + "/misc_small")

        print("Models are available!")
