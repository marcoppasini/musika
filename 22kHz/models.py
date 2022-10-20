import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Cropping1D,
    Cropping2D,
    Dense,
    Dot,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    Lambda,
    LeakyReLU,
    Multiply,
    ReLU,
    Reshape,
    SeparableConv2D,
    UpSampling2D,
    ZeroPadding2D,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.layer_utils import count_params

from layers import ConvSN2D, DenseSN, PosEnc, AddNoise


class Models_functions:
    def __init__(self, args):

        self.args = args
        if self.args.mixed_precision:
            self.mixed_precision = mixed_precision
            self.policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_global_policy(self.policy)
        self.init = tf.keras.initializers.he_uniform()

    def conv_util(
        self,
        inp,
        filters,
        kernel_size=(1, 3),
        strides=(1, 1),
        noise=False,
        upsample=False,
        padding="same",
        bn=True,
    ):

        x = inp

        if upsample:
            x = tf.keras.layers.Conv2DTranspose(
                filters,
                kernel_size=kernel_size,
                strides=strides,
                activation="linear",
                padding=padding,
                kernel_initializer=self.init,
            )(x)
        else:
            x = tf.keras.layers.Conv2D(
                filters,
                kernel_size=kernel_size,
                strides=strides,
                activation="linear",
                padding=padding,
                kernel_initializer=self.init,
            )(x)

        if noise:
            x = AddNoise(datatype=self.args.datatype)(x)

        if bn:
            x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.activations.swish(x)

        return x

    def adain(self, x, emb):
        emb = tf.keras.layers.Conv2D(
            x.shape[-1],
            kernel_size=(1, 1),
            strides=1,
            activation="linear",
            padding="same",
            kernel_initializer=self.init,
            use_bias=True,
        )(emb)
        x = x / (tf.math.reduce_std(x, -2, keepdims=True) + 1e-7)
        return x * emb

    def se_layer(self, x, filters):
        x = tf.reduce_mean(x, -2, keepdims=True)
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation="linear",
            padding="valid",
            kernel_initializer=self.init,
            use_bias=True,
        )(x)
        x = tf.keras.activations.swish(x)
        return tf.keras.layers.Conv2D(
            filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation="sigmoid",
            padding="valid",
            kernel_initializer=self.init,
            use_bias=True,
        )(x)

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
            )(x)

        if noise:
            x = AddNoise(datatype=self.args.datatype)(x)

        if emb is not None:
            x = self.adain(x, emb)
        else:
            x = tf.keras.layers.BatchNormalization()(x)

        x1 = tf.keras.activations.swish(x)

        if se1 is not None:
            x1 = x1 * se1

        return x1

    def res_block_disc(self, inp, filters, kernel_size=(1, 3), kernel_size_2=None, strides=(1, 1)):

        if kernel_size_2 is None:
            kernel_size_2 = kernel_size

        x = tf.keras.layers.Conv2D(
            inp.shape[-1],
            kernel_size=kernel_size_2,
            strides=1,
            activation="linear",
            padding="same",
            kernel_initializer=self.init,
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
            )(inp)

        return x + inp

    def build_encoder2(self):

        dim = 128

        inpf = Input((1, self.args.shape, dim))

        inpfls = tf.split(inpf, 16, -2)
        inpb = tf.concat(inpfls, 0)

        g0 = self.conv_util(inpb, 256, kernel_size=(1, 1), strides=(1, 1), padding="valid")
        g1 = self.conv_util(g0, 256 + 256, kernel_size=(1, 3), strides=(1, 1), padding="valid")
        g2 = self.conv_util(g1, 512 + 128, kernel_size=(1, 3), strides=(1, 1), padding="valid")
        g3 = self.conv_util(g2, 512 + 128, kernel_size=(1, 1), strides=(1, 1), padding="valid")
        g4 = self.conv_util(g3, 512 + 256, kernel_size=(1, 3), strides=(1, 1), padding="valid")
        g5 = self.conv_util(g4, 512 + 256, kernel_size=(1, 2), strides=(1, 1), padding="valid")

        g = tf.keras.layers.Conv2D(
            64,
            kernel_size=(1, 1),
            strides=1,
            padding="valid",
            kernel_initializer=self.init,
            name="cbottle",
            activation="tanh",
        )(g5)

        gls = tf.split(g, 16, 0)
        g = tf.concat(gls, -2)
        gls = tf.split(g, 2, -2)
        g = tf.concat(gls, 0)

        gf = tf.cast(g, tf.float32)
        return Model(inpf, gf, name="ENC2")

    def build_decoder2(self):

        dim = 128
        bottledim = 64

        inpf = Input((1, self.args.shape // 16, bottledim))

        g = inpf

        g = self.conv_util(
            g,
            512 + 128 + 128,
            kernel_size=(1, 4),
            strides=(1, 1),
            upsample=False,
            noise=True,
        )
        g = self.conv_util(
            g,
            512 + 128 + 128,
            kernel_size=(1, 4),
            strides=(1, 2),
            upsample=True,
            noise=True,
        )
        g = self.conv_util(g, 512 + 128, kernel_size=(1, 4), strides=(1, 2), upsample=True, noise=True)
        g = self.conv_util(g, 512, kernel_size=(1, 4), strides=(1, 1), upsample=False, noise=True)
        g = self.conv_util(g, 256 + 128, kernel_size=(1, 4), strides=(1, 2), upsample=True, noise=True)

        gf = tf.keras.layers.Conv2D(
            dim,
            kernel_size=(1, 1),
            strides=1,
            padding="same",
            activation="tanh",
            kernel_initializer=self.init,
        )(g)

        gfls = tf.split(gf, 2, 0)
        gf = tf.concat(gfls, -2)

        gf = tf.cast(gf, tf.float32)

        return Model(inpf, gf, name="DEC2")

    def build_encoder(self):

        dim = ((4 * self.args.hop) // 2) + 1

        inpf = Input((dim, self.args.shape, 1))

        ginp = tf.transpose(inpf, [0, 3, 2, 1])

        g0 = self.conv_util(
            ginp,
            self.args.hop * 2 + 32,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
        )

        g = self.conv_util(
            g0,
            self.args.hop * 2 + 64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
        )
        g = self.conv_util(
            g,
            self.args.hop * 2 + 64 + 64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
        )
        g = self.conv_util(
            g,
            self.args.hop * 2 + 128 + 64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
        )
        g = self.conv_util(
            g,
            self.args.hop * 2 + 128 + 128,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
        )

        g = tf.keras.layers.Conv2D(
            128,
            kernel_size=(1, 1),
            strides=1,
            padding="valid",
            kernel_initializer=self.init,
        )(g)
        gb = tf.keras.activations.tanh(g)

        gbls = tf.split(gb, 2, -2)
        gb = tf.concat(gbls, 0)

        gb = tf.cast(gb, tf.float32)
        return Model(inpf, gb, name="ENC")

    def build_decoder(self):

        dim = ((4 * self.args.hop) // 2) + 1

        inpf = Input((1, self.args.shape // 2, 128))

        g = inpf

        g0 = self.conv_util(g, self.args.hop * 3, kernel_size=(1, 1), strides=(1, 1), noise=True)

        g1 = self.conv_util(g0, self.args.hop * 2, kernel_size=(1, 3), strides=(1, 2), noise=True)
        g2 = self.conv_util(
            g1,
            self.args.hop + self.args.hop // 2,
            kernel_size=(1, 3),
            strides=(1, 2),
            noise=True,
        )
        g = self.conv_util(
            g2,
            self.args.hop + self.args.hop // 4,
            kernel_size=(1, 3),
            strides=(1, 2),
            noise=True,
        )

        g = self.conv_util(
            g,
            self.args.hop + self.args.hop // 2,
            kernel_size=(1, 4),
            strides=(1, 2),
            upsample=True,
            noise=True,
        )
        g = self.conv_util(
            g + g2,
            self.args.hop * 2,
            kernel_size=(1, 4),
            strides=(1, 2),
            upsample=True,
            noise=True,
        )
        g = self.conv_util(
            g + g1,
            self.args.hop * 3,
            kernel_size=(1, 4),
            strides=(1, 2),
            upsample=True,
            noise=True,
        )

        g = self.conv_util(g + g0, self.args.hop * 5, kernel_size=(1, 1), strides=(1, 1), noise=True)

        g = Conv2D(
            dim * 2,
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_initializer=self.init,
            padding="same",
        )(g)
        g = tf.clip_by_value(g, -1.0, 1.0)

        gf, pf = tf.split(g, 2, -1)

        gfls = tf.split(gf, self.args.shape // self.args.window, 0)
        gf = tf.concat(gfls, -2)

        pfls = tf.split(pf, self.args.shape // self.args.window, 0)
        pf = tf.concat(pfls, -2)

        s = tf.transpose(gf, [0, 2, 3, 1])
        p = tf.transpose(pf, [0, 2, 3, 1])

        s = tf.cast(tf.squeeze(s, -1), tf.float32)
        p = tf.cast(tf.squeeze(p, -1), tf.float32)

        return Model(inpf, [s, p], name="DEC")

    def build_critic(self):

        if self.args.conditional:
            sinp = Input(shape=(1, self.args.latlen, self.args.latdepth * 2 + 1))
            sinpf = sinp[:, :, :, :-1]
            sinpc = sinp[:, :, :, -1:]
        else:
            sinp = Input(shape=(1, self.args.latlen, self.args.latdepth * 2))
            sinpf = sinp

        dim = 64 * 2

        sf = tf.keras.layers.Conv2D(
            self.args.latdepth * 4,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation="linear",
            padding="valid",
            kernel_initializer=self.init,
            use_bias=False,
            trainable=False,
        )(sinpf)

        if self.args.conditional:
            sf = tf.concat([sf, tf.cast(sinpc, self.args.datatype)], -1)

        sf = tf.keras.layers.Conv2D(
            256 + 128,
            kernel_size=(1, 3),
            strides=(1, 2),
            activation="linear",
            padding="same",
            kernel_initializer=self.init,
        )(sf)
        sf = tf.keras.layers.LeakyReLU(0.2)(sf)
        sf = self.res_block_disc(sf, 256 + 128 + 128, kernel_size=(1, 3), strides=(1, 2))
        sf = self.res_block_disc(sf, 512 + 128, kernel_size=(1, 3), strides=(1, 2))
        sf = self.res_block_disc(sf, 512 + 256, kernel_size=(1, 3), strides=(1, 2))
        sf = self.res_block_disc(sf, 512 + 128 + 256, kernel_size=(1, 3), strides=(1, 2))
        sfo = self.res_block_disc(sf, 512 + 512, kernel_size=(1, 3), strides=(1, 2), kernel_size_2=(1, 1))
        sf = sfo

        gf = tf.keras.layers.Dense(1, activation="linear", use_bias=True, kernel_initializer=self.init)(Flatten()(sf))

        gf = tf.cast(gf, tf.float32)
        sfo = tf.cast(sfo, tf.float32)

        return Model(sinp, [gf, sfo], name="C")

    def build_critic_rec(self):

        sinp = Input(shape=(1, self.args.latlen // 64, 512 + 512))

        dim = self.args.latdepth * 2

        sf = tf.keras.layers.Conv2DTranspose(
            512,
            kernel_size=(1, 4),
            strides=(1, 2),
            activation="linear",
            padding="same",
            kernel_initializer=self.init,
        )(sinp)
        sf = tf.keras.layers.LeakyReLU(0.2)(sf)

        sf = tf.keras.layers.Conv2DTranspose(
            256 + 128 + 64,
            kernel_size=(1, 4),
            strides=(1, 2),
            activation="linear",
            padding="same",
            kernel_initializer=self.init,
        )(sf)
        sf = tf.keras.layers.LeakyReLU(0.2)(sf)
        sf = tf.keras.layers.Conv2DTranspose(
            256 + 128,
            kernel_size=(1, 4),
            strides=(1, 2),
            activation="linear",
            padding="same",
            kernel_initializer=self.init,
        )(sf)
        sf = tf.keras.layers.LeakyReLU(0.2)(sf)
        sf = tf.keras.layers.Conv2DTranspose(
            256 + 64,
            kernel_size=(1, 4),
            strides=(1, 2),
            activation="linear",
            padding="same",
            kernel_initializer=self.init,
        )(sf)
        sf = tf.keras.layers.LeakyReLU(0.2)(sf)
        sf = tf.keras.layers.Conv2DTranspose(
            256,
            kernel_size=(1, 4),
            strides=(1, 2),
            activation="linear",
            padding="same",
            kernel_initializer=self.init,
        )(sf)
        sf = tf.keras.layers.LeakyReLU(0.2)(sf)
        sf = tf.keras.layers.Conv2DTranspose(
            128 + 64,
            kernel_size=(1, 4),
            strides=(1, 2),
            activation="linear",
            padding="same",
            kernel_initializer=self.init,
        )(sf)
        sf = tf.keras.layers.LeakyReLU(0.2)(sf)

        gf = tf.keras.layers.Conv2D(
            dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation="tanh",
            padding="same",
            kernel_initializer=self.init,
        )(sf)

        gf = tf.cast(gf, tf.float32)

        return Model(sinp, gf, name="CR")

    def build_generator(self):

        dim = self.args.latdepth * 2

        if self.args.conditional:
            inpf = Input((self.args.latlen, self.args.latdepth * 2))
        else:
            inpf = Input((self.args.latlen, self.args.latdepth * 2 + 1))

        inpfls = tf.split(inpf, 2, -2)
        inpb = tf.concat(inpfls, 0)

        inpg = tf.reduce_mean(inpb, -2)
        inp1 = tf.keras.layers.AveragePooling2D((1, 2), padding="valid")(tf.expand_dims(inpb, -3))
        inp2 = tf.keras.layers.AveragePooling2D((1, 2), padding="valid")(inp1)
        inp3 = tf.keras.layers.AveragePooling2D((1, 2), padding="valid")(inp2)
        inp4 = tf.keras.layers.AveragePooling2D((1, 2), padding="valid")(inp3)
        inp5 = tf.keras.layers.AveragePooling2D((1, 2), padding="valid")(inp4)
        inp6 = tf.keras.layers.AveragePooling2D((1, 2), padding="valid")(inp5)

        g = tf.keras.layers.Dense(
            4 * (512 + 256 + 128),
            activation="linear",
            use_bias=True,
            kernel_initializer=self.init,
        )(Flatten()(inp6))
        g = tf.keras.layers.Reshape((1, 4, 512 + 256 + 128))(g)
        g = AddNoise(datatype=self.args.datatype)(g)
        g = self.adain(g, inp5)
        g = tf.keras.activations.swish(g)

        g = self.conv_util_gen(
            g,
            512 + 256,
            kernel_size=(1, 4),
            strides=(1, 2),
            upsample=True,
            noise=True,
            emb=inp4,
        )
        g1 = self.conv_util_gen(
            g,
            512 + 256,
            kernel_size=(1, 1),
            strides=(1, 1),
            upsample=False,
            noise=True,
            emb=inp4,
        )
        g2 = self.conv_util_gen(
            g1,
            512 + 128,
            kernel_size=(1, 4),
            strides=(1, 2),
            upsample=True,
            noise=True,
            emb=inp3,
        )
        g2b = self.conv_util_gen(
            g2,
            512 + 128,
            kernel_size=(1, 3),
            strides=(1, 1),
            upsample=False,
            noise=True,
            emb=inp3,
        )
        g3 = self.conv_util_gen(
            g2b,
            256 + 256,
            kernel_size=(1, 4),
            strides=(1, 2),
            upsample=True,
            noise=True,
            emb=inp2,
            se1=self.se_layer(g, 256 + 256),
        )
        g3 = self.conv_util_gen(
            g3,
            256 + 256,
            kernel_size=(1, 3),
            strides=(1, 1),
            upsample=False,
            noise=True,
            emb=inp2,
            se1=self.se_layer(g1, 256 + 256),
        )
        g4 = self.conv_util_gen(
            g3,
            256 + 128,
            kernel_size=(1, 4),
            strides=(1, 2),
            upsample=True,
            noise=True,
            emb=inp1,
            se1=self.se_layer(g2, 256 + 128),
        )
        g4 = self.conv_util_gen(
            g4,
            256 + 128,
            kernel_size=(1, 3),
            strides=(1, 1),
            upsample=False,
            noise=True,
            emb=inp1,
            se1=self.se_layer(g2b, 256 + 128),
        )
        g5 = self.conv_util_gen(
            g4,
            256,
            kernel_size=(1, 4),
            strides=(1, 2),
            upsample=True,
            noise=True,
            emb=tf.expand_dims(tf.cast(inpb, dtype=self.args.datatype), -3),
        )

        gf = tf.keras.layers.Conv2D(
            dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_initializer=self.init,
            padding="same",
            activation="tanh",
        )(g5)

        gfls = tf.split(gf, 2, 0)
        gf = tf.concat(gfls, -2)

        gf = tf.cast(gf, tf.float32)

        return Model(inpf, gf, name="GEN")

    # Load past models from path to resume training or test
    def load(self, path, load_dec=False):
        gen = self.build_generator()
        critic = self.build_critic()
        enc = self.build_encoder()
        dec = self.build_decoder()
        enc2 = self.build_encoder2()
        dec2 = self.build_decoder2()
        critic_rec = self.build_critic_rec()
        gen_ema = self.build_generator()

        if self.args.mixed_precision:
            opt_disc = self.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(0.0001, 0.9))
            opt_dec = self.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(0.0001, 0.9))
        else:
            opt_disc = tf.keras.optimizers.Adam(0.0001, 0.9)
            opt_dec = tf.keras.optimizers.Adam(0.0001, 0.9)

        if load_dec:
            dec.load_weights(self.args.dec_path + "/dec.h5")
            dec2.load_weights(self.args.dec_path + "/dec2.h5")
            enc.load_weights(self.args.dec_path + "/enc.h5")
            enc2.load_weights(self.args.dec_path + "/enc2.h5")

        else:
            grad_vars = critic.trainable_weights + critic_rec.trainable_weights
            zero_grads = [tf.zeros_like(w) for w in grad_vars]
            opt_disc.apply_gradients(zip(zero_grads, grad_vars))

            grad_vars = gen.trainable_variables
            zero_grads = [tf.zeros_like(w) for w in grad_vars]
            opt_dec.apply_gradients(zip(zero_grads, grad_vars))

            if not self.args.testing:
                opt_disc.set_weights(np.load(path + "/opt_disc.npy", allow_pickle=True))
                opt_dec.set_weights(np.load(path + "/opt_dec.npy", allow_pickle=True))

            if not self.args.testing:
                critic.load_weights(path + "/critic.h5")
                gen.load_weights(path + "/gen.h5")
                # enc.load_weights(self.args.dec_path + "/enc.h5")
                # enc2.load_weights(self.args.dec_path + "/enc2.h5")
                critic_rec.load_weights(path + "/critic_rec.h5")
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
            critic_rec,
            gen_ema,
            [opt_dec, opt_disc],
        )

    def build(self):
        gen = self.build_generator()
        critic = self.build_critic()
        enc = self.build_encoder()
        dec = self.build_decoder()
        enc2 = self.build_encoder2()
        dec2 = self.build_decoder2()
        critic_rec = self.build_critic_rec()
        gen_ema = self.build_generator()

        gen_ema = tf.keras.models.clone_model(gen)
        gen_ema.set_weights(gen.get_weights())

        if self.args.mixed_precision:
            opt_disc = self.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(0.0001, 0.9))
            opt_dec = self.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(0.0001, 0.9))
        else:
            opt_disc = tf.keras.optimizers.Adam(0.0001, 0.9)
            opt_dec = tf.keras.optimizers.Adam(0.0001, 0.9)

        return (
            critic,
            gen,
            enc,
            dec,
            enc2,
            dec2,
            critic_rec,
            gen_ema,
            [opt_dec, opt_disc],
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
                critic_rec,
                gen_ema,
                [opt_dec, opt_disc],
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
                critic_rec,
                gen_ema,
                [opt_dec, opt_disc],
            ) = self.load(self.args.dec_path, load_dec=True)
            print(f"Encoders/Decoders loaded from {self.args.dec_path}")
            print(f"Networks initialized")

        return (
            critic,
            gen,
            enc,
            dec,
            enc2,
            dec2,
            critic_rec,
            gen_ema,
            [opt_dec, opt_disc],
        )

    def initialize_networks(self):

        (
            critic,
            gen,
            enc,
            dec,
            enc2,
            dec2,
            critic_rec,
            gen_ema,
            [opt_dec, opt_disc],
        ) = self.get_networks()

        print(f"Critic params: {count_params(critic.trainable_variables)}")
        print(f"Generator params: {count_params(gen.trainable_variables)}")

        return (
            critic,
            gen,
            enc,
            dec,
            enc2,
            dec2,
            critic_rec,
            gen_ema,
            [opt_dec, opt_disc],
        )
