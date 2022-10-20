import os
import time

import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import random_seed
import gradio as gr
from scipy.io.wavfile import write as write_wav


class Utils_functions:
    def __init__(self, args):

        self.args = args

        melmat = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=args.mel_bins,
            num_spectrogram_bins=(4 * args.hop) // 2 + 1,
            sample_rate=args.sr,
            lower_edge_hertz=0.0,
            upper_edge_hertz=args.sr // 2,
        )
        mel_f = tf.convert_to_tensor(librosa.mel_frequencies(n_mels=args.mel_bins + 2, fmin=0.0, fmax=args.sr // 2))
        enorm = tf.cast(
            tf.expand_dims(
                tf.constant(2.0 / (mel_f[2 : args.mel_bins + 2] - mel_f[: args.mel_bins])),
                0,
            ),
            tf.float32,
        )
        melmat = tf.multiply(melmat, enorm)
        melmat = tf.divide(melmat, tf.reduce_sum(melmat, axis=0))
        self.melmat = tf.where(tf.math.is_nan(melmat), tf.zeros_like(melmat), melmat)

        with np.errstate(divide="ignore", invalid="ignore"):
            self.melmatinv = tf.constant(np.nan_to_num(np.divide(melmat.numpy().T, np.sum(melmat.numpy(), axis=1))).T)

    def conc_tog_specphase(self, S, P):
        S = tf.cast(S, tf.float32)
        P = tf.cast(P, tf.float32)
        S = self.denormalize(S, clip=False)
        S = tf.math.sqrt(self.db2power(S) + 1e-7)
        P = P * np.pi
        Sls = tf.split(S, S.shape[0], 0)
        S = tf.squeeze(tf.concat(Sls, 1), 0)
        Pls = tf.split(P, P.shape[0], 0)
        P = tf.squeeze(tf.concat(Pls, 1), 0)
        SP = tf.cast(S, tf.complex64) * tf.math.exp(1j * tf.cast(P, tf.complex64))
        wv = tf.signal.inverse_stft(
            SP,
            4 * self.args.hop,
            self.args.hop,
            fft_length=4 * self.args.hop,
            window_fn=tf.signal.inverse_stft_window_fn(self.args.hop),
        )
        return np.squeeze(wv)

    def _tf_log10(self, x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    def normalize(self, S, clip=False):
        S = (S - self.args.mu_rescale) / self.args.sigma_rescale
        if clip:
            S = tf.clip_by_value(S, -1.0, 1.0)
        return S

    def normalize_rel(self, S):
        S = S - tf.math.reduce_min(S + 1e-7)
        S = (S / (tf.math.reduce_max(S + 1e-7) + 1e-7)) + 1e-7
        return S

    def denormalize(self, S, clip=False):
        if clip:
            S = tf.clip_by_value(S, -1.0, 1.0)
        return (S * self.args.sigma_rescale) + self.args.mu_rescale

    def amp2db(self, x):
        return 20 * self._tf_log10(tf.clip_by_value(tf.abs(x), 1e-5, 1e100))

    def db2amp(self, x):
        return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)

    def power2db(self, power, ref_value=1.0, amin=1e-10, top_db=None, norm=False):
        log_spec = 10.0 * self._tf_log10(tf.maximum(amin, power))
        log_spec -= 10.0 * self._tf_log10(tf.maximum(amin, ref_value))
        if top_db is not None:
            log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)
        return log_spec

    def power2db_batch(self, power, ref_value=1.0, amin=1e-10, top_db=None, norm=False):
        log_spec = 10.0 * self._tf_log10(tf.maximum(amin, power))
        log_spec -= 10.0 * self._tf_log10(tf.maximum(amin, ref_value))
        if top_db is not None:
            log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec, [-2, -1], keepdims=True) - top_db)
        return log_spec

    def db2power(self, S_db, ref=1.0):
        return ref * tf.math.pow(10.0, 0.1 * S_db)

    def wv2mel(self, wv, topdb=80.0):
        X = tf.signal.stft(
            wv,
            frame_length=4 * self.args.hop,
            frame_step=self.args.hop,
            fft_length=4 * self.args.hop,
            window_fn=tf.signal.hann_window,
            pad_end=False,
        )
        S = self.normalize(self.power2db(tf.abs(X) ** 2, top_db=topdb) - self.args.ref_level_db)
        SM = tf.tensordot(S, self.melmat, 1)
        return SM

    def mel2spec(self, SM):
        return tf.tensordot(SM, tf.transpose(self.melmatinv), 1)

    def spec2mel(self, S):
        return tf.tensordot(S, self.melmat, 1)

    def wv2spec(self, wv, hop_size=256, fac=4):
        X = tf.signal.stft(
            wv,
            frame_length=fac * hop_size,
            frame_step=hop_size,
            fft_length=fac * hop_size,
            window_fn=tf.signal.hann_window,
            pad_end=False,
        )
        return self.normalize(self.power2db(tf.abs(X) ** 2, top_db=None))

    def wv2spec_hop(self, wv, topdb=80.0, hopsize=256):
        X = tf.signal.stft(
            wv,
            frame_length=4 * hopsize,
            frame_step=hopsize,
            fft_length=4 * hopsize,
            window_fn=tf.signal.hann_window,
            pad_end=False,
        )
        S = self.normalize(self.power2db(tf.abs(X) ** 2, top_db=topdb))
        return tf.tensordot(S, self.melmat, 1)

    def distribute(self, x, model, bs=64, dual_out=False):
        outls = []
        if isinstance(x, list):
            bdim = x[0].shape[0]
            for i in range(((bdim - 2) // bs) + 1):
                outls.append(model([el[i * bs : i * bs + bs] for el in x], training=False))
        else:
            bdim = x.shape[0]
            for i in range(((bdim - 2) // bs) + 1):
                outls.append(model(x[i * bs : i * bs + bs], training=False))

        if dual_out:
            return np.concatenate([outls[k][0] for k in range(len(outls))], 0), np.concatenate(
                [outls[k][1] for k in range(len(outls))], 0
            )
        else:
            return np.concatenate(outls, 0)

    def distribute_enc(self, x, model, bs=64):
        outls = []
        if isinstance(x, list):
            bdim = x[0].shape[0]
            for i in range(((bdim - 2) // bs) + 1):
                res = model([el[i * bs : i * bs + bs] for el in x], training=False)
                resls = tf.split(res, self.args.shape // self.args.window, 0)
                res = tf.concat(resls, -2)
                outls.append(res)
        else:
            bdim = x.shape[0]
            for i in range(((bdim - 2) // bs) + 1):
                res = model(x[i * bs : i * bs + bs], training=False)
                resls = tf.split(res, self.args.shape // self.args.window, 0)
                res = tf.concat(resls, -2)
                outls.append(res)

        return np.concatenate(outls, 0)

    def distribute_dec(self, x, model, bs=64):
        outls = []
        bdim = x.shape[0]
        for i in range(((bdim - 2) // bs) + 1):
            inp = x[i * bs : i * bs + bs]
            inpls = tf.split(inp, 2, -2)
            inp = tf.concat(inpls, 0)
            res = model(inp, training=False)
            outls.append(res)
        return np.concatenate([outls[k][0] for k in range(len(outls))], 0), np.concatenate(
            [outls[k][1] for k in range(len(outls))], 0
        )

    def distribute_dec2(self, x, model, bs=64):
        outls = []
        bdim = x.shape[0]
        for i in range(((bdim - 2) // bs) + 1):
            inp1 = x[i * bs : i * bs + bs]
            inpls = tf.split(inp1, 2, -2)
            inp1 = tf.concat(inpls, 0)
            outls.append(model(inp1, training=False))

        return np.concatenate(outls, 0)

    def get_noise_interp(self):
        noiseg = tf.random.normal([1, 64], dtype=tf.float32)

        noisel = tf.concat([tf.random.normal([1, 64], dtype=tf.float32), noiseg], -1)
        noisec = tf.concat([tf.random.normal([1, 64], dtype=tf.float32), noiseg], -1)
        noiser = tf.concat([tf.random.normal([1, 64], dtype=tf.float32), noiseg], -1)

        rl = tf.linspace(noisel, noisec, self.args.latlen + 1, axis=-2)[:, :-1, :]
        rr = tf.linspace(noisec, noiser, self.args.latlen + 1, axis=-2)

        noisetot = tf.concat([rl, rr], -2)
        return tf.image.random_crop(noisetot, [1, self.args.latlen, 64 + 64])

    def generate_example_stereo(self, models_ls):
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
        ) = models_ls
        abb = gen_ema(self.get_noise_interp(), training=False)
        abbls = tf.split(abb, abb.shape[-2] // 16, -2)
        abb = tf.concat(abbls, 0)

        chls = []
        for channel in range(2):

            ab = self.distribute_dec2(
                abb[
                    :,
                    :,
                    :,
                    channel * self.args.latdepth : channel * self.args.latdepth + self.args.latdepth,
                ],
                dec2,
            )
            abls = tf.split(ab, ab.shape[-2] // self.args.shape, -2)
            ab = tf.concat(abls, 0)
            ab_m, ab_p = self.distribute_dec(ab, dec)
            wv = self.conc_tog_specphase(ab_m, ab_p)
            chls.append(wv)

        return np.stack(chls, -1)

    # Save in training loop
    def save_test_image_full(self, path, models_ls=None):

        abwv = self.generate_example_stereo(models_ls)
        abwv2 = self.generate_example_stereo(models_ls)
        abwv3 = self.generate_example_stereo(models_ls)
        abwv4 = self.generate_example_stereo(models_ls)

        # IPython.display.display(
        #     IPython.display.Audio(np.squeeze(np.transpose(abwv)), rate=self.args.sr)
        # )
        # IPython.display.display(
        #     IPython.display.Audio(np.squeeze(np.transpose(abwv2)), rate=self.args.sr)
        # )
        # IPython.display.display(
        #     IPython.display.Audio(np.squeeze(np.transpose(abwv3)), rate=self.args.sr)
        # )
        # IPython.display.display(
        #     IPython.display.Audio(np.squeeze(np.transpose(abwv4)), rate=self.args.sr)
        # )

        write_wav(f"{path}/out1.wav", self.args.sr, np.squeeze(abwv))
        write_wav(f"{path}/out2.wav", self.args.sr, np.squeeze(abwv2))
        write_wav(f"{path}/out3.wav", self.args.sr, np.squeeze(abwv3))
        write_wav(f"{path}/out4.wav", self.args.sr, np.squeeze(abwv4))

        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(20, 20))
        axs[0].imshow(
            np.flip(
                np.array(
                    tf.transpose(
                        self.wv2spec_hop((abwv[:, 0] + abwv[:, 1]) / 2.0, 80.0, 256),
                        [1, 0],
                    )
                ),
                -2,
            ),
            cmap=None,
        )
        axs[0].axis("off")
        axs[0].set_title("Generated1")
        axs[1].imshow(
            np.flip(
                np.array(
                    tf.transpose(
                        self.wv2spec_hop((abwv2[:, 0] + abwv2[:, 1]) / 2.0, 80.0, 256),
                        [1, 0],
                    )
                ),
                -2,
            ),
            cmap=None,
        )
        axs[1].axis("off")
        axs[1].set_title("Generated2")
        axs[2].imshow(
            np.flip(
                np.array(
                    tf.transpose(
                        self.wv2spec_hop((abwv3[:, 0] + abwv3[:, 1]) / 2.0, 80.0, 256),
                        [1, 0],
                    )
                ),
                -2,
            ),
            cmap=None,
        )
        axs[2].axis("off")
        axs[2].set_title("Generated3")
        axs[3].imshow(
            np.flip(
                np.array(
                    tf.transpose(
                        self.wv2spec_hop((abwv4[:, 0] + abwv4[:, 1]) / 2.0, 80.0, 256),
                        [1, 0],
                    )
                ),
                -2,
            ),
            cmap=None,
        )
        axs[3].axis("off")
        axs[3].set_title("Generated4")
        # plt.show()
        plt.savefig(f"{path}/output.png")

    def save_end(
        self,
        epoch,
        gloss,
        closs,
        mloss,
        models_ls=None,
        n_save=3,
        save_path="checkpoints",
    ):
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
        ) = models_ls
        if epoch % n_save == 0:
            print("Saving...")
            path = f"{save_path}/MUSIKA_iterations-{((epoch+1)*self.args.totsamples)//(self.args.bs*1000)}k_losses-{str(gloss)[:9]}-{str(closs)[:9]}-{str(mloss)[:9]}"
            os.mkdir(path)
            critic.save_weights(path + "/critic.h5")
            critic_rec.save_weights(path + "/critic_rec.h5")
            gen.save_weights(path + "/gen.h5")
            gen_ema.save_weights(path + "/gen_ema.h5")
            # enc.save_weights(path + "/enc.h5")
            # dec.save_weights(path + "/dec.h5")
            # enc2.save_weights(path + "/enc2.h5")
            # dec2.save_weights(path + "/dec2.h5")
            np.save(path + "/opt_dec.npy", opt_dec.get_weights())
            np.save(path + "/opt_disc.npy", opt_disc.get_weights())
            self.save_test_image_full(path, models_ls=models_ls)

    def truncated_normal(self, shape, bound=2.0, dtype=tf.float32):
        seed1, seed2 = random_seed.get_seed(tf.random.uniform((), tf.int32.min, tf.int32.max, dtype=tf.int32))
        return tf.random.stateless_parameterized_truncated_normal(shape, [seed1, seed2], 0.0, 1.0, -bound, bound)

    def distribute_gen(self, x, model, bs=64):
        outls = []
        bdim = x.shape[0]
        if bdim == 1:
            bdim = 2
        for i in range(((bdim - 2) // bs) + 1):
            outls.append(model(x[i * bs : i * bs + bs], training=False))
        return np.concatenate(outls, 0)

    def get_noise_interp_multi(self, fac=1, var=2.0):
        noiseg = self.truncated_normal([1, 64], var, dtype=tf.float32)

        if var < 1.75:
            var = 1.75

        noisels = [
            tf.concat([self.truncated_normal([1, 64], var, dtype=tf.float32), noiseg], -1) for i in range(2 + (fac - 1))
        ]
        rls = [
            tf.linspace(noisels[k], noisels[k + 1], self.args.latlen + 1, axis=-2)[:, :-1, :]
            for k in range(len(noisels) - 1)
        ]
        return tf.concat(rls, 0)

    def stfunc(self, z, var, models_ls):

        critic, gen, enc, dec, enc2, dec2, critic_rec, gen_ema, [opt_dec, opt_disc] = models_ls

        var = 0.01 + (3.5 * (var / 100.0))

        if z == 0:
            fac = 1
        elif z == 1:
            fac = 5
        else:
            fac = 10

        bef = time.time()
        ab = self.distribute_gen(self.get_noise_interp_multi(fac, var), gen_ema)
        abls = tf.split(ab, ab.shape[0], 0)
        ab = tf.concat(abls, -2)
        abls = tf.split(ab, ab.shape[-2] // 16, -2)
        abi = tf.concat(abls, 0)

        chls = []
        for channel in range(2):

            ab = self.distribute_dec2(
                abi[:, :, :, channel * self.args.latdepth : channel * self.args.latdepth + self.args.latdepth],
                dec2,
                bs=128,
            )
            abls = tf.split(ab, ab.shape[-2] // self.args.shape, -2)
            ab = tf.concat(abls, 0)

            ab_m, ab_p = self.distribute_dec(ab, dec, bs=128)
            abwv = self.conc_tog_specphase(ab_m, ab_p)
            chls.append(abwv)

        print(
            f"Time for complete generation pipeline: {time.time()-bef} s        {int(np.round((fac*23.)/(time.time()-bef)))}x faster than Real Time!"
        )

        abwvc = np.clip(np.squeeze(np.stack(chls, -1)), -1.0, 1.0)
        spec = np.flip(
            np.array(
                tf.transpose(
                    self.wv2spec_hop((abwvc[: 23 * self.args.sr, 0] + abwvc[: 23 * self.args.sr, 1]) / 2.0, 80.0, 256),
                    [1, 0],
                )
            ),
            -2,
        )

        return (
            spec,
            (self.args.sr, np.int16(abwvc * 32767.0)),
        )

    def render_gradio(self, models_ls, train=True):
        article_text = "Original work by Marco Pasini ([Twitter](https://twitter.com/marco_ppasini)) at Johannes Kepler Universität Linz. Supervised by Jan Schlüter."

        def gradio_func(x, y):
            return self.stfunc(x, y, models_ls)

        iface = gr.Interface(
            fn=gradio_func,
            inputs=[
                gr.inputs.Radio(
                    choices=["23s", "1m 58s", "3m 57s"], type="index", default="1m 58s", label="Generated Music Length",
                ),
                gr.inputs.Slider(
                    minimum=0,
                    maximum=100,
                    step=1,
                    default=25,
                    label="Stability[left]/Variety[right] Tradeoff (Truncation Trick)",
                ),
            ],
            outputs=[
                gr.outputs.Image(label="Log-MelSpectrogram of Generated Audio (first 23 s)"),
                gr.outputs.Audio(type="numpy", label="Generated Audio"),
            ],
            title="musika!",
            description="The generator used for this demo is updated *after* every epoch!",
            article=article_text,
        )

        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("CLICK ON LINK BELOW TO OPEN GRADIO INTERFACE")
        if train:
            iface.launch(prevent_thread_lock=True)
        else:
            iface.launch()
        # iface.launch(share=True, enable_queue=True)
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
