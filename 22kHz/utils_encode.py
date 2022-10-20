import os
import time

import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import random_seed
import gradio as gr
from scipy.io.wavfile import write as write_wav
from pydub import AudioSegment
from glob import glob
from tqdm import tqdm

from utils import Utils_functions


class UtilsEncode_functions:
    def __init__(self, args):

        self.args = args
        self.U = Utils_functions(args)
        self.paths = sorted(glob(self.args.files_path + "/*"))

    def audio_generator(self):
        for p in self.paths:
            _, ext = os.path.splitext(p)
            wvo = AudioSegment.from_file(p, format=ext[1:])
            wvo = wvo.set_frame_rate(self.args.sr)
            wvls = wvo.split_to_mono()
            wvls = [s.get_array_of_samples() for s in wvls]
            wv = np.array(wvls).T.astype(np.float32)
            wv /= np.iinfo(wvls[0].typecode).max
            yield tf.squeeze(wv)

    def create_dataset(self):
        self.ds = (
            tf.data.Dataset.from_generator(
                self.audio_generator, output_signature=(tf.TensorSpec(shape=(None, 2), dtype=tf.float32))
            )
            .prefetch(tf.data.experimental.AUTOTUNE)
            .apply(tf.data.experimental.ignore_errors())
        )

    def compress_files(self, models_ls=None):
        critic, gen, enc, dec, enc2, dec2, critic_rec, gen_ema, [opt_dec, opt_disc] = models_ls
        # self.create_dataset()
        os.makedirs(self.args.save_path, exist_ok=True)
        c = 0
        time_compression_ratio = 8  # TODO: infer time compression ratio
        shape2 = self.args.shape
        pbar = tqdm(self.audio_generator(), position=0, leave=True, total=len(self.paths))
        for wv in pbar:
            try:

                if wv.shape[0] > self.args.hop * self.args.shape * 2 + 3 * self.args.hop:

                    split_limit = (
                        5 * 60 * self.args.sr
                    )  # split very long waveforms (> 5 minutes) and process separately to avoid out of memory errors

                    nsplits = (wv.shape[0] // split_limit) + 1
                    wvsplits = []
                    for ns in range(nsplits):
                        if wv.shape[0] - (ns * split_limit) > self.args.hop * self.args.shape * 2 + 3 * self.args.hop:
                            wvsplits.append(wv[ns * split_limit : (ns + 1) * split_limit, :])

                    for wv in wvsplits:

                        wv = tf.image.random_crop(
                            wv,
                            size=[
                                (((wv.shape[0] - (3 * self.args.hop)) // (self.args.shape * self.args.hop)))
                                * self.args.shape
                                * self.args.hop
                                + 3 * self.args.hop,
                                2,
                            ],
                        )

                        chls = []
                        for channel in range(2):

                            x = wv[:, channel]
                            x = tf.expand_dims(tf.transpose(self.U.wv2spec(x, hop_size=self.args.hop), (1, 0)), -1)
                            x = np.array(x, dtype=np.float32)
                            ds = []
                            num = x.shape[1] // self.args.shape
                            rn = 0
                            for i in range(num):
                                im = x[:, rn + (i * self.args.shape) : rn + (i * self.args.shape) + self.args.shape, :]
                                ds.append(im)
                            x = np.array(ds, dtype=np.float32)
                            lat = self.U.distribute_enc(x, enc)
                            latls = tf.split(lat, lat.shape[0], 0)
                            lat = tf.concat(latls, -2)
                            lat = np.array(tf.squeeze(lat), dtype=np.float32)

                            switch = False
                            if lat.shape[0] > (self.args.max_lat_len * time_compression_ratio):
                                switch = True
                                ds2 = []
                                num2 = lat.shape[-2] // shape2
                                rn2 = 0
                                for j in range(num2):
                                    im2 = lat[rn2 + (j * shape2) : rn2 + (j * shape2) + shape2, :]
                                    ds2.append(im2)
                                lat = np.array(ds2, dtype=np.float32)
                                lat = self.U.distribute_enc(np.expand_dims(lat, -3), enc2)
                                latls = tf.split(lat, lat.shape[0], 0)
                                lat = tf.concat(latls, -2)
                                lat = np.array(tf.squeeze(lat), dtype=np.float32)
                                chls.append(lat)

                        if lat.shape[0] > self.args.max_lat_len and switch:

                            lat = np.concatenate(chls, -1)

                            latc = lat[: (lat.shape[0] // self.args.max_lat_len) * self.args.max_lat_len, :]
                            latcls = tf.split(latc, latc.shape[0] // self.args.max_lat_len, 0)
                            for el in latcls:
                                np.save(self.args.save_path + f"/{c}.npy", el)
                                c += 1
                                pbar.set_postfix({"Saved Files": c})
                            np.save(self.args.save_path + f"/{c}.npy", lat[-self.args.max_lat_len :, :])
                            c += 1
                            pbar.set_postfix({"Saved Files": c})

            except Exception as e:
                print(e)
                print("Exception ignored! Continuing...")
                pass
