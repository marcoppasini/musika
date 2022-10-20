import os

import numpy as np
import tensorflow as tf
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
            try:
                tp, ext = os.path.splitext(p)
                bname = os.path.basename(tp)
                wvo = AudioSegment.from_file(p, format=ext[1:])
                wvo = wvo.set_frame_rate(self.args.sr)
                wvls = wvo.split_to_mono()
                wvls = [s.get_array_of_samples() for s in wvls]
                wv = np.array(wvls).T.astype(np.float32)
                wv /= np.iinfo(wvls[0].typecode).max
                yield np.squeeze(wv), bname
            except Exception as e:
                print(e)
                print("Exception ignored! Continuing...")
                pass

    # def create_dataset(self):
    #     self.ds = (
    #         tf.data.Dataset.from_generator(
    #             self.audio_generator, output_signature=(tf.TensorSpec(shape=(None, 2), dtype=tf.float32))
    #         )
    #         .prefetch(tf.data.experimental.AUTOTUNE)
    #         .apply(tf.data.experimental.ignore_errors())
    #     )

    def compress_files(self, models_ls=None):
        critic, gen, enc, dec, enc2, dec2, gen_ema, [opt_dec, opt_disc], switch = models_ls
        # self.create_dataset()
        os.makedirs(self.args.save_path, exist_ok=True)
        c = 0
        time_compression_ratio = 16  # TODO: infer time compression ratio
        shape2 = self.args.shape
        pbar = tqdm(self.audio_generator(), position=0, leave=True, total=len(self.paths))

        for (wv,bname) in pbar:

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
                            ds = []
                            num = x.shape[1] // self.args.shape
                            rn = 0
                            for i in range(num):
                                ds.append(
                                    x[:, rn + (i * self.args.shape) : rn + (i * self.args.shape) + self.args.shape, :]
                                )
                            del x
                            ds = tf.convert_to_tensor(ds, dtype=tf.float32)
                            lat = self.U.distribute_enc(ds, enc)
                            del ds
                            lat = tf.split(lat, lat.shape[0], 0)
                            lat = tf.concat(lat, -2)
                            lat = tf.squeeze(lat)

                            switch = False
                            if lat.shape[0] > (self.args.max_lat_len * time_compression_ratio):
                                switch = True
                                ds2 = []
                                num2 = lat.shape[-2] // shape2
                                rn2 = 0
                                for j in range(num2):
                                    ds2.append(lat[rn2 + (j * shape2) : rn2 + (j * shape2) + shape2, :])
                                ds2 = tf.convert_to_tensor(ds2, dtype=tf.float32)
                                lat = self.U.distribute_enc(tf.expand_dims(ds2, -3), enc2)
                                del ds2
                                lat = tf.split(lat, lat.shape[0], 0)
                                lat = tf.concat(lat, -2)
                                lat = tf.squeeze(lat)
                                chls.append(lat)

                        if lat.shape[0] > self.args.max_lat_len and switch:

                            lat = tf.concat(chls, -1)

                            del chls

                            latc = lat[: (lat.shape[0] // self.args.max_lat_len) * self.args.max_lat_len, :]
                            latc = tf.split(latc, latc.shape[0] // self.args.max_lat_len, 0)
                            for el in latc:
                                np.save(self.args.save_path + f"/{bname}_{c}.npy", el)
                                c += 1
                                pbar.set_postfix({"Saved Files": c})
                            np.save(self.args.save_path + f"/{bname}_{c}.npy", lat[-self.args.max_lat_len :, :])
                            c += 1
                            pbar.set_postfix({"Saved Files": c})

                            del lat
                            del latc

            except Exception as e:
                print(e)
                print("Exception ignored! Continuing...")
                pass


    def compress_whole_files(self, models_ls=None):
        critic, gen, enc, dec, enc2, dec2, gen_ema, [opt_dec, opt_disc], switch = models_ls
        # self.create_dataset()
        os.makedirs(self.args.save_path, exist_ok=True)
        c = 0
        time_compression_ratio = 16  # TODO: infer time compression ratio
        shape2 = self.args.shape
        pbar = tqdm(self.audio_generator(), position=0, leave=True, total=len(self.paths))

        for (wv,bname) in pbar:

            try:

                # wv_len_orig = wv.shape[0]

                if wv.shape[0] > self.args.hop * self.args.shape * 2 + 3 * self.args.hop:

                    rem = (wv.shape[0] - (3 * self.args.hop)) % (self.args.shape * self.args.hop)

                    if rem != 0:
                        wv = tf.concat([wv, tf.zeros([rem,2], dtype=tf.float32)], 0)

                    chls = []
                    for channel in range(2):

                        x = wv[:, channel]
                        x = tf.expand_dims(tf.transpose(self.U.wv2spec(x, hop_size=self.args.hop), (1, 0)), -1)
                        ds = []
                        num = x.shape[1] // self.args.shape
                        rn = 0
                        for i in range(num):
                            ds.append(
                                x[:, rn + (i * self.args.shape) : rn + (i * self.args.shape) + self.args.shape, :]
                            )
                        del x
                        ds = tf.convert_to_tensor(ds, dtype=tf.float32)
                        lat = self.U.distribute_enc(ds, enc)
                        del ds
                        lat = tf.split(lat, lat.shape[0], 0)
                        lat = tf.concat(lat, -2)
                        lat = tf.squeeze(lat)



                        ds2 = []
                        num2 = lat.shape[-2] // shape2
                        rn2 = 0
                        for j in range(num2):
                            ds2.append(lat[rn2 + (j * shape2) : rn2 + (j * shape2) + shape2, :])
                        ds2 = tf.convert_to_tensor(ds2, dtype=tf.float32)
                        lat = self.U.distribute_enc(tf.expand_dims(ds2, -3), enc2)
                        del ds2
                        lat = tf.split(lat, lat.shape[0], 0)
                        lat = tf.concat(lat, -2)
                        lat = tf.squeeze(lat)
                        chls.append(lat)

                    lat = tf.concat(chls, -1)

                    del chls

                    np.save(self.args.save_path + f"/{bname}.npy", lat)
                    c += 1
                    pbar.set_postfix({"Saved Files": c})

                    del lat

            except Exception as e:
                print(e)
                print("Exception ignored! Continuing...")
                pass