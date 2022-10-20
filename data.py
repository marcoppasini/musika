import tensorflow as tf
from glob import glob

from utils import Utils_functions


class Data_functions:
    def __init__(self, args):

        self.args = args
        self.U = Utils_functions(args)

    options = tf.data.Options()
    options.experimental_deterministic = False

    @tf.function
    def read_npy(self, p):
        x = tf.reshape(
            tf.io.decode_raw(tf.io.read_file(p), tf.float32)[-(self.args.max_lat_len * self.args.latdepth * 2) :],
            [self.args.max_lat_len, self.args.latdepth * 2],
        )
        randnum = tf.random.uniform((), 0, self.args.max_lat_len - self.args.latlen, dtype=tf.int64)
        x = x[randnum : randnum + self.args.latlen, :]
        return x

    def create_dataset(self):

        print("Calculating total number of samples in data folder...")
        datalen = len(glob(self.args.train_path + "/*.npy"))
        print(f"Found {datalen} total samples")

        options = tf.data.Options()
        options.experimental_deterministic = False

        if datalen > self.args.totsamples:
            ds = tf.data.Dataset.list_files(self.args.train_path + "/*.npy").shuffle(datalen).take(self.args.totsamples)
        else:
            ds = (
                tf.data.Dataset.list_files(self.args.train_path + "/*.npy")
                .repeat((self.args.totsamples // datalen) + 1)
                .shuffle(datalen * ((self.args.totsamples // datalen) + 1))
                .take(self.args.totsamples)
            )

        ds = (
            ds.map(self.read_npy, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(self.args.bs, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
            .with_options(options)
        )  # .apply(tf.data.experimental.ignore_errors())

        print("Dataset is ready!")

        return ds
