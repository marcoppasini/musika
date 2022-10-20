import tensorflow as tf
import tensorboard
import numpy as np
from tqdm import tqdm
import time
import datetime
import os
import subprocess
from utils import Utils_functions
from models import Models_functions
from losses import *


class Train_functions:
    def __init__(self, args):

        self.args = args
        self.U = Utils_functions(args)
        self.M = Models_functions(args)

    def gradient_penalty(self, x, net):
        x_hat = x
        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat, _ = net(x_hat, training=True)
        gradients = t.gradient(d_hat, x_hat)
        ddx = tf.sqrt(1e-6 + tf.reduce_sum(gradients**2, axis=[1, 2, 3]))
        d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)
        return d_regularizer

    def train_all(self, a, ema, g_train=True, disc_train=True, models_ls=None):

        critic, gen, enc, dec, enc2, dec2, gen_ema, [opt_dec, opt_disc], switch = models_ls

        a = tf.expand_dims(a, -3)
        a = self.U.rand_channel_swap(a)

        noiseg = tf.random.normal([self.args.bs, self.args.coorddepth], dtype=tf.float32)

        noisel = tf.concat([tf.random.normal([self.args.bs, self.args.coorddepth], dtype=tf.float32), noiseg], -1)
        noisec = tf.concat([tf.random.normal([self.args.bs, self.args.coorddepth], dtype=tf.float32), noiseg], -1)
        noiser = tf.concat([tf.random.normal([self.args.bs, self.args.coorddepth], dtype=tf.float32), noiseg], -1)

        rl = tf.linspace(noisel, noisec, self.args.coordlen + 1, axis=-2)[:, :-1, :]
        rr = tf.linspace(noisec, noiser, self.args.coordlen + 1, axis=-2)

        noisetot = tf.concat([rl, rr], -2)

        noisetot = self.U.center_coordinate(noisetot)
        noise = self.U.crop_coordinate(noisetot)

        with tf.GradientTape() as tape_gen, tf.GradientTape() as tape_disc, tf.GradientTape() as tape_gp:
            if not disc_train:
                tape_disc.stop_recording()
            if not g_train:
                tape_gen.stop_recording()

            tape_gp.watch(a)

            ab = gen(noise, training=True)

            loss_dtr = 0.0
            loss_dtf = 0.0
            loss_gt = 0.0
            loss_did = 0.0
            loss_gp = 0.0
            if disc_train or g_train:

                ca = critic(a, training=True)
                cab = critic(ab, training=True)

                switch.assign(self.U.update_switch(switch, ca, cab))

                grad_gp = tape_gp.gradient(tf.reduce_sum(ca), [a])[0]
                loss_gp = tf.reduce_mean(tf.reduce_sum(tf.reshape(grad_gp**2, [tf.shape(grad_gp)[0], -1]), -1))

            if disc_train:

                loss_dtr = d_loss_r(ca)
                loss_dtf = d_loss_f(cab)

                loss_dt = (loss_dtr + loss_dtf) / 2.0

                loss_d = loss_dt + self.args.gp_max_weight * (-switch) * loss_gp

                if self.args.mixed_precision:
                    loss_d = opt_disc.get_scaled_loss(loss_d)

            if g_train:

                loss_gt = g_loss_f(cab)
                loss_gen = loss_gt

                if self.args.mixed_precision:
                    loss_gen = opt_dec.get_scaled_loss(loss_gen)

        if disc_train:
            grad_disc = tape_disc.gradient(loss_d, critic.trainable_weights)
            if self.args.mixed_precision:
                grad_disc = opt_disc.get_unscaled_gradients(grad_disc)
            opt_disc.apply_gradients(zip(grad_disc, critic.trainable_weights))

        if g_train:
            grad_dec = tape_gen.gradient(loss_gen, gen.trainable_variables)
            if self.args.mixed_precision:
                grad_dec = opt_dec.get_unscaled_gradients(grad_dec)
            opt_dec.apply_gradients(zip(grad_dec, gen.trainable_variables))

            ema.apply(gen.trainable_variables)

        return loss_dtr, loss_dtf, loss_gp, loss_gt

    # @tf.function(jit_compile=True)
    # def train_tot(self, a, ema, models_ls=None):
    #     return self.train_all(a, ema, g_train=True, disc_train=True, models_ls=models_ls)

    def update_lr(self, lr, opts=None):
        opt_dec, opt_disc = opts
        opt_dec.learning_rate = lr
        opt_disc.learning_rate = lr * 1.0

    def train(self, ds, models_ls=None):

        @tf.function(jit_compile=self.args.xla)
        def train_tot(a, ema, models_ls=None):
            return self.train_all(a, ema, g_train=True, disc_train=True, models_ls=models_ls)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = (
            f"{self.args.log_path}/MUSIKA_latlen_{self.args.latlen}_latdepth_{self.args.latdepth}_sr_{self.args.sr}/"
            + current_time
            + "/train"
        )
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        exp_path = f"{self.args.save_path}/MUSIKA_latlen_{self.args.latlen}_latdepth_{self.args.latdepth}_sr_{self.args.sr}_time_{current_time}"
        os.makedirs(exp_path, exist_ok=True)

        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")

        _ = subprocess.Popen(
            [
                "tensorboard",
                "--logdir",
                f"{self.args.log_path}/MUSIKA_latlen_{self.args.latlen}_latdepth_{self.args.latdepth}_sr_{self.args.sr}",
                "--port",
                "6006",
            ]
        )
        print("CLICK ON LINK BELOW TO OPEN TENSORBOARD INTERFACE")
        print("http://localhost:6006/")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")
        print("--------------------------------")

        ema = tf.train.ExponentialMovingAverage(decay=0.999)
        critic, gen, enc, dec, enc2, dec2, gen_ema, [opt_dec, opt_disc], switch = models_ls
        ema.apply(gen_ema.trainable_variables)

        self.update_lr(self.args.lr, [opt_dec, opt_disc])
        c = 0
        g = 0
        m = 0
        idloss = 0.0

        print("Preparing for Training (this can take one or two minutes)...")

        for epoch in range(self.args.epochs):
            bef = time.time()
            bef_loop = time.time()

            dtr_list = []
            dtf_list = []
            did_list = []
            gt_list = []
            id_list = []

            pbar = tqdm(
                ds,
                desc=f"Epoch {epoch}/{self.args.epochs}",
                position=0,
                leave=True,
                total=self.args.totsamples // self.args.bs,
            )

            for batchi, (wv) in enumerate(pbar):
                a = wv

                dloss_tr, dloss_tf, dloss_id, gloss_t = train_tot(a, ema, models_ls=models_ls)

                with train_summary_writer.as_default():
                    tf.summary.scalar("disc_loss_r", dloss_tr, step=m)
                    tf.summary.scalar("disc_loss_f", dloss_tf, step=m)
                    tf.summary.scalar("gen_loss", gloss_t, step=m)
                    tf.summary.scalar("gradient_penalty", dloss_id, step=m)
                    tf.summary.scalar("gp_weight", -switch.value() * self.args.gp_max_weight, step=m)
                    tf.summary.scalar("lr", self.args.lr, step=m)

                dtr_list.append(dloss_tr)
                dtf_list.append(dloss_tf)
                did_list.append(dloss_id)
                gt_list.append(gloss_t)

                c += 1
                g += 1
                m += 1

                if batchi % 20 == 0:

                    pbar.set_postfix(
                        {
                            "DR": np.mean(dtr_list[-g:], axis=0),
                            "DF": np.mean(dtf_list[-g:], axis=0),
                            "G": np.mean(gt_list[-g:], axis=0),
                            "GP": np.mean(did_list[-g:], axis=0),
                            "LR": self.args.lr,
                            "TIME": (time.time() - bef_loop) / 20,
                        }
                    )
                    bef_loop = time.time()
                nbatch = batchi

            for var, var_ema in zip(gen.trainable_variables, gen_ema.trainable_variables):
                var_ema.assign(ema.average(var))

            self.U.save_end(
                epoch,
                np.mean(gt_list[-self.args.save_every * c :], axis=0),
                np.mean(dtr_list[-self.args.save_every * c :], axis=0),
                np.mean(dtf_list[-self.args.save_every * c :], axis=0),
                n_save=self.args.save_every,
                models_ls=models_ls,
                save_path=exp_path,
            )

            c = 0
            g = 0
