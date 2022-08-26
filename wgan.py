import os
import numpy as np
import pandas as pd
import scipy.optimize as spopt
import scipy.stats as stats
import pickle
import numpy.random as rand
import matplotlib as mpl
from matplotlib import rc
import time

rc('font', **{'family': 'serif', 'serif': ['cmr10']})
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['mathtext.fontset'] = 'cm'
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import Distance
import astropy.units as u
from tqdm import tqdm, trange
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers as opt
from tensorflow.keras.layers import Input, Dense, GRU, Dropout, Reshape, BatchNormalization, Activation, Concatenate
import tensorflow.keras.backend as K
import george
from george.kernels import Matern32Kernel
import time
import corner
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
from prdc import compute_prdc

# tf.config.run_functions_eagerly(True)

plt.rcParams.update({'font.size': 26})
pd.options.mode.chained_assignment = None

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class WGANModel(keras.Model):
    def __init__(
            self,
            discriminator,
            generator,
            latent_dims,
            discriminator_extra_steps=1,
            gp_weight=1.0,
    ):
        super(WGANModel, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dims = latent_dims
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    @tf.function(input_signature=[tf.TensorSpec([None, None, 10], tf.float32)])
    def call(self, inputs):
        x = self.generator(inputs)
        return x

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGANModel, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_data, fake_data):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
        diff = fake_data - real_data
        interpolated = real_data + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(1e-12 + tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, X):
        # Get the batch size
        if isinstance(X, tuple):
            X = X[0]
        batch_size = tf.shape(X)[0]
        timesteps = tf.shape(X)[1]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            noise = tf.random.normal((batch_size, self.latent_dims))
            noise = tf.reshape(noise, (batch_size, 1, self.latent_dims))
            noise = tf.repeat(noise, timesteps, 1)
            self.__call__(noise)
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(noise, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(X, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, X, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )
        # Train the generator
        # Get the latent vector
        noise = tf.random.normal((batch_size, self.latent_dims))
        noise = tf.reshape(noise, (batch_size, 1, self.latent_dims))
        noise = tf.repeat(noise, timesteps, 1)
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_data = self.generator(noise, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_data, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}


class WGAN:
    """
    Wasserstein GAN implementation for supernova light curve generation
    """

    def __init__(self, experiment='general', latent_dims=100, clr=0.0005, glr=0.0005, GP=True,
                 z_lim=None, g_dropout=0.5, c_dropout=0.5, redshift=False,
                 gen_units=100, crit_units=100, sn_type='II', n_critic=1,
                 gp_weight=10):
        """
        :param latent_dims: int, number of latent dimensions to draw random seed for generator from
        :param clr: float, initial learning rate to use for critic
        :param glr: float, initial learning rate to use for generator
        :param GP: Boolean, specifies whether to Gaussian Process interpolate training light curves
        :param z_lim: float, upper redshift limit for sample
        :param g_dropout: float, dropout fraction to use in generator model
        :param c_dropout: float, dropout fraction to use in critic model
        :param gen_units: int, number of GRU units in each layer of generator model
        :param crit_units: int, number of GRU units in each layer of generator model
        :param sn_type: string, SN class to look at
        """
        self.experiment = experiment
        self.latent_dims = latent_dims
        self.clr = clr
        self.glr = glr
        self.GP = GP
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
        self.z_lim = z_lim
        self.g_dropout = g_dropout
        self.c_dropout = c_dropout
        self.gen_units = gen_units
        self.crit_units = crit_units
        self.sn_type = sn_type
        self.redshift = redshift
        # WGAN Paper guidance-----------------------------
        self.n_critic = n_critic
        self.c_optimizer = opt.Adam(lr=self.clr, beta_1=0.5, beta_2=0.9)
        self.g_optimizer = opt.Adam(lr=self.glr, beta_1=0.5, beta_2=0.9)
        self.gp_weight = gp_weight
        # ------------------------------------------------
        # --

        type_dict = {}

        type_dict[1] = [834, 825, 840, 845, 819, 851]  # IIn
        type_dict[2] = [830, 829, 822, 847, 865, 849, 850, 848, 832, 859, 804]  # IIb
        type_dict[0] = [864, 866, 835, 844, 837, 838, 855, 863, 843, 861, 860, 858, 857, 856, 852, 839,
                        801, 802, 867, 824, 808, 811, 831, 817]
        type_dict[3] = [833, 807, 854, 813, 815, 816, 803, 827, 821, 842, 841, 828, 818]  # Ib
        type_dict[4] = [846, 805, 862, 823, 814, 812, 810]  # Ic
        type_dict[5] = [826, 806, 853, 836, 820, 809]  # Ic-BL

        self.type_dict = {}
        for key, vals in type_dict.items():
            for val in vals:
                self.type_dict[val] = key
        self.class_label_decoder = {0.0: 'II', 1.0: 'IIn', 2.0: 'IIb', 3.0: 'Ib', 4.0: 'Ic', 5.0: 'Ic-BL'}
        self.class_label_encoder = {val: key for key, val in self.class_label_decoder.items()}
        # --
        self.n_output = 12
        if self.redshift:
            self.n_output += 1
        self.name = f'WGAN_DES_sim_{self.sn_type}_CCSNe_clr{self.clr}_glr{self.glr}_ld{self.latent_dims}' \
                    f'_GP{self.GP}_zlim{self.z_lim}_gN{self.gen_units}_cN{self.crit_units}' \
                    f'_gd{self.g_dropout}_cd{self.c_dropout}' \
                    f'_ncrit{self.n_critic}_gpw{self.gp_weight}_redshift{self.redshift}'

        if not os.path.exists(os.path.join('Data', 'Models', 'Weights', self.experiment)):
            os.mkdir(os.path.join('Data', 'Models', 'Weights', self.experiment))
        if not os.path.exists(os.path.join('Data', 'Models', 'Plots', self.experiment)):
            os.mkdir(os.path.join('Data', 'Models', 'Plots', self.experiment))
        if not os.path.exists(os.path.join('Data', 'Models', 'LCs', self.experiment)):
            os.mkdir(os.path.join('Data', 'Models', 'LCs', self.experiment))
        self.weight_root = os.path.join('Data', 'Models', 'Weights', self.experiment, self.name)
        self.plot_root = os.path.join('Data', 'Models', 'Plots', self.experiment, self.name)
        self.lc_root = os.path.join('Data', 'Models', 'LCs', self.experiment, self.name)
        if not os.path.exists(self.weight_root):
            os.mkdir(self.weight_root)
        if not os.path.exists(self.plot_root):
            os.mkdir(self.plot_root)
        if not os.path.exists(self.lc_root):
            os.mkdir(self.lc_root)
        self.dataset_name = f'WGAN_DES_sim_colourFalse_GP{self.GP}_zlim{self.z_lim}_redshift'
        self.dataset_path = os.path.join('Data', 'Datasets', f'{self.dataset_name}.csv')
        if os.path.exists(self.dataset_path):
            self.train_df = pd.read_csv(self.dataset_path)
            self.scaling_factors = pickle.load(open(os.path.join('Data', 'Datasets', f'{self.dataset_name}'
                                                                                     f'_scaling_factors.pkl'), 'rb'))
        else:
            print('Dataset does not already exist, creating now...')
            self.train_df, self.scaling_factors = self.__prepare_dataset__()
        min = np.min(np.r_[self.train_df.g_err, self.train_df.r_err,
                           self.train_df.i_err, self.train_df.z_err])
        max = np.max(np.r_[self.train_df.g_err, self.train_df.r_err,
                           self.train_df.i_err, self.train_df.z_err])
        min_z, max_z = self.train_df.redshift.min(), self.train_df.redshift.max()
        self.scaling_factors.append(min)
        self.scaling_factors.append(max)
        self.scaling_factors.append(min_z)
        self.scaling_factors.append(max_z)
        self.train_df[['g_err', 'r_err', 'i_err', 'z_err']] = (self.train_df[['g_err', 'r_err', 'i_err', 'z_err']]
                                                               - min) / (max - min)
        self.train_df[['redshift']] = (self.train_df[['redshift']] - min_z) / (max_z - min_z)
        self.train_df = self.train_df[self.train_df.sn_type == self.class_label_encoder[sn_type]]
        self.wgan_dir = os.path.join(self.weight_root, 'model_weights')

        '''
        # Optimizer

        # Build discriminator
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss, optimizer=self.c_optimizer)
        print(self.critic.summary())

        # Build generator
        self.generator = self.build_generator()
        print(self.generator.summary())

        # Build combined model

        i = Input(shape=(None, self.latent_dims))
        lcs = self.generator(i)

        self.critic.trainable = False

        valid = self.critic(lcs)

        self.combined = Model(i, valid)
        self.combined.compile(loss=self.wasserstein_loss, optimizer=self.g_optimizer)
        print(self.combined.summary())'''
        self.wgan = WGANModel(self.build_critic(), self.build_generator(), self.latent_dims,
                              gp_weight=self.gp_weight, discriminator_extra_steps=self.n_critic)
        self.wgan.compile(d_optimizer=self.c_optimizer, g_optimizer=self.g_optimizer,
                          g_loss_fn=self.generator_loss, d_loss_fn=self.discriminator_loss)

    def __prepare_dataset__(self):
        """
        Builds dataset
        :return: all_new_df, Pandas DataFrame containing training data
                 scaling_factors, list of factor to scale numbers from 0 to 1 back to physical values
        """
        data_dir = 'Data/DESSIMBIAS5YRCC_V19/PIP_MV_GLOBAL_BIASCOR_DESSIMBIAS5YRCC_V19'
        x = os.listdir(data_dir)
        x.sort()

        all_df = None

        types = []
        total, sn_type, z_cut, n_obs = 0, 0, 0, 0

        print('Reading in files...')
        for file in tqdm(x):
            # if all_df is not None and all_df.shape[0] > 100:
            #    break
            if 'HEAD' in file:
                head_file = file
                phot_file = file.replace('HEAD', 'PHOT')
                # try:
                head = Table.read(os.path.join(data_dir, head_file), hdu=1)
                # except:
                #     continue
                head_df = head.to_pandas()
                total += head_df.shape[0]
                if self.sn_type == 'Ia':
                    head_df = head_df[head_df.SNTYPE == 101]
                else:
                    head_df = head_df[head_df.SNTYPE == 120]
                sn_type += head_df.shape[0]
                head_df = head_df[head_df.HOSTGAL_SPECZ < self.z_lim]
                z_cut += head_df.shape[0]
                head_df = head_df[head_df.NOBS > 20]
                n_obs += head_df.shape[0]
                for col, dtype in head_df.dtypes.items():
                    if dtype == object:  # Only process byte object columns.
                        head_df[col] = head_df[col].apply(lambda x: x.decode("utf-8").lstrip().rstrip())
                phot = Table.read(os.path.join(data_dir, phot_file), hdu=1)
                phot_df = phot.to_pandas()
                for col, dtype in phot_df.dtypes.items():
                    if dtype == object:  # Only process byte object columns.
                        phot_df[col] = phot_df[col].apply(lambda x: x.decode("utf-8").lstrip().rstrip())
                phot_df['mag'] = -2.5 * np.log10(phot_df.FLUXCAL) + 27
                phot_df['mag_err'] = (2.5 / np.log(10)) * phot_df.FLUXCALERR / phot_df.FLUXCAL

                for i, row in head_df.reset_index().iterrows():
                    sn_phot_df = phot_df.iloc[row.PTROBS_MIN:row.PTROBS_MAX + 1, :]
                    sn_phot_df = sn_phot_df[sn_phot_df.MJD > 50000]
                    sn_phot_df['t'] = (sn_phot_df.MJD - row.SIM_PEAKMJD) / (1 + row.HOSTGAL_SPECZ)
                    sn_phot_df = sn_phot_df[['t', 'FLT', 'mag', 'mag_err']]
                    sn_phot_df['sn'] = row.SNID
                    sn_phot_df['redshift'] = row.HOSTGAL_SPECZ
                    if row.SIM_TEMPLATE_INDEX in self.type_dict.keys():
                        sn_phot_df['sn_type'] = self.type_dict[row.SIM_TEMPLATE_INDEX]
                        types.append(self.type_dict[row.SIM_TEMPLATE_INDEX])
                    else:
                        sn_phot_df['sn_type'] = row.SIM_TEMPLATE_INDEX
                        types.append(row.SIM_TEMPLATE_INDEX)
                    if all_df is None:
                        all_df = sn_phot_df.copy()
                    else:
                        all_df = pd.concat([all_df, sn_phot_df])
            else:
                continue
        # Drop nans and infs, they only cause problems!
        all_df = all_df[~(np.isnan(all_df.mag) | np.isinf(all_df.mag))]

        # Get bounds for light curve scaling and rescale so between 0 and 1
        all_df = all_df[all_df.mag < 27.428694]  # 99th percentile
        min_t, max_t = all_df.t.min(), all_df.t.max()
        min_mag, max_mag = all_df.mag.max(), all_df.mag.min()
        all_df = all_df[all_df.mag_err < 1]  # Remove points with magnitude uncertainties greater than 1 mag
        scaled_mag = (all_df['mag'] - min_mag) / (max_mag - min_mag)
        all_df['mag'] = scaled_mag
        all_df['mag_err'] /= np.abs(max_mag - min_mag)
        # all_df = all_df[~(all_df.mag < 0.1)]  # Remove bad points # & (all_df.mag_err > 0.5))]

        all_new_df = None
        used_count, skip_count, gp_error, no_peak, no_points, nans, less_than_zero = 0, 0, 0, 0, 0, 0, 0

        sn_list = list(all_df.sn.unique())

        for i, sn in tqdm(enumerate(sn_list), total=len(sn_list)):
            sn_df = all_df[all_df.sn == sn]
            new_sn_df = pd.DataFrame()
            lens = [sn_df[sn_df.FLT == f].shape[0] for f in ['g', 'r', 'i', 'z']]
            if np.std(lens) > 0:
                use_len = np.min(lens)
            else:
                use_len = lens[0]

            skip = False
            plt.close('all')
            plt.figure()
            for f in ['g', 'r', 'i', 'z']:
                fdf = sn_df[sn_df.FLT == f]  # .sort_values('t')
                gp = george.GP(Matern32Kernel(1))
                if use_len < fdf.shape[0]:
                    drop_count = fdf.shape[0] - use_len
                    skip_num = int(np.floor(drop_count / 2))
                    fdf = fdf.iloc[skip_num: skip_num + use_len, :]
                t = (fdf.t - min_t) / (max_t - min_t)
                fit_t = (fdf.t - min_t) / (max_t - min_t)
                new_sn_df[f'{f}_t'] = t.values
                sn_t_min, sn_t_max = np.min(t), np.max(t)
                y = fdf.mag
                y_err = fdf.mag_err

                # Gradient function for optimisation of kernel size------------------------------

                def ll(p):
                    gp.set_parameter_vector(p)
                    return -gp.log_likelihood(y, quiet=True)

                def grad_ll(p):
                    gp.set_parameter_vector(p)
                    return -gp.grad_log_likelihood(y, quiet=True)

                try:
                    gp.compute(t, y_err)
                    p0 = gp.kernel.get_parameter_vector()[0]
                    results = spopt.minimize(ll, p0, jac=grad_ll)
                    mu, cov = gp.predict(y, t)
                    std = np.sqrt(np.diag(cov))
                except:
                    if not skip:
                        gp_error += 1
                    skip = True
                    continue
                test_t, test_mu = t[mu > 0], mu[mu > 0]
                if len(test_mu) == 0:
                    if not skip:
                        no_points += 1
                    skip = True
                    continue
                if np.argmax(test_mu) == 0 or (np.argmin(test_mu) == 0 and np.argmax(test_mu) == len(test_mu) - 1) \
                        or len(mu[mu > 0]) < 8:
                    if not skip:
                        no_peak += 1
                    skip = True
                    continue
                if any(np.isnan(val) for val in mu):
                    if not skip:
                        nans += 1
                    skip = True
                elif np.count_nonzero(np.isnan(mu)) > 0:
                    if not skip:
                        nans += 1
                    skip = True

                new_sn_df[f] = mu
                new_sn_df[f'{f}_err'] = std

            new_sn_df['sn'] = sn
            new_sn_df['sn_type'] = sn_df.sn_type.values[0]
            new_sn_df['redshift'] = sn_df.redshift.values[0]
            if 'g' in new_sn_df.columns and 'r' in new_sn_df.columns and 'i' in new_sn_df.columns and 'z' \
                    in new_sn_df.columns:
                new_sn_df = new_sn_df.loc[np.min(new_sn_df[['g', 'r', 'i', 'z']].values, axis=1) >= 0, :]
                if new_sn_df.shape[0] < 8:
                    less_than_zero += 1
                    skip = True
            if skip:
                skip_count += 1
                continue
            used_count += 1
            if all_new_df is None:
                all_new_df = new_sn_df.copy()
            else:
                all_new_df = pd.concat([all_new_df, new_sn_df])
        print(used_count, skip_count, gp_error, less_than_zero, nans, no_peak, no_points)
        all_new_df.to_csv(self.dataset_path, index=False)
        scaling_factors = [min_t, max_t, min_mag, max_mag]
        pickle.dump(scaling_factors,
                    open(os.path.join('Data', 'Datasets', f'{self.dataset_name}_scaling_factors.pkl'), 'wb'))
        return all_new_df, scaling_factors

    def wasserstein_loss(self, y_true, y_pred):
        """
        Loss function for Wasserstein GAN
        :param y_true: True labels of data
        :param y_pred: Output of critic model
        :return: Loss
        """
        return K.mean(y_true * y_pred)

    def build_generator(self):
        """
        Builds generator model
        :return: model, keras Model object for generator
        """
        input = Input(shape=(None, self.latent_dims))
        gru1 = GRU(self.gen_units, activation='tanh', return_sequences=True)(input)
        dr1 = Dropout(self.g_dropout)(gru1)
        gru2 = GRU(self.gen_units, activation='tanh', return_sequences=True)(dr1)
        dr2 = Dropout(self.g_dropout)(gru2)
        output = GRU(self.n_output, return_sequences=True, activation='tanh')(dr2)
        model = Model(input, output)
        return model

    def build_critic(self):
        """
        Builds critic model
        :return: model, keras Model object for critic
        """
        input = Input(shape=(None, self.n_output))
        gru1 = GRU(self.crit_units, return_sequences=True)(input)
        dr1 = Dropout(self.c_dropout)(gru1)
        gru2 = GRU(self.crit_units)(dr1)
        dr2 = Dropout(self.c_dropout)(gru2)
        output = Dense(1, activation=None)(dr2)
        model = Model(input, output)
        return model

    def plot_train_sample(self):
        """
        Generates light curve plots for training sample
        """
        print('Generating plots for training sample...')
        if not os.path.exists(os.path.join(self.plot_root, 'Training_sample')):
            os.mkdir(os.path.join(self.plot_root, 'Training_sample'))
        for sn in tqdm(self.train_df.sn.unique()):
            sndf = self.train_df[self.train_df.sn == sn]
            plt.figure(figsize=(12, 8))
            for b_ind, band in enumerate(['g', 'r', 'i', 'z']):
                ax = plt.subplot(2, 2, b_ind + 1)
                ax.errorbar(sndf[f'{band}_t'], sndf[band], yerr=sndf[f'{band}_err'], fmt='x', label=band)
                ax.legend()
            plt.savefig(os.path.join(self.plot_root, 'Training_sample', f'{sn}.jpg'))
            plt.close('all')

    @staticmethod
    def discriminator_loss(real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    @staticmethod
    def generator_loss(fake_img):
        return -tf.reduce_mean(fake_img)

    def train(self, epochs=100, batch_size=1, plot_interval=None):
        """
        Trains generator and critic
        :param epochs: int, number of epochs to run training for
        :param batch_size: int, size of each batch (currently only works for size of 1)
        :param plot_interval: int, number of epochs between showing examples plots
        """
        print('Starting training...')
        if not os.path.exists(os.path.join(self.plot_root, 'Train_plots')):
            os.mkdir(os.path.join(self.plot_root, 'Train_plots'))
        rng = np.random.default_rng(123)

        sne = self.train_df.sn.unique()
        n_batches = int(len(sne) / batch_size)

        if os.path.exists(self.wgan_dir):
            # current_epoch = np.max([int(val.split('.')[0]) for val in os.listdir(self.wgan_dir)])
            # print(f'Model already exists, resuming training from epoch {current_epoch}...')
            # self.wgan = keras.models.load_model(os.path.join(self.wgan_dir, f'{current_epoch}.tf'))
            raise ValueError('A weights path already exists for this model, please delete '
                             'or rename it')
        else:
            current_epoch = 0
            os.mkdir(self.wgan_dir)

        real = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        self.train_df.to_csv('input_data.csv')

        epoch_g_losses, epoch_d_losses = [], []

        for epoch in range(current_epoch, epochs):
            rng.shuffle(sne)
            g_losses, d_losses, real_predictions, fake_predictions = [], [], [], []
            t = trange(n_batches)
            for batch in t:
                # Select real data
                sn = sne[batch]
                sndf = self.train_df[self.train_df.sn == sn]
                sndf[['g_t', 'r_t', 'i_t', 'z_t']] = 2 * (sndf[['g_t', 'r_t', 'i_t', 'z_t']] - 0.5)
                sndf[['g', 'r', 'i', 'z']] = 2 * (sndf[['g', 'r', 'i', 'z']] - 0.5)
                sndf[['g_err', 'r_err', 'i_err', 'z_err']] = 2 * (sndf[['g_err', 'r_err', 'i_err', 'z_err']]
                                                                  - 0.5)
                if self.redshift:
                    # sndf[['redshift']] = 2 * (sndf[['redshift']] - 0.5)
                    X = sndf[['g_t', 'r_t', 'i_t', 'z_t', 'g', 'r', 'i', 'z', 'g_err',
                              'r_err', 'i_err', 'z_err', 'redshift']].values
                else:
                    X = sndf[['g_t', 'r_t', 'i_t', 'z_t', 'g', 'r', 'i', 'z', 'g_err',
                                      'r_err', 'i_err', 'z_err']].values

                sn_type = X[0, -1]
                X = X.reshape((1, *X.shape))
                d_loss, g_loss = self.wgan.train_on_batch(X)

                if np.count_nonzero(np.isnan(X)) > 0:
                    continue

                # noise = rand.uniform(-1, 1, size=(batch_size, self.latent_dims))
                noise = rand.normal(size=(batch_size, self.latent_dims))
                noise = np.reshape(noise, (batch_size, 1, self.latent_dims))
                noise = np.repeat(noise, X.shape[1], 1)

                test_gen_lcs = self.wgan.generator.predict(noise)
                if np.count_nonzero(np.isnan(test_gen_lcs)) > 0:
                    raise ValueError('NaN generated, check how this happened')

                d_losses.append(d_loss)
                g_losses.append(g_loss)
                t.set_description(f'g_loss={np.around(np.mean(g_losses), 5)},'
                                  f' d_loss={np.around(np.mean(d_losses), 5)}')
                t.refresh()
            self.wgan.save(os.path.join(self.wgan_dir, f'{epoch + 1}.tf'))
            full_g_loss = np.mean(g_losses)
            full_d_loss = np.mean(d_losses)
            print(f'{epoch + 1}/{epochs} g_loss={full_g_loss}, d_loss={full_d_loss}')  # , '
            # f'Real prediction: {np.mean(real_predictions)} +- {np.std(real_predictions)}, '
            # f'Fake prediction: {np.mean(fake_predictions)} +- {np.std(fake_predictions)}')
            # f' Ranges: x [{np.min(gen_lcs[:, :, 0])}, {np.max(gen_lcs[:, :, 0])}], '
            # f'y [{np.min(gen_lcs[:, :, 1])}, {np.max(gen_lcs[:, :, 1])}]')

            plot_test = test_gen_lcs[0, :, :]
            fig = plt.figure(figsize=(12, 8))
            x = plot_test[:, 0]
            X = X.reshape((*X.shape[1:],))
            for f_ind, f in enumerate(['g', 'r', 'i', 'z']):
                ax = fig.add_subplot(2, 2, f_ind + 1)
                x, y, y_err = plot_test[:, f_ind], plot_test[:, f_ind + 4], plot_test[:, f_ind + 8]
                ax.errorbar(x, y, yerr=y_err, label=f, fmt='x')
                ax.errorbar(X[:, f_ind], X[:, f_ind + 4], yerr=X[:, f_ind + 8], fmt='x')

                ax.legend()
            plt.suptitle(f'Epoch {epoch + 1}/{epochs}')  #: Type {self.class_label_dict[sn_type]}')
            plt.savefig(os.path.join(self.plot_root, 'Train_plots', f'{epoch + 1}.png'))
            if plot_interval is not None:
                if (epoch + 1) % plot_interval == 0:
                    plt.show()
            plt.close('all')

    def sample_analysis(self, n=1, epoch=1000, plot_lcs=False, name_suffix=None, file_format='png'):
        if name_suffix is not None:
            self.plot_root = self.plot_root + f'_{name_suffix}'
            self.wgan_dir = os.path.join(os.path.split(self.wgan_dir)[0] + f'_{name_suffix}', 'model_weights')
        if not os.path.exists(os.path.join(self.plot_root, 'Real_Plots')):
            os.mkdir(os.path.join(self.plot_root, 'Real_Plots'))
        if not os.path.exists(os.path.join(self.plot_root, 'Generated_Plots')):
            os.mkdir(os.path.join(self.plot_root, 'Generated_Plots'))
        if not os.path.exists(os.path.join(self.plot_root, 'Summary_Plots')):
            os.mkdir(os.path.join(self.plot_root, 'Summary_Plots'))
        if not os.path.exists(os.path.join(self.plot_root, 'Summary_Plots', str(epoch))):
            os.mkdir(os.path.join(self.plot_root, 'Summary_Plots', str(epoch)))
        if not os.path.exists(os.path.join(self.plot_root, 'Generated_Plots', str(epoch))):
            os.mkdir(os.path.join(self.plot_root, 'Generated_Plots', str(epoch)))

        real_df, gen_df = None, None
        real_gp_df, gen_gp_df = None, None
        real_scaled_df, gen_scaled_df = None, None
        self.wgan = keras.models.load_model(os.path.join(self.wgan_dir, f'{epoch}.tf'))
        sn_props = ['g_max', 'r_max', 'i_max', 'z_max', 'g_Mmax', 'r_Mmax', 'i_Mmax', 'z_Mmax',
                    'g15', 'r15', 'i15', 'z15', 'g_rise', 'r_rise', 'i_rise', 'z_rise', 'g-r', 'r-i', 'i-z', 'redshift']
        real_prop_dict = {key: [] for key in sn_props}
        gen_prop_dict = {key: [] for key in sn_props}
        colour_dict = {'g': 'g', 'r': 'r', 'i': 'b', 'z': 'k'}
        label_dict = {'g_max': r'Peak $m_g$',
                      'r_max': r'Peak $m_r$',
                      'i_max': r'Peak $m_i$',
                      'z_max': r'Peak $m_z$',
                      'g_Mmax': r'Peak $M_g$',
                      'r_Mmax': r'Peak $M_r$',
                      'i_Mmax': r'Peak $M_i$',
                      'z_Mmax': r'Peak $M_z$',
                      'g15': r'$\Delta m_{g,15}$',
                      'r15': r'$\Delta m_{r,15}$',
                      'i15': r'$\Delta m_{i,15}$',
                      'z15': r'$\Delta m_{z,15}$',
                      'g_rise': r'$g$-band rise time',
                      'r_rise': r'$r$-band rise time',
                      'i_rise': r'$i$-band rise time',
                      'z_rise': r'$z$-band rise time',
                      'g-r': r'$g-r$',
                      'r-i': r'$r-i$',
                      'i-z': r'$i-z$',
                      'redshift': 'Redshift'
                      }
        bin_step_dict = {'g_max': 0.25,
                         'r_max': 0.25,
                         'i_max': 0.25,
                         'z_max': 0.25,
                         'g_Mmax': 0.25,
                         'r_Mmax': 0.25,
                         'i_Mmax': 0.25,
                         'z_Mmax': 0.25,
                         'g15': 0.1,
                         'r15': 0.1,
                         'i15': 0.1,
                         'z15': 0.1,
                         'g_rise': 4,
                         'r_rise': 4,
                         'i_rise': 4,
                         'z_rise': 4,
                         'g-r': 0.25,
                         'r-i': 0.25,
                         'i-z': 0.25,
                         'redshift': 0.02
                         }
        gen_number = 1
        for sn in tqdm(self.train_df.sn.unique(), total=len(self.train_df.sn.unique())):
            # if len(real_prop_dict['g15']) > 40:
            #     break
            sndf = self.train_df[self.train_df.sn == sn]
            # Fit for real SN first
            if self.redshift:
                sndf = sndf[['sn', 'g_t', 'r_t', 'i_t', 'z_t', 'g', 'r', 'i', 'z', 'g_err',
                             'r_err', 'i_err', 'z_err', 'redshift']]
            else:
                sndf = sndf[['sn', 'g_t', 'r_t', 'i_t', 'z_t', 'g', 'r', 'i', 'z', 'g_err',
                             'r_err', 'i_err', 'z_err']]
            timesteps = sndf.shape[0]
            full_sndf = sndf.copy()

            gen_tries, gen_count = 0, 0
            # for i in range(n):
            while gen_count < n:
                if gen_tries > 20 and gen_count == 0:
                    break
                noise = rand.normal(size=(1, self.latent_dims))
                noise = np.reshape(noise, (1, 1, self.latent_dims))
                noise = np.repeat(noise, timesteps, 1)
                gen_lcs = self.wgan.generator.predict(noise)
                X = gen_lcs[0, :, :]
                days = X[:, 0:4] * (self.scaling_factors[1] - self.scaling_factors[0])
                day_diffs = np.diff(days, axis=0)
                gen_tries += 1
                if np.min(day_diffs.flatten()) < 0.5:
                    continue
                gen_count += 1

                gen_sndf = pd.DataFrame(X, columns=sndf.columns[1:])

                gen_sndf['sn'] = f'gen_{gen_number}'
                gen_number += 1
                full_sndf = pd.concat([full_sndf, gen_sndf])

            for sname in full_sndf.sn.unique():
                sndf = full_sndf[full_sndf.sn == sname]
                gen = True if 'gen' in str(sname) else False

                if gen:
                    sndf[['g_t', 'r_t', 'i_t', 'z_t']] = (sndf[['g_t', 'r_t', 'i_t', 'z_t']] + 1) / 2
                    sndf[['g', 'r', 'i', 'z']] = (sndf[['g', 'r', 'i', 'z']] + 1) / 2
                    sndf[['g_err', 'r_err', 'i_err', 'z_err']] = (sndf[['g_err', 'r_err', 'i_err', 'z_err']] + 1) / 2
                    sndf[['redshift']] = (sndf[['redshift']] + 1) / 2
                sndf[['g_err', 'r_err', 'i_err', 'z_err']] = sndf[['g_err', 'r_err', 'i_err', 'z_err']] * (
                        self.scaling_factors[5] - self.scaling_factors[4]) + self.scaling_factors[4]
                all_t = np.r_[sndf.g_t.values, sndf.r_t.values, sndf.i_t.values, sndf.z_t.values]
                day_step = 0.1
                t_step = day_step / (self.scaling_factors[1] - self.scaling_factors[0])
                fit_t = np.arange(all_t.min(), all_t.max(), t_step)
                snfitdf = pd.DataFrame(fit_t, columns=['t'])

                for f in ['g', 'r', 'i', 'z']:
                    gp = george.GP(Matern32Kernel(1))
                    t, y, y_err = sndf[f'{f}_t'], sndf[f], sndf[f'{f}_err']
                    gp.compute(t, y_err)
                    p0 = gp.kernel.get_parameter_vector()[0]

                    def ll(p):
                        gp.set_parameter_vector(p)
                        return -gp.log_likelihood(y, quiet=True)

                    def grad_ll(p):
                        gp.set_parameter_vector(p)
                        return -gp.grad_log_likelihood(y, quiet=True)

                    results = spopt.minimize(ll, p0, jac=grad_ll)
                    mu, cov = gp.predict(y, fit_t)
                    std = np.sqrt(np.diag(cov))
                    snfitdf[f] = mu
                    snfitdf[f'{f}_err'] = std

                sndf[['g_t', 'r_t', 'i_t', 'z_t']] = sndf[['g_t', 'r_t', 'i_t', 'z_t']] * \
                                                     (self.scaling_factors[1] - self.scaling_factors[0]) + \
                                                     self.scaling_factors[0]
                sndf[['g', 'r', 'i', 'z']] = sndf[['g', 'r', 'i', 'z']] * (
                        self.scaling_factors[3] - self.scaling_factors[2]) + self.scaling_factors[2]
                sndf[['g_err', 'r_err', 'i_err', 'z_err']] = sndf[['g_err', 'r_err', 'i_err', 'z_err']] * (
                        self.scaling_factors[3] - self.scaling_factors[2]) * -1
                snfitdf['t'] = snfitdf['t'] * (self.scaling_factors[1] - self.scaling_factors[0]) + \
                               self.scaling_factors[0]
                snfitdf[['g', 'r', 'i', 'z']] = snfitdf[['g', 'r', 'i', 'z']] * (
                        self.scaling_factors[3] - self.scaling_factors[2]) + self.scaling_factors[2]
                snfitdf[['g_err', 'r_err', 'i_err', 'z_err']] = snfitdf[['g_err', 'r_err', 'i_err', 'z_err']] * (
                        self.scaling_factors[3] - self.scaling_factors[2]) * -1
                if self.redshift:
                    sndf[['redshift']] = sndf[['redshift']] * (
                            self.scaling_factors[7] - self.scaling_factors[6]) + self.scaling_factors[6]
                    redshift = sndf.redshift.mean()
                    dist_mod = 5 * np.log10(Distance(z=redshift, cosmology=self.cosmo, unit=u.Mpc).value * 1e5)

                maxes = [snfitdf[snfitdf[f] == snfitdf[f].min()]['t'].values[0] for f in ['g', 'r', 'i', 'z']]
                tmax = np.min(maxes)
                first_max = [np.argmin(snfitdf[f].values) for f in ['g', 'r', 'i', 'z']]
                if 0 in first_max:
                    continue
                # tmax = snfitdf[snfitdf.g == snfitdf.g.min()]['t'].values[0]  # Just for g-band

                first_t = np.min(sndf[['g_t', 'r_t', 'i_t', 'z_t']].values.flatten())
                t_shift = tmax
                t0 = first_t - t_shift
                sndf[['g_t', 'r_t', 'i_t', 'z_t']] -= t_shift
                snfitdf['t'] -= t_shift
                snfitdf['first_g'] = sndf[['g_t']].min().values[0]
                snfitdf['first_r'] = sndf[['r_t']].min().values[0]
                snfitdf['first_i'] = sndf[['i_t']].min().values[0]
                snfitdf['first_z'] = sndf[['z_t']].min().values[0]
                if self.redshift:
                    sndf[['g_tz', 'r_tz', 'i_tz', 'z_tz']] = sndf[['g_t', 'r_t', 'i_t', 'z_t']] / (1 + redshift)
                    sndf[['g_abs', 'r_abs', 'i_abs', 'z_abs']] = sndf[['g', 'r', 'i', 'z']] - dist_mod
                    snfitdf[['tz']] = snfitdf[['t']] / (1 + redshift)
                    snfitdf[['g_abs', 'r_abs', 'i_abs', 'z_abs']] = sndf[['g', 'r', 'i', 'z']] - dist_mod
                    snfitdf['first_gz'] = sndf[['g_tz']].min().values[0]
                    snfitdf['first_rz'] = sndf[['r_tz']].min().values[0]
                    snfitdf['first_iz'] = sndf[['i_tz']].min().values[0]
                    snfitdf['first_zz'] = sndf[['z_tz']].min().values[0]
                if -snfitdf['t'].min() > 3 * snfitdf['t'].max():  # Remove objects which don't peak
                    continue
                if gen:
                    if gen_df is None:
                        gen_df = sndf.copy()
                    else:
                        gen_df = pd.concat([gen_df, sndf])
                else:
                    if real_df is None:
                        real_df = sndf.copy()
                    else:
                        real_df = pd.concat([real_df, sndf])
                if gen:
                    if gen_gp_df is None:
                        gen_gp_df = snfitdf.copy()
                    else:
                        gen_gp_df = pd.concat([gen_gp_df, snfitdf])
                else:
                    if real_gp_df is None:
                        real_gp_df = snfitdf.copy()
                    else:
                        real_gp_df = pd.concat([real_gp_df, snfitdf])

                # Scaled light curves
                scaled_sndf = sndf.copy()
                for f_ind, f in enumerate(['g', 'r', 'i', 'z']):
                    fmax = scaled_sndf[f].min()
                    fabsmax = scaled_sndf[f'{f}_abs'].min()
                    scaled_sndf[f] -= fmax
                    scaled_sndf[f'{f}_abs'] -= fabsmax
                if gen:
                    if gen_scaled_df is None:
                        gen_scaled_df = scaled_sndf.copy()
                    else:
                        gen_scaled_df = pd.concat([gen_scaled_df, scaled_sndf])
                else:
                    if real_scaled_df is None:
                        real_scaled_df = scaled_sndf.copy()
                    else:
                        real_scaled_df = pd.concat([real_scaled_df, scaled_sndf])

                # Get SN properties
                if plot_lcs:
                    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 8))
                stop = 0

                for f_ind, f in enumerate(['g', 'r', 'i', 'z']):
                    max_df = snfitdf[snfitdf[f] == snfitdf[f].min()]
                    if max_df.shape[0] > 1:
                        max_df = max_df.iloc[[0], :]
                    max_mag = snfitdf[f].min()
                    max_absmag = max_mag - dist_mod
                    m15 = snfitdf[np.around(snfitdf['t'], 1) == np.around(max_df['t'].values[0] + 15, 1)][f] \
                          - max_mag
                    rise = max_df['t'].values[0] - t0
                    if rise < 0:
                        print(max_df['t'].values[0], first_t, t0, rise)
                        raise ValueError('Rise less than 0, check')
                    if plot_lcs:
                        ax = axs.flatten()[f_ind]
                        ax.errorbar(sndf[f'{f}_t'], sndf[f], yerr=sndf[f'{f}_err'], fmt='x', color=colour_dict[f])
                        ax.plot(snfitdf['t'], snfitdf[f], label=f, c=colour_dict[f])
                        ax.fill_between(snfitdf['t'], snfitdf[f] - snfitdf[f'{f}_err'],
                                        snfitdf[f] + snfitdf[f'{f}_err'], alpha=0.3, color=colour_dict[f])
                        ax.scatter(max_df['t'], max_mag, color=colour_dict[f])
                        if not m15.empty:
                            ax.scatter(max_df['t'].values[0] + 15, max_mag + m15, color=colour_dict[f])
                        ax.legend()
                    if gen:
                        gen_prop_dict[f'{f}_max'].append(max_mag)
                        gen_prop_dict[f'{f}_Mmax'].append(max_absmag)
                        gen_prop_dict[f'{f}_rise'].append(rise)
                        if not m15.empty:
                            gen_prop_dict[f'{f}15'].append(m15.values[0])
                        else:
                            stop += 1
                            gen_prop_dict[f'{f}15'].append(np.nan)
                        if f == 'g':
                            max_colour = max_df.g - max_df.r
                            gen_prop_dict['g-r'].append(max_colour.values[0])
                        elif f == 'r':
                            max_colour = max_df.r - max_df.i
                            gen_prop_dict['r-i'].append(max_colour.values[0])
                        elif f == 'i':
                            max_colour = max_df.i - max_df.z
                            gen_prop_dict['i-z'].append(max_colour.values[0])
                        elif f == 'z':
                            gen_prop_dict['redshift'].append(redshift)
                    else:
                        real_prop_dict[f'{f}_max'].append(max_mag)
                        real_prop_dict[f'{f}_Mmax'].append(max_absmag)
                        real_prop_dict[f'{f}_rise'].append(rise)
                        if not m15.empty:
                            real_prop_dict[f'{f}15'].append(m15.values[0])
                        else:
                            stop += 1
                            real_prop_dict[f'{f}15'].append(np.nan)
                        if f == 'g':
                            max_colour = max_df.g - max_df.r
                            real_prop_dict['g-r'].append(max_colour.values[0])
                        elif f == 'r':
                            max_colour = max_df.r - max_df.i
                            real_prop_dict['r-i'].append(max_colour.values[0])
                        elif f == 'i':
                            max_colour = max_df.i - max_df.z
                            real_prop_dict['i-z'].append(max_colour.values[0])
                        elif f == 'z':
                            real_prop_dict['redshift'].append(redshift)
                if plot_lcs:
                    ax.invert_yaxis()
                    axs[1, 0].set_xlabel('Phase')
                    axs[1, 1].set_xlabel('Phase')
                    ax0 = fig.add_subplot(111, frameon=False)
                    ax0.tick_params(axis='both', which='both', left=False, right=False, bottom=False, top=False,
                                    labelbottom=False, labelleft=False, labeltop=False)
                    ax0.set_ylabel('Apparent magnitude', labelpad=40)
                    plt.subplots_adjust(hspace=0, wspace=0)
                    if gen:
                        plt.savefig(os.path.join(self.plot_root, 'Generated_Plots', str(epoch), f'{sname}.png'),
                                    bbox_inches='tight')
                    else:
                        plt.savefig(os.path.join(self.plot_root, 'Real_Plots', f'{sname}.png'), bbox_inches='tight')
                    plt.close('all')

        param_list = []
        for param in ['_max', '15', '_rise', 'colour']:
            if param == 'colour':
                param_list = param_list + ['g-r', 'r-i', 'i-z']
            else:
                param_list = param_list + [f'{f}{param}' for f in ['g', 'r', 'i', 'z']]
        if self.redshift:
            param_list.append('redshift')
        real_corner_data, gen_corner_data = [], []
        for ind, key in enumerate(param_list):
            real_corner_data.append(real_prop_dict[key])
            gen_corner_data.append(gen_prop_dict[key])
        real_corner_data, gen_corner_data = np.array(real_corner_data), np.array(gen_corner_data)
        real_corner_data = real_corner_data.T  # ((real_corner_data.shape[1], real_corner_data.shape[0]))
        gen_corner_data = gen_corner_data.T  # ((gen_corner_data.shape[1], gen_corner_data.shape[0]))
        final_real_corner_data, final_gen_corner_data = [], []  # For removing nans
        for sn in real_corner_data:
            if np.isnan(sn).sum() == 0:
                final_real_corner_data.append(sn)
        for sn in gen_corner_data:
            if np.isnan(sn).sum() == 0:
                final_gen_corner_data.append(sn)
        final_real_corner_data, final_gen_corner_data = np.array(final_real_corner_data), \
                                                        np.array(final_gen_corner_data)
        prdc = compute_prdc(real_features=final_real_corner_data, fake_features=final_gen_corner_data, nearest_k=5)
        print(prdc)
        
        all_data = np.concatenate((final_real_corner_data, final_gen_corner_data))
        scaler = StandardScaler()
        # all_data = scaler.fit_transform(all_data)
        embed_data = TSNE(learning_rate='auto').fit_transform(all_data)
        real_embed_data, gen_embed_data = embed_data[:final_real_corner_data.shape[0], :], \
                                          embed_data[final_real_corner_data.shape[0]:, :]
        plt.scatter(real_embed_data[:, 0], real_embed_data[:, 1], c='b', marker='o', label='Real')
        plt.scatter(gen_embed_data[:, 0], gen_embed_data[:, 1], c='r', marker='v', label='Generated')
        plt.legend()
        plt.savefig(os.path.join(self.plot_root, 'Summary_Plots', str(epoch), f'TSNE.{file_format}'),
                    bbox_inches='tight')
        plt.show()
        embed_data = umap.UMAP(n_neighbors=30).fit_transform(all_data)
        real_embed_data, gen_embed_data = embed_data[:final_real_corner_data.shape[0], :], \
                                          embed_data[final_real_corner_data.shape[0]:, :]
        plt.scatter(real_embed_data[:, 0], real_embed_data[:, 1], c='b', marker='o', label='Real')
        plt.scatter(gen_embed_data[:, 0], gen_embed_data[:, 1], c='r', marker='v', label='Generated')
        plt.legend()
        plt.savefig(os.path.join(self.plot_root, 'Summary_Plots', str(epoch), f'UMAP.{file_format}'),
                    bbox_inches='tight')
        plt.show()

        # figure = corner.corner(final_real_corner_data, color='r', labels=[label_dict[key] for key in param_list],
        #                       quantiles=(0.16, 0.84), levels=(1 - np.exp(-2),), labelpad=0.1, plot_contours=True)
        # corner.corner(final_gen_corner_data, fig=figure, color='b',
        #              quantiles=(0.16, 0.84), levels=((1 - np.exp(-2),),), plot_contours=True)
        # plt.savefig(os.path.join(self.plot_root, 'Summary_Plots', str(epoch), f'corner.{file_format}'),
        #           bbox_inches='tight')
        # plt.show()

        for param in ['_max', '_Mmax', '15', '_rise', 'colour', 'redshift']:
            if param == 'colour':
                iter_list = ['g-r', 'r-i', 'i-z']
            elif param == 'redshift':
                iter_list = ['redshift']
            else:
                iter_list = [f'{f}{param}' for f in ['g', 'r', 'i', 'z']]
            fig_width = 10 if len(iter_list) == 1 else 18
            fig, ax = plt.subplots(2, len(iter_list), figsize=(fig_width, 12), sharex='col', sharey='row')
            for ind, key in enumerate(iter_list):
                bin_step = bin_step_dict[key]
                all_vals = np.array(real_prop_dict[key] + gen_prop_dict[key])
                low = np.floor(np.nanmin(all_vals) * (1 / bin_step)) / (1 / bin_step)
                up = np.ceil(np.nanmax(all_vals) * (1 / bin_step)) / (1 / bin_step)
                bins = np.arange(low, up + bin_step, bin_step)
                # Plot histograms
                if len(iter_list) > 1:
                    ax1, ax2 = ax[0, ind], ax[1, ind]
                else:
                    ax1, ax2 = ax[0], ax[1]
                ax1.hist(real_prop_dict[key], bins=bins, density=True, histtype='step', color='r', ls='-',
                                label='Real')
                ax1.hist(gen_prop_dict[key], bins=bins, density=True, histtype='step', color='b', ls='--',
                                label='Generated')
                # Plot CDFs
                real_data = np.array(real_prop_dict[key])
                real_data = real_data[~np.isnan(real_data)]
                real_data = np.sort(real_data)
                gen_data = np.array(gen_prop_dict[key])
                gen_data = gen_data[~np.isnan(gen_data)]
                gen_data = np.sort(gen_data)
                ks = stats.ks_2samp(real_data, gen_data)
                ax2.plot(real_data, np.arange(1, real_data.shape[0] + 1) / real_data.shape[0], c='r', ls='-')
                ax2.plot(gen_data, np.arange(1, gen_data.shape[0] + 1) / gen_data.shape[0], c='b', ls='--')
                ax2.set_xlabel(label_dict[key])
                ax2.set_ylim([0, 1.19])
                if ind == 0:
                    ax1.legend()
                    ax1.set_ylabel('Frequency Density')
                    ax2.set_ylabel('Cumulative Frequency Density')
            plt.subplots_adjust(hspace=0, wspace=0)
            fig.align_ylabels()
            plt.savefig(os.path.join(self.plot_root, 'Summary_Plots', str(epoch), f'all{param}.{file_format}'),
                        bbox_inches='tight')
        plt.show()
        t_step = 1

        real_grts, real_rits, real_izts = [], [], []
        real_gr, real_ri, real_iz = [], [], []
        real_gr_err, real_ri_err, real_iz_err = [], [], []
        for t_low in np.arange(real_gp_df['t'].min(), real_gp_df['t'].max(), t_step):
            t_up = t_low + t_step
            # real_ts.append(t_low + 0.5 * t_step)
            tdf = real_gp_df[(real_gp_df['t'] > t_low) & (real_gp_df['t'] < t_up)]
            grdf = tdf[(tdf['t'] > tdf['first_g']) & (tdf['t'] > tdf['first_r'])]
            if grdf.shape[0] > 0:
                grdf['g-r'] = grdf.g - grdf.r
                real_grts.append(t_low + 0.5 * t_step)
                real_gr.append(grdf['g-r'].mean())
                real_gr_err.append(grdf['g-r'].std())
            ridf = tdf[(tdf['t'] > tdf['first_r']) & (tdf['t'] > tdf['first_i'])]
            if ridf.shape[0] > 0:
                ridf['r-i'] = ridf.r - ridf.i
                real_rits.append(t_low + 0.5 * t_step)
                real_ri.append(ridf['r-i'].mean())
                real_ri_err.append(ridf['r-i'].std())
            izdf = tdf[(tdf['t'] > tdf['first_i']) & (tdf['t'] > tdf['first_z'])]
            if izdf.shape[0] > 0:
                izdf['i-z'] = izdf.i - izdf.z
                real_izts.append(t_low + 0.5 * t_step)
                real_iz.append(izdf['i-z'].mean())
                real_iz_err.append(izdf['i-z'].std())
        real_grts, real_rits, real_izts = np.array(real_grts), np.array(real_rits), np.array(real_izts)
        real_gr, real_ri, real_iz = np.array(real_gr), np.array(real_ri), np.array(real_iz)
        real_gr_err, real_ri_err, real_iz_err = np.array(real_gr_err), np.array(real_ri_err), np.array(real_iz_err)

        gp_grts, gp_rits, gp_izts = [], [], []
        gp_gr, gp_ri, gp_iz = [], [], []
        gp_gr_err, gp_ri_err, gp_iz_err = [], [], []
        for t_low in np.arange(gen_gp_df['t'].min(), gen_gp_df['t'].max(), t_step):
            t_up = t_low + t_step
            # gp_ts.append(t_low + 0.5 * t_step)
            tdf = gen_gp_df[(gen_gp_df['t'] > t_low) & (gen_gp_df['t'] < t_up)]
            grdf = tdf[(tdf['t'] > tdf['first_g']) & (tdf['t'] > tdf['first_r'])]
            if grdf.shape[0] > 0:
                grdf['g-r'] = grdf.g - grdf.r
                gp_grts.append(t_low + 0.5 * t_step)
                gp_gr.append(grdf['g-r'].mean())
                gp_gr_err.append(grdf['g-r'].std())
            ridf = tdf[(tdf['t'] > tdf['first_r']) & (tdf['t'] > tdf['first_i'])]
            if ridf.shape[0] > 0:
                ridf['r-i'] = ridf.r - ridf.i
                gp_rits.append(t_low + 0.5 * t_step)
                gp_ri.append(ridf['r-i'].mean())
                gp_ri_err.append(ridf['r-i'].std())
            izdf = tdf[(tdf['t'] > tdf['first_i']) & (tdf['t'] > tdf['first_z'])]
            if izdf.shape[0] > 0:
                izdf['i-z'] = izdf.i - izdf.z
                gp_izts.append(t_low + 0.5 * t_step)
                gp_iz.append(izdf['i-z'].mean())
                gp_iz_err.append(izdf['i-z'].std())
        gp_gr, gp_ri, gp_iz = np.array(gp_gr), np.array(gp_ri), np.array(gp_iz)
        gp_gr_err, gp_ri_err, gp_iz_err = np.array(gp_gr_err), np.array(gp_ri_err), np.array(gp_iz_err)
        gp_grts, gp_rits, gp_izts = np.array(gp_grts), np.array(gp_rits), np.array(gp_izts)

        fig, ax = plt.subplots(3, figsize=(9, 16), sharex=True)
        ax[0].plot(real_grts, real_gr, c='r', ls='-', label='Real')
        ax[0].fill_between(real_grts, real_gr - real_gr_err, real_gr + real_gr_err, color='r', alpha=0.3)
        ax[0].plot(gp_grts, gp_gr, c='b', ls='--', label='Generated')
        ax[0].fill_between(gp_grts, gp_gr - gp_gr_err, gp_gr + gp_gr_err, color='b', alpha=0.3)
        ax[1].plot(real_rits, real_ri, c='r', ls='-', label='Real')
        ax[1].fill_between(real_rits, real_ri - real_ri_err, real_ri + real_ri_err, color='r', alpha=0.3)
        ax[1].plot(gp_rits, gp_ri, c='b', ls='--', label='Generated')
        ax[1].fill_between(gp_rits, gp_ri - gp_ri_err, gp_ri + gp_ri_err, color='b', alpha=0.3)
        ax[2].plot(real_izts, real_iz, c='r', ls='-', label='Real')
        ax[2].fill_between(real_izts, real_iz - real_iz_err, real_iz + real_iz_err, color='r', alpha=0.3)
        ax[2].plot(gp_izts, gp_iz, c='b', ls='--', label='Generated')
        ax[2].fill_between(gp_izts, gp_iz - gp_iz_err, gp_iz + gp_iz_err, color='b', alpha=0.3)
        ax[0].legend()
        ax[2].set_xlabel('Phase')
        ax[0].set_ylabel(f'$g-r$')
        ax[1].set_ylabel(f'$r-i$')
        ax[2].set_ylabel(f'$i-z$')
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.savefig(os.path.join(self.plot_root, 'Summary_Plots', str(epoch), f'colour_curves.{file_format}'),
                    bbox_inches='tight')
        plt.show()

        t_step = 5
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        fig2, axs2 = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        fig3, axs3 = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        fig4, axs4 = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        for f_ind, f in enumerate(['g', 'r', 'i', 'z']):
            ax, ax2, ax3, ax4 = axs.flatten()[f_ind], axs2.flatten()[f_ind], axs3.flatten()[f_ind], \
                                axs4.flatten()[f_ind]
            ts, mags, mag_errs, scale_mags, scale_mag_errs = [], [], [], [], []
            for t_low in np.arange(real_scaled_df[f'{f}_t'].min(), real_scaled_df[f'{f}_t'].max(), t_step):
                t_up = t_low + t_step
                tdf = real_df[(real_df[f'{f}_t'] > t_low) & (real_df[f'{f}_t'] < t_up)]
                scale_tdf = real_scaled_df[(real_scaled_df[f'{f}_t'] > t_low) & (real_scaled_df[f'{f}_t'] < t_up)]
                ts.append(tdf[f'{f}_t'].mean())
                mags.append(tdf[f].mean())
                mag_errs.append(tdf[f].std())
                scale_mags.append(scale_tdf[f].mean())
                scale_mag_errs.append(scale_tdf[f].std())
            ts, mags, mag_errs, scale_mags, scale_mag_errs = np.array(ts), np.array(mags), np.array(mag_errs), \
                                                             np.array(scale_mags), np.array(scale_mag_errs)
            ax.plot(ts, mags, c='r', ls='-', label='Real')
            ax.fill_between(ts, mags - mag_errs, mags + mag_errs, color='r', alpha=0.3)
            ax2.plot(ts, scale_mags, c='r', ls='-', label='Real')
            ax2.fill_between(ts, scale_mags - scale_mag_errs, scale_mags + scale_mag_errs, color='r', alpha=0.3)
            if self.redshift:
                ts, mags, mag_errs, scale_mags, scale_mag_errs = [], [], [], [], []
                for t_low in np.arange(real_scaled_df[f'{f}_tz'].min(), real_scaled_df[f'{f}_tz'].max(), t_step):
                    t_up = t_low + t_step
                    tdf = real_df[(real_df[f'{f}_tz'] > t_low) & (real_df[f'{f}_tz'] < t_up)]
                    scale_tdf = real_scaled_df[(real_scaled_df[f'{f}_tz'] > t_low) & (real_scaled_df[f'{f}_tz'] < t_up)]
                    ts.append(tdf[f'{f}_tz'].mean())
                    mags.append(tdf[f'{f}_abs'].mean())
                    mag_errs.append(tdf[f'{f}_abs'].std())
                    scale_mags.append(scale_tdf[f'{f}_abs'].mean())
                    scale_mag_errs.append(scale_tdf[f'{f}_abs'].std())
                ts, mags, mag_errs, scale_mags, scale_mag_errs = np.array(ts), np.array(mags), np.array(mag_errs), \
                                                                 np.array(scale_mags), np.array(scale_mag_errs)
                ax3.plot(ts, mags, c='r', ls='-', label='Real')
                ax3.fill_between(ts, mags - mag_errs, mags + mag_errs, color='r', alpha=0.3)
                ax4.plot(ts, scale_mags, c='r', ls='-', label='Real')
                ax4.fill_between(ts, scale_mags - scale_mag_errs, scale_mags + scale_mag_errs, color='r', alpha=0.3)
                # hatch='/', facecolor='none', edgecolor='r')
                # ax, ax2 = axs.flatten()[f_ind], axs2.flatten()[f_ind]
            ts, mags, mag_errs, scale_mags, scale_mag_errs = [], [], [], [], []
            for t_low in np.arange(gen_scaled_df[f'{f}_t'].min(), gen_scaled_df[f'{f}_t'].max(), t_step):
                t_up = t_low + t_step
                tdf = gen_df[(gen_df[f'{f}_t'] > t_low) & (gen_df[f'{f}_t'] < t_up)]
                scale_tdf = gen_scaled_df[(gen_scaled_df[f'{f}_t'] > t_low) & (gen_scaled_df[f'{f}_t'] < t_up)]
                ts.append(tdf[f'{f}_t'].mean())
                mags.append(tdf[f].mean())
                mag_errs.append(tdf[f].std())
                scale_mags.append(scale_tdf[f].mean())
                scale_mag_errs.append(scale_tdf[f].std())
            ts, mags, mag_errs, scale_mags, scale_mag_errs = np.array(ts), np.array(mags), np.array(mag_errs), \
                                                             np.array(scale_mags), np.array(scale_mag_errs)
            ax.plot(ts, mags, c='b', ls='--', label='Generated')
            ax.fill_between(ts, mags - mag_errs, mags + mag_errs, color='b', alpha=0.3)
            ax2.plot(ts, scale_mags, c='b', ls='--', label='Generated')
            ax2.fill_between(ts, scale_mags - scale_mag_errs, scale_mags + scale_mag_errs, color='b', alpha=0.3)
            # hatch='x', facecolor="none", edgecolor='b')
            if self.redshift:
                ts, mags, mag_errs, scale_mags, scale_mag_errs = [], [], [], [], []
                for t_low in np.arange(gen_scaled_df[f'{f}_tz'].min(), gen_scaled_df[f'{f}_tz'].max(), t_step):
                    t_up = t_low + t_step
                    tdf = gen_df[(gen_df[f'{f}_tz'] > t_low) & (gen_df[f'{f}_tz'] < t_up)]
                    scale_tdf = gen_scaled_df[(gen_scaled_df[f'{f}_tz'] > t_low) & (gen_scaled_df[f'{f}_tz'] < t_up)]
                    ts.append(tdf[f'{f}_tz'].mean())
                    mags.append(tdf[f'{f}_abs'].mean())
                    mag_errs.append(tdf[f'{f}_abs'].std())
                    scale_mags.append(scale_tdf[f'{f}_abs'].mean())
                    scale_mag_errs.append(scale_tdf[f'{f}_abs'].std())
                ts, mags, mag_errs, scale_mags, scale_mag_errs = np.array(ts), np.array(mags), np.array(mag_errs), \
                                                                 np.array(scale_mags), np.array(scale_mag_errs)
                ax3.plot(ts, mags, c='b', ls='--', label='Generated')
                ax3.fill_between(ts, mags - mag_errs, mags + mag_errs, color='b', alpha=0.3)
                ax4.plot(ts, scale_mags, c='b', ls='--', label='Generated')
                ax4.fill_between(ts, scale_mags - scale_mag_errs, scale_mags + scale_mag_errs, color='b', alpha=0.3)
                # hatch='x', facecolor="none", edgecolor='b')
            ax.annotate(f, (0.03, 0.9), xycoords='axes fraction')
            ax2.annotate(f, (0.03, 0.9), xycoords='axes fraction')
            if self.redshift:
                ax3.annotate(f, (0.03, 0.9), xycoords='axes fraction')
                ax4.annotate(f, (0.03, 0.9), xycoords='axes fraction')
        ax.invert_yaxis()
        ax2.invert_yaxis()
        axs[1, 0].legend(loc='lower right')
        axs[1, 0].set_xlabel('Phase')
        axs[1, 1].set_xlabel('Phase')
        ax0 = fig.add_subplot(111, frameon=False)
        ax0.tick_params(axis='both', which='both', left=False, right=False, bottom=False, top=False,
                        labelbottom=False, labelleft=False, labeltop=False)
        ax0.set_ylabel('Apparent magnitude', labelpad=40)
        axs2[1, 0].legend(loc='lower right')
        axs2[1, 0].set_xlabel('Observer-frame phase')
        axs2[1, 1].set_xlabel('Phase')
        ax0 = fig2.add_subplot(111, frameon=False)
        ax0.tick_params(axis='both', which='both', left=False, right=False, bottom=False, top=False,
                        labelbottom=False, labelleft=False, labeltop=False)
        ax0.set_ylabel('Magnitude shift', labelpad=40)
        if self.redshift:
            ax3.invert_yaxis()
            ax4.invert_yaxis()
            axs3[1, 0].legend(loc='lower right')
            axs3[1, 0].set_xlabel('Rest-frame phase')
            axs3[1, 1].set_xlabel('Phase')
            ax0 = fig3.add_subplot(111, frameon=False)
            ax0.tick_params(axis='both', which='both', left=False, right=False, bottom=False, top=False,
                            labelbottom=False, labelleft=False, labeltop=False)
            ax0.set_ylabel('Absolute magnitude', labelpad=40)
            axs4[1, 0].legend(loc='lower right')
            axs4[1, 0].set_xlabel('Rest-frame phase')
            axs4[1, 1].set_xlabel('Rest-frame phase')
            ax0 = fig4.add_subplot(111, frameon=False)
            ax0.tick_params(axis='both', which='both', left=False, right=False, bottom=False, top=False,
                            labelbottom=False, labelleft=False, labeltop=False)
            ax0.set_ylabel('Magnitude shift', labelpad=40)
        plt.figure(1)
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.savefig(os.path.join(self.plot_root, 'Summary_Plots', str(epoch), f'mean_lcs.{file_format}'),
                    bbox_inches='tight')
        plt.figure(2)
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.savefig(os.path.join(self.plot_root, 'Summary_Plots', str(epoch), f'mean_lcs_shift.{file_format}'),
                    bbox_inches='tight')
        if self.redshift:
            plt.figure(3)
            plt.subplots_adjust(hspace=0, wspace=0)
            plt.savefig(os.path.join(self.plot_root, 'Summary_Plots', str(epoch), f'mean_lcs_abs.{file_format}'),
                        bbox_inches='tight')
            plt.figure(4)
            plt.subplots_adjust(hspace=0, wspace=0)
            plt.savefig(os.path.join(self.plot_root, 'Summary_Plots', str(epoch), f'mean_lcs_abs_shift.{file_format}'),
                        bbox_inches='tight')
        plt.show()
        plt.close('all')

        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        for f_ind, f in enumerate(['g', 'r', 'i', 'z']):
            ax = axs.flatten()[f_ind]
            real_x, real, real_err_low, real_err_up, gen_x, gen, gen_err_low, gen_err_up = [], [], [], [], [], [], [], []
            for low in np.arange(18., 25., 0.5):
                up = low + 0.5
                real_x.append(np.mean([low, up]) + 0.05)
                gen_x.append(np.mean([low, up]) - 0.05)
                real_bin = real_df[(real_df[f] > low) & (real_df[f] < up)]
                if real_bin.shape[0] > 0:
                    real_low, real_mid, real_up = np.percentile(real_bin[f'{f}_err'].values, [16, 50, 84])
                    # real_low, real_up = real_mid - real_low, real_up - real_mid
                else:
                    real_low, real_mid, real_up = np.nan, np.nan, np.nan
                real.append(real_mid)
                real_err_low.append(real_low)
                real_err_up.append(real_up)
                gen_bin = gen_df[(gen_df[f] > low) & (gen_df[f] < up)]
                if gen_bin.shape[0] > 0:
                    gen_low, gen_mid, gen_up = np.percentile(gen_bin[f'{f}_err'].values, [16, 50, 84])
                    # gen_low, gen_up = gen_mid - gen_low, gen_up - gen_mid
                else:
                    gen_low, gen_mid, gen_up = np.nan, np.nan, np.nan
                gen_low, gen_up = gen_mid - gen_low, gen_up - gen_mid
                gen.append(gen_mid)
                gen_err_low.append(gen_low)
                gen_err_up.append(gen_up)
            real_err = np.stack([np.array(real_err_low), np.array(real_err_up)])
            gen_err = np.stack([np.array(gen_err_low), np.array(gen_err_up)])
            # ax.scatter(real_df[f], real_df[f'{f}_err'], marker='X', c='w', edgecolors='r', s=10, label='Real')
            # ax.scatter(gen_df[f], gen_df[f'{f}_err'], marker='o', c='w', edgecolors='b', s=10, label='Generated')
            ax.errorbar(real_x, real, yerr=real_err, fmt='X', mfc='w', mec='r', ecolor='r', capsize=2, ms=10, label='Real')
            ax.errorbar(gen_x, gen, yerr=gen_err, fmt='o', mfc='w', mec='b', ecolor='b', ms=10, capsize=2, label='Generated')
            low, up = ax.get_ylim()
            up = np.min([up, 0.9])
            ax.set_ylim([low, up])
            ax.annotate(f, (0.03, 0.9), xycoords='axes fraction')
        axs[0, 0].legend(bbox_to_anchor=(1.0, 1.3), loc='upper center', ncol=4)

        ax0 = fig.add_subplot(111, frameon=False)
        ax0.tick_params(axis='both', which='both', left=False, right=False, bottom=False, top=False,
                        labelbottom=False, labelleft=False, labeltop=False)
        ax0.set_ylabel('Magnitude Error', labelpad=65)
        axs[1, 0].set_xlabel('Apparent Magnitude')
        axs[1, 1].set_xlabel('Apparent Magnitude')
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.savefig(os.path.join(self.plot_root, 'Summary_Plots', str(epoch), f'mag_vs_error.{file_format}'),
                    bbox_inches='tight')
        plt.show()

        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        for f_ind, f in enumerate(['g', 'r', 'i', 'z']):
            ax = axs.flatten()[f_ind]
            real_diffs = np.diff(real_df[f'{f}_t'].values, axis=0)
            real_diffs = real_diffs[real_diffs > 0]
            gen_diffs = np.diff(gen_df[f'{f}_t'].values, axis=0)
            gen_diffs = gen_diffs[gen_diffs > 0]
            min, max = np.min(np.r_[real_diffs, gen_diffs]), np.max(np.r_[real_diffs, gen_diffs])
            day_step = 1
            bins = np.arange(min, max + day_step, day_step)
            ax.hist(real_diffs, bins=bins, density=True, histtype='step', color='r', ls='-', label='Real')
            ax.hist(gen_diffs, bins=bins, density=True, histtype='step', color='b', ls='--', label='Generated')
            ax.annotate(f, (0.03, 0.9), xycoords='axes fraction')
        axs[0, 0].legend()
        ax0 = fig.add_subplot(111, frameon=False)
        ax0.tick_params(axis='both', which='both', left=False, right=False, bottom=False, top=False,
                        labelbottom=False, labelleft=False, labeltop=False)
        ax0.set_ylabel('Frequency Density', labelpad=40)
        axs[1, 0].set_xlabel('Cadence')
        axs[1, 1].set_xlabel('Cadence')
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.savefig(os.path.join(self.plot_root, 'Summary_Plots', str(epoch), f'candence_hists.{file_format}'),
                    bbox_inches='tight')

        real_df['N'] = np.nan
        gen_df = gen_df.reset_index()
        for sn in real_df.sn.unique():
            sndf = real_df[real_df.sn == sn]
            N = sndf.shape[0]
            idx = real_df[real_df.sn == sn].index
            real_df.loc[idx, 'N'] = N
        for sn in gen_df.sn.unique():
            sndf = gen_df[gen_df.sn == sn]
            N = sndf.shape[0]
            idx = gen_df[gen_df.sn == sn].index
            gen_df.loc[idx, 'N'] = N
        Ns = np.sort(np.r_[real_df.N.unique(), gen_df.N.unique()])
        real_cadence_means = {f: [] for f in ['g', 'r', 'i', 'z']}
        real_cadence_errs = {f: [] for f in ['g', 'r', 'i', 'z']}
        gen_cadence_means = {f: [] for f in ['g', 'r', 'i', 'z']}
        gen_cadence_errs = {f: [] for f in ['g', 'r', 'i', 'z']}
        for N in Ns:
            rNdf = real_df[real_df.N == N]
            gNdf = gen_df[gen_df.N == N]
            for f_ind, f in enumerate(['g', 'r', 'i', 'z']):
                rNdiff = np.diff(rNdf[f'{f}_t'].values, axis=0)
                rNdiff = rNdiff[rNdiff > 0]
                real_cadence_means[f].append(np.mean(rNdiff))
                real_cadence_errs[f].append(np.std(rNdiff))
                gNdiff = np.diff(gNdf[f'{f}_t'].values, axis=0)
                gNdiff = gNdiff[gNdiff > 0]
                gen_cadence_means[f].append(np.mean(gNdiff))
                gen_cadence_errs[f].append(np.std(gNdiff))

        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        for f_ind, f in enumerate(['g', 'r', 'i', 'z']):
            ax = axs.flatten()[f_ind]
            ax.errorbar(np.array(Ns) - 0.1, real_cadence_means[f], yerr=real_cadence_errs[f], fmt='X', color='r',
                        label='Real')
            ax.errorbar(np.array(Ns) + 0.1, gen_cadence_means[f], yerr=gen_cadence_errs[f], fmt='o', color='b',
                        label='Generated')
        axs[0, 0].legend()
        ax0 = fig.add_subplot(111, frameon=False)
        ax0.tick_params(axis='both', which='both', left=False, right=False, bottom=False, top=False,
                        labelbottom=False, labelleft=False, labeltop=False)
        ax0.set_ylabel('Cadence', labelpad=40)
        axs[1, 0].set_xlabel('N')
        axs[1, 1].set_xlabel('N')
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.savefig(os.path.join(self.plot_root, 'Summary_Plots', str(epoch), f'candence_vs_N.{file_format}'),
                    bbox_inches='tight')
        plt.show()

    def lc_plot(self, col, row, scale=4, epoch=1000, timesteps=12, file_format='png'):
        if not os.path.exists(os.path.join(self.plot_root, 'Tile_Plots')):
            os.mkdir(os.path.join(self.plot_root, 'Tile_Plots'))
        if not os.path.exists(os.path.join(self.plot_root, 'Tile_Plots', str(epoch))):
            os.mkdir(os.path.join(self.plot_root, 'Tile_Plots', str(epoch)))
        n = row * col
        colour_dict = {'g': 'g', 'r': 'r', 'i': 'b', 'z': 'k'}
        self.wgan = keras.models.load_model(os.path.join(self.wgan_dir, f'{epoch}.tf'))
        fig, axs = plt.subplots(row, col, figsize=(scale * col, scale * row))
        for ind in range(n):
            ax = axs.flatten()[ind]
            sn = np.random.choice(self.train_df.sn.unique())
            sndf = self.train_df[self.train_df.sn == sn]
            sndf = sndf[['sn', 'g_t', 'r_t', 'i_t', 'z_t', 'g', 'r', 'i', 'z', 'g_err',
                         'r_err', 'i_err', 'z_err']]
            if timesteps is None:
                n_steps = np.min([sndf.shape[0], 18])
                # elif n_steps > 15:
                #     n_steps -= 5
            else:
                n_steps = timesteps

            noise = rand.normal(size=(1, self.latent_dims))
            noise = np.reshape(noise, (1, 1, self.latent_dims))
            noise = np.repeat(noise, n_steps, 1)
            gen_lcs = self.wgan.generator.predict(noise)
            X = gen_lcs[0, :, :]
            gen_sndf = pd.DataFrame(X, columns=sndf.columns[1:])
            # gen_sndf = sndf.copy()
            gen_sndf[['g_t', 'r_t', 'i_t', 'z_t']] = (gen_sndf[['g_t', 'r_t', 'i_t', 'z_t']] + 1) / 2
            gen_sndf[['g', 'r', 'i', 'z']] = (gen_sndf[['g', 'r', 'i', 'z']] + 1) / 2
            gen_sndf[['g_err', 'r_err', 'i_err', 'z_err']] = (gen_sndf[['g_err', 'r_err', 'i_err', 'z_err']] + 1) \
                                                             / 2
            gen_sndf[['g_err', 'r_err', 'i_err', 'z_err']] = gen_sndf[['g_err', 'r_err', 'i_err', 'z_err']] * (
                    self.scaling_factors[5] - self.scaling_factors[4]) + self.scaling_factors[4]
            gen_sndf[['g_t', 'r_t', 'i_t', 'z_t']] = gen_sndf[['g_t', 'r_t', 'i_t', 'z_t']] * \
                                                     (self.scaling_factors[1] - self.scaling_factors[0]) + \
                                                     self.scaling_factors[0]
            gen_sndf[['g', 'r', 'i', 'z']] = gen_sndf[['g', 'r', 'i', 'z']] * (
                    self.scaling_factors[3] - self.scaling_factors[2]) + self.scaling_factors[2]
            gen_sndf[['g_err', 'r_err', 'i_err', 'z_err']] = gen_sndf[['g_err', 'r_err', 'i_err', 'z_err']] * (
                    self.scaling_factors[3] - self.scaling_factors[2]) * -1
            t_max = gen_sndf[gen_sndf.g == gen_sndf.g.min()].g_t.values[0]
            gen_sndf[['g_t', 'r_t', 'i_t', 'z_t']] -= t_max
            for f_ind, f in enumerate(['g', 'r', 'i', 'z']):
                ax.errorbar(gen_sndf[f'{f}_t'], gen_sndf[f], yerr=gen_sndf[f'{f}_err'],
                            color=colour_dict[f], fmt='x', label=f)
            # if ind >= n - col:
            #     ax.set_xlabel('Phase')
            # if ind % col == 0:
            #     ax.set_ylabel('Apparent Magnitude')
            ax.invert_yaxis()
        ax0 = fig.add_subplot(111, frameon=False)
        ax0.tick_params(axis='both', which='both', left=False, right=False, bottom=False, top=False,
                        labelbottom=False, labelleft=False, labeltop=False)
        ax0.set_xlabel('Phase', labelpad=40)
        ax0.set_ylabel('Apparent Magnitude', labelpad=40)
        axs.flatten()[int(np.floor(col / 2))].legend(bbox_to_anchor=(0.5, 1.35), loc='upper center', ncol=4)
        plt.savefig(os.path.join(self.plot_root, 'Tile_Plots', str(epoch), f'{row}x{col}.{file_format}'),
                    bbox_inches='tight')
        plt.show()

    def prdc(self, epoch=2000, repeats=10, n=10):
        real_df, gen_df = None, None
        real_gp_df, gen_gp_df = None, None
        real_scaled_df, gen_scaled_df = None, None
        self.wgan = keras.models.load_model(os.path.join(self.wgan_dir, f'{epoch}.tf'))
        sn_props = ['g_max', 'r_max', 'i_max', 'z_max', 'g15', 'r15', 'i15', 'z15',
                    'g_rise', 'r_rise', 'i_rise', 'z_rise', 'g-r', 'r-i', 'i-z']
        gen_number = 1
        all_real_data, all_gen_data = None, None
        for _ in range(repeats):
            real_prop_dict = {key: [] for key in sn_props}
            gen_prop_dict = {key: [] for key in sn_props}
            for sn in tqdm(self.train_df.sn.unique(), total=len(self.train_df.sn.unique())):
                # if len(real_prop_dict['g15']) > 40:
                #    break
                sndf = self.train_df[self.train_df.sn == sn]
                # Fit for real SN first
                sndf = sndf[['sn', 'g_t', 'r_t', 'i_t', 'z_t', 'g', 'r', 'i', 'z', 'g_err',
                             'r_err', 'i_err', 'z_err']]
                timesteps = sndf.shape[0]
                full_sndf = sndf.copy()

                gen_tries, gen_count = 0, 0
                # for i in range(n):
                while gen_count < n:
                    if gen_tries > 20 and gen_count == 0:
                        break
                    noise = rand.normal(size=(1, self.latent_dims))
                    noise = np.reshape(noise, (1, 1, self.latent_dims))
                    noise = np.repeat(noise, timesteps, 1)
                    gen_lcs = self.wgan.generator.predict(noise)
                    X = gen_lcs[0, :, :]
                    days = X[:, 0:4] * (self.scaling_factors[1] - self.scaling_factors[0])
                    day_diffs = np.diff(days, axis=0)
                    gen_tries += 1
                    if np.min(day_diffs.flatten()) < 0.5:
                        continue
                    gen_count += 1

                    gen_sndf = pd.DataFrame(X, columns=sndf.columns[1:])
                    gen_sndf['sn'] = f'gen_{gen_number}'
                    gen_number += 1
                    full_sndf = pd.concat([full_sndf, gen_sndf])

                for sname in full_sndf.sn.unique():
                    sndf = full_sndf[full_sndf.sn == sname]
                    gen = True if 'gen' in str(sname) else False

                    if gen:
                        sndf[['g_t', 'r_t', 'i_t', 'z_t']] = (sndf[['g_t', 'r_t', 'i_t', 'z_t']] + 1) / 2
                        sndf[['g', 'r', 'i', 'z']] = (sndf[['g', 'r', 'i', 'z']] + 1) / 2
                        sndf[['g_err', 'r_err', 'i_err', 'z_err']] = (sndf[
                                                                          ['g_err', 'r_err', 'i_err', 'z_err']] + 1) \
                                                                     / 2
                    sndf[['g_err', 'r_err', 'i_err', 'z_err']] = sndf[['g_err', 'r_err', 'i_err', 'z_err']] * (
                            self.scaling_factors[5] - self.scaling_factors[4]) + self.scaling_factors[4]

                    all_t = np.r_[sndf.g_t.values, sndf.r_t.values, sndf.i_t.values, sndf.z_t.values]
                    day_step = 0.1
                    t_step = day_step / (self.scaling_factors[1] - self.scaling_factors[0])
                    fit_t = np.arange(all_t.min(), all_t.max(), t_step)
                    snfitdf = pd.DataFrame(fit_t, columns=['t'])

                    for f in ['g', 'r', 'i', 'z']:
                        gp = george.GP(Matern32Kernel(1))
                        t, y, y_err = sndf[f'{f}_t'], sndf[f], sndf[f'{f}_err']
                        gp.compute(t, y_err)
                        p0 = gp.kernel.get_parameter_vector()[0]

                        def ll(p):
                            gp.set_parameter_vector(p)
                            return -gp.log_likelihood(y, quiet=True)

                        def grad_ll(p):
                            gp.set_parameter_vector(p)
                            return -gp.grad_log_likelihood(y, quiet=True)

                        results = spopt.minimize(ll, p0, jac=grad_ll)
                        mu, cov = gp.predict(y, fit_t)
                        std = np.sqrt(np.diag(cov))
                        snfitdf[f] = mu
                        snfitdf[f'{f}_err'] = std

                    sndf[['g_t', 'r_t', 'i_t', 'z_t']] = sndf[['g_t', 'r_t', 'i_t', 'z_t']] * \
                                                         (self.scaling_factors[1] - self.scaling_factors[0]) + \
                                                         self.scaling_factors[0]
                    sndf[['g', 'r', 'i', 'z']] = sndf[['g', 'r', 'i', 'z']] * (
                            self.scaling_factors[3] - self.scaling_factors[2]) + self.scaling_factors[2]
                    sndf[['g_err', 'r_err', 'i_err', 'z_err']] = sndf[['g_err', 'r_err', 'i_err', 'z_err']] * (
                            self.scaling_factors[3] - self.scaling_factors[2]) * -1
                    snfitdf['t'] = snfitdf['t'] * (self.scaling_factors[1] - self.scaling_factors[0]) + \
                                   self.scaling_factors[0]
                    snfitdf[['g', 'r', 'i', 'z']] = snfitdf[['g', 'r', 'i', 'z']] * (
                            self.scaling_factors[3] - self.scaling_factors[2]) + self.scaling_factors[2]
                    snfitdf[['g_err', 'r_err', 'i_err', 'z_err']] = snfitdf[['g_err', 'r_err', 'i_err', 'z_err']] * (
                            self.scaling_factors[3] - self.scaling_factors[2]) * -1

                    maxes = [snfitdf[snfitdf[f] == snfitdf[f].min()]['t'].values[0] for f in ['g', 'r', 'i', 'z']]
                    tmax = np.min(maxes)
                    first_max = [np.argmin(snfitdf[f].values) for f in ['g', 'r', 'i', 'z']]
                    if 0 in first_max:
                        continue
                    # tmax = snfitdf[snfitdf.g == snfitdf.g.min()]['t'].values[0]  # Just for g-band

                    first_t = np.min(sndf[['g_t', 'r_t', 'i_t', 'z_t']].values.flatten())
                    t_shift = tmax
                    t0 = first_t - t_shift
                    sndf[['g_t', 'r_t', 'i_t', 'z_t']] -= t_shift
                    snfitdf['t'] -= t_shift
                    snfitdf['first_g'] = sndf[['g_t']].min().values[0]
                    snfitdf['first_r'] = sndf[['r_t']].min().values[0]
                    snfitdf['first_i'] = sndf[['i_t']].min().values[0]
                    snfitdf['first_z'] = sndf[['z_t']].min().values[0]
                    if -snfitdf['t'].min() > 3 * snfitdf['t'].max():  # Remove objects which don't peak
                        continue
                    if gen:
                        if gen_df is None:
                            gen_df = sndf.copy()
                        else:
                            gen_df = pd.concat([gen_df, sndf])
                    else:
                        if real_df is None:
                            real_df = sndf.copy()
                        else:
                            real_df = pd.concat([real_df, sndf])
                    if gen:
                        if gen_gp_df is None:
                            gen_gp_df = snfitdf.copy()
                        else:
                            gen_gp_df = pd.concat([gen_gp_df, snfitdf])
                    else:
                        if real_gp_df is None:
                            real_gp_df = snfitdf.copy()
                        else:
                            real_gp_df = pd.concat([real_gp_df, snfitdf])

                    # Scaled light curves
                    scaled_sndf = sndf.copy()
                    for f_ind, f in enumerate(['g', 'r', 'i', 'z']):
                        fmax = scaled_sndf[f].min()
                        scaled_sndf[f] -= fmax
                    if gen:
                        if gen_scaled_df is None:
                            gen_scaled_df = scaled_sndf.copy()
                        else:
                            gen_scaled_df = pd.concat([gen_scaled_df, scaled_sndf])
                    else:
                        if real_scaled_df is None:
                            real_scaled_df = scaled_sndf.copy()
                        else:
                            real_scaled_df = pd.concat([real_scaled_df, scaled_sndf])

                    # Get SN properties
                    stop = 0

                    for f_ind, f in enumerate(['g', 'r', 'i', 'z']):
                        max_df = snfitdf[snfitdf[f] == snfitdf[f].min()]
                        if max_df.shape[0] > 1:
                            max_df = max_df.iloc[[0], :]
                        max_mag = snfitdf[f].min()
                        m15 = snfitdf[np.around(snfitdf['t'], 1) == np.around(max_df['t'].values[0] + 15, 1)][f] \
                              - max_mag
                        rise = max_df['t'].values[0] - t0
                        if rise < 0:
                            print(max_df['t'].values[0], first_t, t0, rise)
                            raise ValueError('Rise less than 0, check')
                        if gen:
                            gen_prop_dict[f'{f}_max'].append(snfitdf[f].min())
                            gen_prop_dict[f'{f}_rise'].append(rise)
                            if not m15.empty:
                                gen_prop_dict[f'{f}15'].append(m15.values[0])
                            else:
                                stop += 1
                                gen_prop_dict[f'{f}15'].append(np.nan)
                            if f == 'g':
                                max_colour = max_df.g - max_df.r
                                gen_prop_dict['g-r'].append(max_colour.values[0])
                            elif f == 'r':
                                max_colour = max_df.r - max_df.i
                                gen_prop_dict['r-i'].append(max_colour.values[0])
                            elif f == 'i':
                                max_colour = max_df.i - max_df.z
                                gen_prop_dict['i-z'].append(max_colour.values[0])
                        else:
                            real_prop_dict[f'{f}_max'].append(snfitdf[f].min())
                            real_prop_dict[f'{f}_rise'].append(rise)
                            if not m15.empty:
                                real_prop_dict[f'{f}15'].append(m15.values[0])
                            else:
                                stop += 1
                                real_prop_dict[f'{f}15'].append(np.nan)
                            if f == 'g':
                                max_colour = max_df.g - max_df.r
                                real_prop_dict['g-r'].append(max_colour.values[0])
                            elif f == 'r':
                                max_colour = max_df.r - max_df.i
                                real_prop_dict['r-i'].append(max_colour.values[0])
                            elif f == 'i':
                                max_colour = max_df.i - max_df.z
                                real_prop_dict['i-z'].append(max_colour.values[0])
            param_list = []
            for param in ['_max', '15', '_rise', 'colour']:
                if param == 'colour':
                    param_list = param_list + ['g-r', 'r-i', 'i-z']
                else:
                    param_list = param_list + [f'{f}{param}' for f in ['g', 'r', 'i', 'z']]
            real_corner_data, gen_corner_data = [], []
            for ind, key in enumerate(param_list):
                real_corner_data.append(real_prop_dict[key])
                gen_corner_data.append(gen_prop_dict[key])
            real_corner_data, gen_corner_data = np.array(real_corner_data), np.array(gen_corner_data)
            real_corner_data = real_corner_data.T  # ((real_corner_data.shape[1], real_corner_data.shape[0]))
            gen_corner_data = gen_corner_data.T  # ((gen_corner_data.shape[1], gen_corner_data.shape[0]))
            final_real_corner_data, final_gen_corner_data = [], []  # For removing nans
            for sn in real_corner_data:
                if np.isnan(sn).sum() == 0:
                    final_real_corner_data.append(sn)
            for sn in gen_corner_data:
                if np.isnan(sn).sum() == 0:
                    final_gen_corner_data.append(sn)
            final_real_corner_data, final_gen_corner_data = np.array(final_real_corner_data), \
                                                            np.array(final_gen_corner_data)
            if all_real_data is None:
                all_real_data = final_real_corner_data.copy()
            if all_gen_data is None:
                all_gen_data = final_gen_corner_data.copy()
            else:
                all_gen_data = np.concatenate([all_gen_data, final_gen_corner_data])
            prdc = compute_prdc(real_features=final_real_corner_data, fake_features=final_gen_corner_data, nearest_k=5)
            if _ == 0:
                all_prdc = {key: [val] for key, val in prdc.items()}
            else:
                for key, val in prdc.items():
                    all_prdc[key].append(val)
        all_prdc = {key: (np.mean(val), np.std(val)) for key, val in all_prdc.items()}
        print(all_prdc)
        combined_prdc = compute_prdc(real_features=all_real_data, fake_features=all_gen_data, nearest_k=5)
        print(combined_prdc)

    def prdc_explore(self, start=1000, stop=10000, step=500, n=1):
        sn_props = ['g_max', 'r_max', 'i_max', 'z_max', 'g15', 'r15', 'i15', 'z15',
                    'g_rise', 'r_rise', 'i_rise', 'z_rise', 'g-r', 'r-i', 'i-z']
        gen_number = 1
        all_real_data, all_gen_data = None, None
        epoch_range = np.arange(start, stop, step)
        d, c = [], []
        for epoch in epoch_range:
            self.wgan = keras.models.load_model(os.path.join(self.wgan_dir, f'{epoch}.tf'))
            real_prop_dict = {key: [] for key in sn_props}
            gen_prop_dict = {key: [] for key in sn_props}
            real_df, gen_df = None, None
            real_gp_df, gen_gp_df = None, None
            real_scaled_df, gen_scaled_df = None, None
            for sn in tqdm(self.train_df.sn.unique(), total=len(self.train_df.sn.unique())):
                # if len(real_prop_dict['g15']) > 20:
                #    break
                sndf = self.train_df[self.train_df.sn == sn]
                # Fit for real SN first
                sndf = sndf[['sn', 'g_t', 'r_t', 'i_t', 'z_t', 'g', 'r', 'i', 'z', 'g_err',
                             'r_err', 'i_err', 'z_err']]

                timesteps = sndf.shape[0]
                full_sndf = sndf.copy()

                gen_tries, gen_count = 0, 0
                # for i in range(n):
                while gen_count < n:
                    if gen_tries > 20 and gen_count == 0:
                        break
                    noise = rand.normal(size=(1, self.latent_dims))
                    noise = np.reshape(noise, (1, 1, self.latent_dims))
                    noise = np.repeat(noise, timesteps, 1)
                    gen_lcs = self.wgan.generator.predict(noise)
                    X = gen_lcs[0, :, :]
                    days = X[:, 0:4] * (self.scaling_factors[1] - self.scaling_factors[0])
                    day_diffs = np.diff(days, axis=0)
                    gen_tries += 1
                    if np.min(day_diffs.flatten()) < 0.5:
                        continue
                    gen_count += 1

                    gen_sndf = pd.DataFrame(X, columns=sndf.columns[1:])
                    gen_sndf['sn'] = f'gen_{gen_number}'
                    gen_number += 1
                    full_sndf = pd.concat([full_sndf, gen_sndf])

                for sname in full_sndf.sn.unique():
                    sndf = full_sndf[full_sndf.sn == sname]
                    gen = True if 'gen' in str(sname) else False

                    if gen:
                        sndf[['g_t', 'r_t', 'i_t', 'z_t']] = (sndf[['g_t', 'r_t', 'i_t', 'z_t']] + 1) / 2
                        sndf[['g', 'r', 'i', 'z']] = (sndf[['g', 'r', 'i', 'z']] + 1) / 2
                        sndf[['g_err', 'r_err', 'i_err', 'z_err']] = (sndf[
                                                                          ['g_err', 'r_err', 'i_err', 'z_err']] + 1) \
                                                                     / 2
                    sndf[['g_err', 'r_err', 'i_err', 'z_err']] = sndf[['g_err', 'r_err', 'i_err', 'z_err']] * (
                            self.scaling_factors[5] - self.scaling_factors[4]) + self.scaling_factors[4]
                    all_t = np.r_[sndf.g_t.values, sndf.r_t.values, sndf.i_t.values, sndf.z_t.values]
                    day_step = 0.1
                    t_step = day_step / (self.scaling_factors[1] - self.scaling_factors[0])
                    fit_t = np.arange(all_t.min(), all_t.max(), t_step)
                    snfitdf = pd.DataFrame(fit_t, columns=['t'])

                    for f in ['g', 'r', 'i', 'z']:
                        gp = george.GP(Matern32Kernel(1))
                        t, y, y_err = sndf[f'{f}_t'], sndf[f], sndf[f'{f}_err']
                        gp.compute(t, y_err)
                        p0 = gp.kernel.get_parameter_vector()[0]

                        def ll(p):
                            gp.set_parameter_vector(p)
                            return -gp.log_likelihood(y, quiet=True)

                        def grad_ll(p):
                            gp.set_parameter_vector(p)
                            return -gp.grad_log_likelihood(y, quiet=True)

                        results = spopt.minimize(ll, p0, jac=grad_ll)
                        mu, cov = gp.predict(y, fit_t)
                        std = np.sqrt(np.diag(cov))
                        snfitdf[f] = mu
                        snfitdf[f'{f}_err'] = std

                    sndf[['g_t', 'r_t', 'i_t', 'z_t']] = sndf[['g_t', 'r_t', 'i_t', 'z_t']] * \
                                                         (self.scaling_factors[1] - self.scaling_factors[0]) + \
                                                         self.scaling_factors[0]
                    sndf[['g', 'r', 'i', 'z']] = sndf[['g', 'r', 'i', 'z']] * (
                            self.scaling_factors[3] - self.scaling_factors[2]) + self.scaling_factors[2]
                    sndf[['g_err', 'r_err', 'i_err', 'z_err']] = sndf[['g_err', 'r_err', 'i_err', 'z_err']] * (
                            self.scaling_factors[3] - self.scaling_factors[2]) * -1
                    snfitdf['t'] = snfitdf['t'] * (self.scaling_factors[1] - self.scaling_factors[0]) + \
                                   self.scaling_factors[0]
                    snfitdf[['g', 'r', 'i', 'z']] = snfitdf[['g', 'r', 'i', 'z']] * (
                            self.scaling_factors[3] - self.scaling_factors[2]) + self.scaling_factors[2]
                    snfitdf[['g_err', 'r_err', 'i_err', 'z_err']] = snfitdf[['g_err', 'r_err', 'i_err', 'z_err']] * (
                            self.scaling_factors[3] - self.scaling_factors[2]) * -1

                    maxes = [snfitdf[snfitdf[f] == snfitdf[f].min()]['t'].values[0] for f in ['g', 'r', 'i', 'z']]
                    tmax = np.min(maxes)
                    first_max = [np.argmin(snfitdf[f].values) for f in ['g', 'r', 'i', 'z']]
                    if 0 in first_max:
                        continue
                    # tmax = snfitdf[snfitdf.g == snfitdf.g.min()]['t'].values[0]  # Just for g-band

                    first_t = np.min(sndf[['g_t', 'r_t', 'i_t', 'z_t']].values.flatten())
                    t_shift = tmax
                    t0 = first_t - t_shift
                    sndf[['g_t', 'r_t', 'i_t', 'z_t']] -= t_shift
                    snfitdf['t'] -= t_shift
                    snfitdf['first_g'] = sndf[['g_t']].min().values[0]
                    snfitdf['first_r'] = sndf[['r_t']].min().values[0]
                    snfitdf['first_i'] = sndf[['i_t']].min().values[0]
                    snfitdf['first_z'] = sndf[['z_t']].min().values[0]
                    if -snfitdf['t'].min() > 3 * snfitdf['t'].max():  # Remove objects which don't peak
                        continue
                    if gen:
                        if gen_df is None:
                            gen_df = sndf.copy()
                        else:
                            gen_df = pd.concat([gen_df, sndf])
                    else:
                        if real_df is None:
                            real_df = sndf.copy()
                        else:
                            real_df = pd.concat([real_df, sndf])
                    if gen:
                        if gen_gp_df is None:
                            gen_gp_df = snfitdf.copy()
                        else:
                            gen_gp_df = pd.concat([gen_gp_df, snfitdf])
                    else:
                        if real_gp_df is None:
                            real_gp_df = snfitdf.copy()
                        else:
                            real_gp_df = pd.concat([real_gp_df, snfitdf])

                    # Get SN properties
                    stop = 0

                    for f_ind, f in enumerate(['g', 'r', 'i', 'z']):
                        max_df = snfitdf[snfitdf[f] == snfitdf[f].min()]
                        if max_df.shape[0] > 1:
                            max_df = max_df.iloc[[0], :]
                        max_mag = snfitdf[f].min()
                        m15 = snfitdf[np.around(snfitdf['t'], 1) == np.around(max_df['t'].values[0] + 15, 1)][f] \
                              - max_mag
                        rise = max_df['t'].values[0] - t0
                        if rise < 0:
                            print(max_df['t'].values[0], first_t, t0, rise)
                            raise ValueError('Rise less than 0, check')
                        if gen:
                            gen_prop_dict[f'{f}_max'].append(snfitdf[f].min())
                            gen_prop_dict[f'{f}_rise'].append(rise)
                            if not m15.empty:
                                gen_prop_dict[f'{f}15'].append(m15.values[0])
                            else:
                                stop += 1
                                gen_prop_dict[f'{f}15'].append(np.nan)
                            if f == 'g':
                                max_colour = max_df.g - max_df.r
                                gen_prop_dict['g-r'].append(max_colour.values[0])
                            elif f == 'r':
                                max_colour = max_df.r - max_df.i
                                gen_prop_dict['r-i'].append(max_colour.values[0])
                            elif f == 'i':
                                max_colour = max_df.i - max_df.z
                                gen_prop_dict['i-z'].append(max_colour.values[0])
                        else:
                            real_prop_dict[f'{f}_max'].append(snfitdf[f].min())
                            real_prop_dict[f'{f}_rise'].append(rise)
                            if not m15.empty:
                                real_prop_dict[f'{f}15'].append(m15.values[0])
                            else:
                                stop += 1
                                real_prop_dict[f'{f}15'].append(np.nan)
                            if f == 'g':
                                max_colour = max_df.g - max_df.r
                                real_prop_dict['g-r'].append(max_colour.values[0])
                            elif f == 'r':
                                max_colour = max_df.r - max_df.i
                                real_prop_dict['r-i'].append(max_colour.values[0])
                            elif f == 'i':
                                max_colour = max_df.i - max_df.z
                                real_prop_dict['i-z'].append(max_colour.values[0])
            param_list = []
            for param in ['_max', '15', '_rise', 'colour']:
                if param == 'colour':
                    param_list = param_list + ['g-r', 'r-i', 'i-z']
                else:
                    param_list = param_list + [f'{f}{param}' for f in ['g', 'r', 'i', 'z']]
            real_corner_data, gen_corner_data = [], []
            for ind, key in enumerate(param_list):
                real_corner_data.append(real_prop_dict[key])
                gen_corner_data.append(gen_prop_dict[key])
            real_corner_data, gen_corner_data = np.array(real_corner_data), np.array(gen_corner_data)
            real_corner_data = real_corner_data.T  # ((real_corner_data.shape[1], real_corner_data.shape[0]))
            gen_corner_data = gen_corner_data.T  # ((gen_corner_data.shape[1], gen_corner_data.shape[0]))
            final_real_corner_data, final_gen_corner_data = [], []  # For removing nans
            for sn in real_corner_data:
                if np.isnan(sn).sum() == 0:
                    final_real_corner_data.append(sn)
            for sn in gen_corner_data:
                if np.isnan(sn).sum() == 0:
                    final_gen_corner_data.append(sn)
            final_real_corner_data, final_gen_corner_data = np.array(final_real_corner_data), \
                                                            np.array(final_gen_corner_data)
            prdc = compute_prdc(real_features=final_real_corner_data, fake_features=final_gen_corner_data, nearest_k=5)
            d.append(prdc['density'])
            c.append(prdc['coverage'])
        sort_epochs = epoch_range[np.argsort(d)[::-1]]
        print(sort_epochs[:10])
        plt.plot(epoch_range, d, label='Density')
        plt.plot(epoch_range, c, label='Coverage')
        plt.legend()
        plt.show()

    def timing(self, N=10000, epoch=1000, timesteps=15):
        self.wgan = keras.models.load_model(os.path.join(self.wgan_dir, f'{epoch}.tf'))
        print('Starting generation...')
        t1 = time.time()
        noise = rand.normal(size=(N, self.latent_dims))
        noise = np.reshape(noise, (N, 1, self.latent_dims))
        noise = np.repeat(noise, timesteps, 1)
        gen_lcs = self.wgan.generator.predict(noise)
        t2 = time.time()
        time_elapsed = t2 - t1
        print(f'Time to generate {N} light curves with {timesteps} time steps: {time_elapsed} seconds')

    def prepare_snn(self, N=10000, epoch=1000, timesteps=None):
        if not os.path.exists(os.path.join(self.lc_root, str(epoch))):
            os.mkdir(os.path.join(self.lc_root, str(epoch)))
        self.wgan = keras.models.load_model(os.path.join(self.wgan_dir, f'{epoch}.tf'))
        zp_ab = {'g': 20.802, 'r': 21.436, 'i': 21.866, 'z': 22.214}
        all_head_df, all_lc_df = None, None
        print('Generating light curves...')
        for ind in tqdm(range(N)):
            sn = np.random.choice(self.train_df.sn.unique())
            sndf = self.train_df[self.train_df.sn == sn]
            sndf = sndf[['sn', 'g_t', 'r_t', 'i_t', 'z_t', 'g', 'r', 'i', 'z', 'g_err',
                         'r_err', 'i_err', 'z_err']]
            if timesteps is None:
                n_steps = sndf.shape[0]
            else:
                n_steps = timesteps
            noise = rand.normal(size=(1, self.latent_dims))
            noise = np.reshape(noise, (1, 1, self.latent_dims))
            noise = np.repeat(noise, n_steps, 1)
            gen_lcs = self.wgan.generator.predict(noise)
            X = gen_lcs[0, :, :]
            gen_sndf = pd.DataFrame(X, columns=sndf.columns[1:])
            # gen_sndf = sndf.copy()
            gen_sndf[['g_t', 'r_t', 'i_t', 'z_t']] = (gen_sndf[['g_t', 'r_t', 'i_t', 'z_t']] + 1) / 2
            gen_sndf[['g', 'r', 'i', 'z']] = (gen_sndf[['g', 'r', 'i', 'z']] + 1) / 2
            gen_sndf[['g_err', 'r_err', 'i_err', 'z_err']] = (gen_sndf[['g_err', 'r_err', 'i_err', 'z_err']] + 1) \
                                                             / 2
            gen_sndf[['g_err', 'r_err', 'i_err', 'z_err']] = gen_sndf[['g_err', 'r_err', 'i_err', 'z_err']] * (
                    self.scaling_factors[5] - self.scaling_factors[4]) + self.scaling_factors[4]
            gen_sndf[['g_t', 'r_t', 'i_t', 'z_t']] = gen_sndf[['g_t', 'r_t', 'i_t', 'z_t']] * \
                                                     (self.scaling_factors[1] - self.scaling_factors[0]) + \
                                                     self.scaling_factors[0]
            gen_sndf[['g', 'r', 'i', 'z']] = gen_sndf[['g', 'r', 'i', 'z']] * (
                    self.scaling_factors[3] - self.scaling_factors[2]) + self.scaling_factors[2]
            gen_sndf[['g_err', 'r_err', 'i_err', 'z_err']] = np.abs(gen_sndf[['g_err', 'r_err', 'i_err', 'z_err']] * (
                    self.scaling_factors[3] - self.scaling_factors[2]))
            t_max = gen_sndf[gen_sndf.g == gen_sndf.g.min()].g_t.values[0]
            gen_sndf[['g_t', 'r_t', 'i_t', 'z_t']] -= t_max
            for f in ['g', 'r', 'i', 'z']:
                gen_sndf[f] = np.power(10, -(gen_sndf[f] + zp_ab[f]) / 2.5)
                gen_sndf[f'{f}_err'] = (np.log(10) / 2.5) * gen_sndf[f] * gen_sndf[f'{f}_err']
            lc_data = []
            for i, row in gen_sndf.iterrows():
                for f in ['g', 'r', 'i', 'z']:
                    lc_data.append([ind, row[f'{f}_t'], row[f], row[f'{f}_err'], f])
            head_data = [ind, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 120]  # 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
            lc_df = pd.DataFrame(lc_data, columns=['SNID', 'MJD', 'FLUXCAL', 'FLUXCALERR', 'FLT'], dtype='str')
            head_df = pd.DataFrame([head_data], columns=['SNID', 'PEAKMJD', 'HOSTGAL_PHOTOZ', 'HOSTGAL_PHOTOZ_ERR',
                                                         'HOSTGAL_SPECZ', 'HOSTGAL_SPECZ_ERR', 'SIM_REDSHIFT_CMB',
                                                         'SIM_PEAKMAG_z', 'SIM_PEAKMAG_g', 'SIM_PEAKMAG_r',
                                                         'SIM_PEAKMAG_i', 'SNTYPE'], dtype='str')
            if all_lc_df is None:
                all_head_df, all_lc_df = head_df.copy(), lc_df.copy()
            else:
                all_head_df, all_lc_df = pd.concat([all_head_df, head_df]), pd.concat([all_lc_df, lc_df])
        all_head_df.to_csv(os.path.join(self.lc_root, str(epoch), 'HEAD.csv'), index=False)
        print(all_head_df)
        all_lc_df.to_csv(os.path.join(self.lc_root, str(epoch), 'PHOT.csv'), index=False)
