import os
import numpy as np
import pandas as pd
import scipy.optimize as spopt
import pickle
import numpy.random as rand
import matplotlib as mpl
from matplotlib import rc
# rc('font', **{'family': 'serif', 'serif': ['cmr10']})
# mpl.rcParams['axes.unicode_minus'] = False
# mpl.rcParams['mathtext.fontset'] = 'cm'
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

# tf.config.run_functions_eagerly(True)

plt.rcParams.update({'font.size': 22})
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
        norm = tf.sqrt(tf.reduce_sum(1e-16 + tf.square(grads), axis=[1, 2]))
        # print(norm.numpy())
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
    def __init__(self, experiment='general', latent_dims=100, clr=0.0005, glr=0.0005, device='gpu:0', GP=True,
                 z_lim=None, batch_norm=False, mode='template', g_dropout=0.5, c_dropout=0.5,
                 gen_units=100, crit_units=100, sn_type='II', ds=1, inc_colour=False, n_critic=1):
        """
        :param latent_dims: int, number of latent dimensions to draw random seed for generator from
        :param clr: float, initial learning rate to use for critic
        :param glr: float, initial learning rate to use for generator
        :param device: string, device to use for model training
        :param GP: Boolean, specifies whether to Gaussian Process interpolate training light curves
        :param z_lim: float, upper redshift limit for sample
        :param batch_norm: Boolean, specifies whether to apply batch normalisation to critic
        :param mode: string, must be one of 'template' or 'observed'
        :param g_dropout: float, dropout fraction to use in generator model
        :param c_dropout: float, dropout fraction to use in critic model
        :param gen_units: int, number of GRU units in each layer of generator model
        :param crit_units: int, number of GRU units in each layer of generator model
        :param sn_type: string, SN class to look at
        :param ds: int, data structure for training. 1 for [band_time, band_mag, band_mag_err, etc.],
                2 for [band_time, band_time, ..., band_mag, band_mag, ..., band_mag_err, band_mag_err, ...]
        :param inc_colour: Boolean, whether to include colours in training process
        """
        self.experiment = experiment
        self.latent_dims = latent_dims
        self.clr = clr
        self.glr = glr
        self.device = device
        self.GP = GP
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
        self.z_lim = z_lim
        self.batch_norm = batch_norm
        self.mode = mode
        self.ds = ds
        self.inc_colour = inc_colour
        if self.mode.lower() not in ['template', 'observed']:
            raise ValueError('mode must be one of template and observed')
        self.g_dropout = g_dropout
        self.c_dropout = c_dropout
        self.gen_units = gen_units
        self.crit_units = crit_units
        # WGAN Paper guidance-----------------------------
        self.n_critic = n_critic
        self.c_optimizer = opt.Adam(lr=self.clr, beta_1=0.5, beta_2=0.9)
        self.g_optimizer = opt.Adam(lr=self.glr, beta_1=0.5, beta_2=0.9)
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
        if self.mode == 'observed':
            self.n_output = 12
        else:
            self.n_output = 5
        self.name = f'WGAN_DES_sim_{sn_type}_CCSNe_{self.mode}_clr{self.clr}_glr{self.glr}_ld{self.latent_dims}' \
                        f'_GP{self.GP}_zlim{self.z_lim}_bn{self.batch_norm}_gN{self.gen_units}_cN{self.crit_units}' \
                        f'_gd{self.g_dropout}_cd{self.c_dropout}_ds{self.ds}_colour{self.inc_colour}' \
                        f'_ncrit{self.n_critic}'
        if not os.path.exists(os.path.join('Data', 'Models', self.experiment)):
            os.mkdir(os.path.join('Data', 'Models', self.experiment))
        if not os.path.exists(os.path.join('Data', 'Models', self.experiment, 'WGAN')):
            os.mkdir(os.path.join('Data', 'Models', self.experiment, 'WGAN'))
        if not os.path.exists(os.path.join('Data', 'Models', self.experiment, 'WGAN', self.mode)):
            os.mkdir(os.path.join('Data', 'Models', self.experiment, 'WGAN', self.mode))
        self.root = os.path.join('Data', 'Models', self.experiment, 'WGAN', self.mode, self.name)
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        self.dataset_name = f'WGAN_DES_sim_{self.mode}_colour{self.inc_colour}_GP{self.GP}_zlim{self.z_lim}'
        self.dataset_path = os.path.join('Data', 'Datasets', f'{self.dataset_name}.csv')
        if os.path.exists(self.dataset_path):
            self.train_df = pd.read_csv(self.dataset_path)
            self.scaling_factors = pickle.load(open(os.path.join('Data', 'Datasets', f'{self.dataset_name}'
                                                                                     f'_scaling_factors.pkl'), 'rb'))
        else:
            print('Dataset does not already exist, creating now...')
            self.train_df, self.scaling_factors = self.__prepare_dataset__()
        self.train_df = self.train_df[self.train_df.sn_type == self.class_label_encoder[sn_type]]
        self.wgan_dir = os.path.join(self.root, 'model_weights')

        with tf.device(self.device):
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
            self.wgan = WGANModel(self.build_critic(), self.build_generator(), self.latent_dims)
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
            if 'HEAD' in file:
                head_file = file
                phot_file = file.replace('HEAD', 'PHOT')
                # try:
                head = Table.read(os.path.join(data_dir, head_file), hdu=1)
                # except:
                #     continue
                head_df = head.to_pandas()
                total += head_df.shape[0]
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
                    if self.mode == 'template':
                        dist_mod = 5 * np.log10(Distance(z=row.REDSHIFT_FINAL, unit=u.pc).value) - 5
                        sn_phot_df['mag'] -= dist_mod
                    sn_phot_df['sn'] = row.SNID
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
        if self.mode == 'template':
            all_df = all_df[all_df.mag < -10.733477]  # 99th percentile
        elif self.mode == 'observed':
            all_df = all_df[all_df.mag < 27.428694]  # 99th percentile
        min_t, max_t = all_df.t.min(), all_df.t.max()
        min_mag, max_mag = all_df.mag.max(), all_df.mag.min()
        all_df = all_df[all_df.mag_err < 1]  # Remove points with magnitude uncertainties greater than 1 mag
        scaled_mag = all_df['mag'].values - np.mean([min_mag, max_mag])
        scaled_mag /= (max_mag - min_mag) / 2
        all_df['mag'] = scaled_mag
        all_df['mag_err'] /= np.abs(max_mag - min_mag) / 2
        # all_df = all_df[~(all_df.mag < 0.1)]  # Remove bad points # & (all_df.mag_err > 0.5))]

        all_new_df = None
        used_count, skip_count, gp_error, no_peak, no_points, nans, less_than_zero = 0, 0, 0, 0, 0, 0, 0

        sn_list = list(all_df.sn.unique())

        for i, sn in tqdm(enumerate(sn_list), total=len(sn_list)):
            sn_df = all_df[all_df.sn == sn]
            if self.mode == 'template':
                # If rnn, can have different length inputs for each SN but still need to have same number of data
                # points in each band, this section calculates the times to use for each SN
                min_times, max_times = [], []
                for f in ['g', 'r', 'i', 'z']:
                    fdf = sn_df[sn_df.FLT == f]
                    min_times.append(fdf.t.min())
                    max_times.append(fdf.t.max())
                all_t = sn_df['t'].values

                all_t = all_t[all_t >= np.max(min_times)]
                all_t = all_t[all_t <= np.min(max_times)]
                all_t = (all_t - min_t) / (max_t - min_t)
                ind = 0
                while True:
                    if ind == len(all_t):
                        break
                    val = all_t[-(ind + 1)]
                    if any(np.abs(all_t[all_t != val] - val) < (1.0 / (max_t - min_t))):
                        all_t = np.delete(all_t, -(ind + 1))
                    else:
                        ind += 1
                new_sn_df = pd.DataFrame(all_t, columns=['t'])
            else:
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
                if self.mode == 'observed':
                    if use_len < fdf.shape[0]:
                        drop_count = fdf.shape[0] - use_len
                        skip_num = int(np.floor(drop_count / 2))
                        fdf = fdf.iloc[skip_num: skip_num + use_len, :]
                    if self.inc_colour:
                        min_filt = ['g', 'r', 'i', 'z'][np.argmin(lens)]
                        t = (fdf.t - min_t) / (max_t - min_t)
                        fit_t = (sn_df[sn_df.FLT == min_filt]['t'] - min_t) / (max_t - min_t)
                        new_sn_df['t'] = t.values
                    else:
                        t = (fdf.t - min_t) / (max_t - min_t)
                        fit_t = (fdf.t - min_t) / (max_t - min_t)
                        new_sn_df[f'{f}_t'] = t.values
                else:
                    t = (fdf.t - min_t) / (max_t - min_t)
                    fit_t = all_t
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
                    if self.mode == 'template':
                        mu, cov = gp.predict(y, fit_t)
                    elif self.mode == 'observed':
                        mu, cov = gp.predict(y, t)
                    std = np.sqrt(np.diag(cov))
                except:
                    if not skip:
                        gp_error += 1
                    skip = True
                    continue
                if self.mode == 'observed':
                    test_t, test_mu = t[mu > 0], mu[mu > 0]
                else:
                    test_t, test_mu = all_t[mu > 0], mu[mu > 0]
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
        print(used_count, skip_count, gp_error, less_than_zero)
        all_new_df.to_csv(self.dataset_path)
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
        with tf.device(self.device):
            input = Input(shape=(None, self.latent_dims))
            if self.inc_colour:
                gru1 = GRU(self.gen_units, activation='relu', return_sequences=True)(input)
                dr1 = Dropout(self.g_dropout)(gru1)
                gru2 = GRU(int(self.gen_units / 2), activation='relu', return_sequences=True)(dr1)
                dr2 = Dropout(self.g_dropout)(gru2)
                gru3 = GRU(int(self.gen_units / 4), activation='relu', return_sequences=True)(dr2)
                dr3 = Dropout(self.g_dropout)(gru3)
                output = GRU(self.n_output, return_sequences=True, activation='tanh')(dr3)
                model = Model(input, output)
            else:
                gru1 = GRU(self.gen_units, activation='relu', return_sequences=True)(input)
                dr1 = Dropout(self.g_dropout)(gru1)
                gru2 = GRU(self.gen_units, activation='relu', return_sequences=True)(dr1)
                dr2 = Dropout(self.g_dropout)(gru2)
                gru3 = GRU(int(self.gen_units / 4), activation='relu', return_sequences=True)(dr2)
                dr3 = Dropout(self.g_dropout)(gru3)
                output = GRU(self.n_output, return_sequences=True, activation='sigmoid')(dr2)
                model = Model(input, output)
            return model

    def build_critic(self):
        """
        Builds critic model
        :return: model, keras Model object for critic
        """
        with tf.device(self.device):
            input = Input(shape=(None, self.n_output))
            gru1 = GRU(self.crit_units, return_sequences=True)(input)
            dr1 = Dropout(self.c_dropout)(gru1)
            if self.batch_norm:
                bn1 = BatchNormalization()(gru1)
                gru2 = GRU(self.crit_units, return_sequences=True)(bn1)
                bn2 = BatchNormalization()(gru2)
                output = GRU(1, activation=None)(bn2)
            else:
                # gru2 = GRU(self.crit_units, return_sequences=True)(dr1)
                # dr2 = Dropout(self.c_dropout)(gru2)
                # output = GRU(1, activation=None)(dr2)
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
        if not os.path.exists(os.path.join(self.root, 'Training_sample')):
            os.mkdir(os.path.join(self.root, 'Training_sample'))
        for sn in tqdm(self.train_df.sn.unique()):
            sndf = self.train_df[self.train_df.sn == sn]
            plt.figure(figsize=(12, 8))
            for b_ind, band in enumerate(['g', 'r', 'i', 'z']):
                ax = plt.subplot(2, 2, b_ind + 1)
                if self.mode == 'template':
                    ax.scatter(sndf['t'], sndf[band], label=band)
                elif self.mode == 'observed':
                    ax.errorbar(sndf[f'{band}_t'], sndf[band], yerr=sndf[f'{band}_err'], fmt='x', label=band)
                ax.legend()
            plt.savefig(os.path.join(self.root, 'Training_sample', f'{sn}.jpg'))
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
        if not os.path.exists(os.path.join(self.root, 'Train_plots')):
            os.mkdir(os.path.join(self.root, 'Train_plots'))
        with tf.device(self.device):
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

            for epoch in range(current_epoch, epochs):
                rng.shuffle(sne)
                g_losses, d_losses, real_predictions, fake_predictions = [], [], [], []
                t = trange(n_batches)
                for batch in t:
                    # Select real data
                    sn = sne[batch]
                    sndf = self.train_df[self.train_df.sn == sn]
                    if self.mode == 'observed' and not self.inc_colour:
                        if self.ds == 1:
                            X = sndf[['g_t', 'r_t', 'i_t', 'z_t', 'g', 'r', 'i', 'z', 'g_err',
                                      'r_err', 'i_err', 'z_err']].values
                        elif self.ds == 2:
                            X = sndf[['g_t', 'g', 'g_err', 'r_t', 'r', 'r_err', 'i_t', 'i', 'i_err',
                                      'z_t', 'z', 'z_err']].values
                        else:
                            raise ValueError('Invalid option for data structure')
                    elif self.mode == 'observed' and self.inc_colour:
                        sndf[['g', 'r', 'i', 'z']] = 2 * (sndf[['g', 'r', 'i', 'z']] - 0.5)
                        sndf[['g_err', 'r_err', 'i_err', 'z_err']] = 2 * sndf[['g_err', 'r_err', 'i_err', 'z_err']]
                        sndf['g-r'] = sndf.g.values - sndf.r.values
                        sndf['r-i'] = sndf.r.values - sndf.i.values
                        sndf['i-z'] = sndf.i.values - sndf.z.values
                        X = sndf[['t', 'g', 'g-r', 'r', 'r-i', 'i', 'i-z', 'z', 'g_err',
                                  'r_err', 'i_err', 'z_err']].values
                    elif self.mode == 'template':
                        X = sndf[['t', 'g', 'r', 'i', 'z']].values

                    sn_type = X[0, -1]
                    X = X.reshape((1, *X.shape))
                    d_loss, g_loss = self.wgan.train_on_batch(X)

                    if np.count_nonzero(np.isnan(X)) > 0:
                        continue

                    noise = rand.uniform(-1, 1, size=(batch_size, self.latent_dims))
                    # noise = rand.normal(size=(batch_size, self.latent_dims))
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
                print(f'{epoch + 1}/{epochs} g_loss={full_g_loss}, d_loss={full_d_loss}')#, '
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
                    if self.mode == 'observed' and not self.inc_colour:
                        if self.ds == 1:
                            x, y, y_err = plot_test[:, f_ind], plot_test[:, f_ind + 4], plot_test[:, f_ind + 8]
                            ax.errorbar(x, y, yerr=y_err, label=f, fmt='x')
                            ax.errorbar(X[:, f_ind], X[:, f_ind + 4], yerr=X[:, f_ind + 8], fmt='x')
                        elif self.ds == 2:
                            x, y, y_err = plot_test[:, f_ind * 3], plot_test[:, f_ind * 3 + 1], plot_test[:, f_ind*3+2]
                            ax.errorbar(x, y, yerr=y_err, label=f, fmt='x')
                            ax.errorbar(X[:, f_ind * 3], X[:, f_ind * 3 + 1], yerr=X[:, f_ind * 3 + 2], fmt='x')
                        # ax.errorbar(x, y, yerr=y_err, label=f, fmt='x')
                        # ax.errorbar(X[:, f_ind], X[:, f_ind + 4], yerr=X[:, f_ind + 8], fmt='x')
                    elif self.mode == 'observed' and self.inc_colour:
                        x, y, y_err = plot_test[:, 0], plot_test[:, 1 + f_ind * 2], plot_test[:, f_ind + 8]
                        ax.errorbar(x, y, yerr=y_err, label=f, fmt='x')
                        ax.errorbar(X[:, 0], X[:, 1 + f_ind * 2], yerr=X[:, f_ind + 8], fmt='x')
                    elif self.mode == 'template':
                        y = plot_test[:, f_ind + 1]
                        ax.scatter(x, y, label=f)
                        ax.scatter(X[:, 0], X[:, f_ind + 1])
                    ax.legend()
                plt.suptitle(f'Epoch {epoch + 1}/{epochs}')  #: Type {self.class_label_dict[sn_type]}')
                plt.savefig(os.path.join(self.root, 'Train_plots', f'{epoch + 1}.png'))
                if plot_interval is not None:
                    if (epoch + 1) % plot_interval == 0:
                        plt.show()
                plt.close('all')

    def colour_analysis(self, epoch=-1, n=1):
        """
        Compares colour distribution of training set to generated light curves
        :param epoch: Epoch of model weights to use for light curve generation
        :param n: Number of light curves to generate
        """
        if epoch == -1:
            epoch = len(os.listdir(self.generator_dir))
        self.generator.load_weights(os.path.join(self.generator_dir, f'{epoch}.h5'))
        all_new_df, all_gen_df = None, None
        for sn in self.train_df.sn.unique():
            # Deal with real data first -----------------------------------------------------
            sn_df = self.train_df[self.train_df.sn == sn]

            gp = george.GP(Matern32Kernel(1))

            t = sn_df['t']
            fine_t = np.arange(t.min(), t.max(), 0.001)
            y = sn_df.g
            y_err = sn_df.g_err

            # Gradient function for optimisation of kernel size------------------------------

            def ll(p):
                gp.set_parameter_vector(p)
                return -gp.log_likelihood(y, quiet=True)

            def grad_ll(p):
                gp.set_parameter_vector(p)
                return -gp.grad_log_likelihood(y, quiet=True)

            gp.compute(t, y_err)
            p0 = gp.kernel.get_parameter_vector()[0]
            results = spopt.minimize(ll, p0, jac=grad_ll)
            mu, cov = gp.predict(y, fine_t)
            tmax = fine_t[np.argmax(mu)]
            t -= tmax

            new_df = sn_df.copy()
            new_df['t'] = new_df['t'] * (self.scaling_factors[1] - self.scaling_factors[0]) + self.scaling_factors[0]
            for f in ['g', 'r', 'i', 'z']:
                new_df[f] = new_df[f] * ((self.scaling_factors[3] - self.scaling_factors[2]) / 2) + \
                            np.mean([self.scaling_factors[3], self.scaling_factors[2]])
            if all_new_df is None:
                all_new_df = new_df.copy()
            else:
                all_new_df = pd.concat([all_new_df, new_df])

            # Then deal with generated light curve--------------------------------------------

            noise = rand.normal(size=(n, self.latent_dims))
            noise = np.reshape(noise, (n, 1, self.latent_dims))
            noise = np.repeat(noise, new_df.shape[0], 1)
            labels = np.full((n, new_df.shape[0], 1), new_df.sn_type.values[0])
            gen_lc = self.generator.predict([noise, labels])

            gen_lc_df = None

            for i in range(n):
                single_lc_df = pd.DataFrame(gen_lc[i, :, :], columns=['t', 'g', 'r', 'i', 'z', 'sn_type'])
                single_lc_df['sn'] = i
                if gen_lc_df is None:
                    gen_lc_df = single_lc_df.copy()
                else:
                    gen_lc_df = pd.concat([gen_lc_df, single_lc_df])

            # gen_lc = pd.DataFrame(gen_lc.reshape(gen_lc.shape[1:]), columns=['t', 'g', 'r', 'i', 'z', 'sn_type'])

            gp = george.GP(Matern32Kernel(1))

            t = gen_lc_df['t']
            fine_t = np.arange(t.min(), t.max(), 0.001)
            y = gen_lc_df.g
            y_err = 0.02

            def ll(p):
                gp.set_parameter_vector(p)
                return -gp.log_likelihood(y, quiet=True)

            def grad_ll(p):
                gp.set_parameter_vector(p)
                return -gp.grad_log_likelihood(y, quiet=True)

            gp.compute(t, y_err)
            p0 = gp.kernel.get_parameter_vector()[0]
            results = spopt.minimize(ll, p0, jac=grad_ll)
            mu, cov = gp.predict(y, fine_t)
            tmax = fine_t[np.argmax(mu)]
            t -= tmax

            gdf = gen_lc_df.copy()
            gdf['t'] = gdf['t'] * (self.scaling_factors[1] - self.scaling_factors[0]) + self.scaling_factors[0]
            for f in ['g', 'r', 'i', 'z']:
                gdf[f] = gdf[f] * ((self.scaling_factors[3] - self.scaling_factors[2]) / 2) + \
                            np.mean([self.scaling_factors[3], self.scaling_factors[2]])
            if all_gen_df is None:
                all_gen_df = gdf.copy()
            else:
                all_gen_df = pd.concat([all_gen_df, gdf])

        for f in [('g', 'r'), ('r', 'i'), ('i', 'z')]:
            f1, f2 = f
            all_new_df[f'{f1}-{f2}'] = all_new_df[f1] - all_new_df[f2]
            all_gen_df[f'{f1}-{f2}'] = all_gen_df[f1] - all_gen_df[f2]

        for sn_type in all_new_df.sn_type.unique():
            for f in ['g-r', 'r-i', 'i-z']:
                plt.figure()
                for label, df in zip(['Real', 'Simulated'], [all_new_df, all_gen_df]):
                    type_df = df[df.sn_type == sn_type]
                    bins = pd.cut(type_df['t'].values, np.arange(type_df.t.min(), type_df.t.max(), 5))
                    ts, cs, c_errs = [], [], []
                    for i, cat in enumerate(bins.categories):
                        tdf = type_df.iloc[np.argwhere(bins.codes == i).flatten(), :]
                        c, c_err = tdf[f].mean(), tdf[f].std()
                        ts.append(cat.mid)
                        cs.append(c)
                        c_errs.append(c_err)
                    ts, cs, c_errs = np.array(ts), np.array(cs), np.array(c_errs)
                    if label == 'Real':
                        plt.plot(ts, cs, label=label)
                        plt.fill_between(ts, cs - c_errs, cs + c_errs, alpha=0.3)
                    else:
                        plt.errorbar(ts, cs, yerr=c_errs, fmt='x', label=label)
                plt.title(f'SNe {self.class_label_dict[sn_type]}')
                plt.xlabel('Phase (days)')
                plt.ylabel(f)
                plt.legend()
                if not os.path.exists(os.path.join(self.root, 'colour_curves')):
                    os.mkdir(os.path.join(self.root, 'colour_curves'))
                plt.savefig(os.path.join(self.root, 'colour_curves', f'{self.class_label_dict[sn_type]}_{f}_{n}.png'))
                plt.close('all')

    def gen_lightcurves(self, n=10, length=20):
        noise = rand.normal(size=(n, self.latent_dims))
        noise = np.reshape(noise, (n, 1, self.latent_dims))
        noise = np.repeat(noise, length, 1)
        self.generator.load_weights(self.generator_path)
        gen_lcs = self.generator.predict(noise)
        gen_lcs[:, :, 0] = gen_lcs[:, :, 0] * (self.scaling_factors[1] - self.scaling_factors[0]) + \
                           self.scaling_factors[0]
        gen_lcs[:, :, 1:] = gen_lcs[:, :, 1:] * ((self.scaling_factors[3] - self.scaling_factors[2]) / 2) + np.mean(
            [self.scaling_factors[3], self.scaling_factors[2]])
        for i in range(n):
            # Plot generated data
            data = gen_lcs[i, :, :]
            x = data[:, 0]
            fig = plt.figure(figsize=(12, 8))
            for f_ind, f in enumerate(['g', 'r', 'i', 'z']):
                ax = fig.add_subplot(2, 2, f_ind + 1)
                x = gen_lcs[i, :, 0]
                y = gen_lcs[i, :, f_ind + 1]
                ax.scatter(x, y, label=f)
                ax.legend()
                ax.invert_yaxis()
            plt.show()

    def plot_lightcurves(self, n=10, length=12, model=None, epochs=np.arange(1, 201)):
        for epoch in tqdm(epochs):
            if not os.path.exists(os.path.join(self.root, 'Generated_Plots')):
                os.mkdir(os.path.join(self.root, 'Generated_Plots'))
            noise = rand.normal(size=(n, self.latent_dims))
            noise = np.reshape(noise, (n, 1, self.latent_dims))
            noise = np.repeat(noise, length, 1)
            # self.wgan(np.zeros((1, 15, 12)))
            self.wgan = keras.models.load_model(os.path.join(self.wgan_dir, f'{epoch}.tf'))
            gen_lcs = self.wgan.generator.predict(noise)
            gen_lcs[:, :, 0:4] = gen_lcs[:, :, 0:4] * (self.scaling_factors[1] - self.scaling_factors[0]) + \
                                self.scaling_factors[0]
            gen_lcs[:, :, 4:8] = gen_lcs[:, :, 4:8] * ((self.scaling_factors[3] - self.scaling_factors[2]) / 2) + np.mean(
                [self.scaling_factors[3], self.scaling_factors[2]])
            gen_lcs[:, :, 8:] = gen_lcs[:, :, 8:] * ((self.scaling_factors[3] - self.scaling_factors[2]) / 2)
            sn = np.random.choice(self.train_df.sn.values)
            sndf = self.train_df[self.train_df.sn == sn]
            X = sndf[['g_t', 'r_t', 'i_t', 'z_t', 'g', 'r', 'i', 'z', 'g_err',
                                      'r_err', 'i_err', 'z_err']].values
            X = X.reshape((1, *X.shape))
            X[:, :, 0:4] = X[:, :, 0:4] * (self.scaling_factors[1] - self.scaling_factors[0]) + \
                                 self.scaling_factors[0]
            X[:, :, 4:8] = X[:, :, 4:8] * (
                        (self.scaling_factors[3] - self.scaling_factors[2]) / 2) + np.mean(
                [self.scaling_factors[3], self.scaling_factors[2]])
            X[:, :, 8:] = X[:, :, 8:] * ((self.scaling_factors[3] - self.scaling_factors[2]) / 2)
            for i in np.arange(n):
                # Plot generated data
                data = gen_lcs[i, :, :]
                g_band_max = gen_lcs[i, :, 0][np.argmin(gen_lcs[i, :, 4])]
                g_band_max_real = X[0, :, 0][np.argmin(X[0, :, 4])]
                fig = plt.figure(figsize=(12, 8))
                ax_list = []
                for f_ind, f in enumerate(['g', 'r', 'i', 'z']):
                    ax = fig.add_subplot(2, 2, f_ind + 1)
                    x = gen_lcs[i, :, f_ind] - g_band_max
                    y = gen_lcs[i, :, f_ind + 4]
                    y_err = gen_lcs[i, :, f_ind + 8]
                    ax.errorbar(x, y, yerr=y_err, label=f'{f} (Generated)', fmt='x')
                    x = X[0, :, f_ind] - g_band_max_real
                    y = X[0, :, f_ind + 4]  # - 2
                    y_err = X[0, :, f_ind + 8]
                    ax.errorbar(x, y, yerr=y_err, label=f'{f} (Real)', fmt='x')
                    ax.legend(fontsize=16) #, loc='upper right')
                    # ax.annotate(f, (0.03, 0.9), xycoords='axes fraction')
                    ax.invert_yaxis()
                    ax.set_xlabel('Days since g-band maximum')
                    if f_ind in [0, 2]:
                        ax.set_ylabel('Apparent magnitude')
                    ax_list.append(ax)
                ax_list[0].get_shared_x_axes().join(*ax_list)
                ax_list[0].get_shared_y_axes().join(*ax_list)
                ax_list[1].set_yticklabels([])
                ax_list[3].set_yticklabels([])
                plt.subplots_adjust(hspace=0, wspace=0)
                plt.savefig(os.path.join(self.root, 'Generated_Plots', f'{epoch}-{i + 1}.png'))
                # plt.show()
                plt.close('all')
                # raise ValueError('Nope')
                # break
