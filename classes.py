import os

import numpy as np
import pandas as pd
import scipy.optimize as spopt
import pickle
import numpy.random as rand
import matplotlib.pyplot as plt
from astropy.table import Table
from tqdm import tqdm, trange
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers as opt
from tensorflow.keras.layers import Input, Dense, GRU, Dropout, Reshape, BatchNormalization, Activation
import george
from george.kernels import Matern32Kernel
import time


class GAN:
    def __init__(self, latent_dims=100, lr=0.00001, gen_activation='sigmoid', device='gpu:0', labels='hard', GP=True,
                 generator_type='dense', cadence=None, data_type='sim', z_lim=None, batch_norm=False, error=False,
                 frame='obs', dropout=0.5, units=100):
        self.latent_dims = latent_dims
        self.lr = lr
        self.gen_activation = gen_activation
        self.device = device
        self.labels = labels
        self.GP = GP
        self.generator_type = generator_type
        self.cadence = cadence
        self.data_type = data_type
        self.z_lim = z_lim
        self.batch_norm = batch_norm
        self.error = error
        self.frame = frame
        self.dropout = dropout
        self.units = units
        if self.error:
            self.n_output = 12
        else:
            self.n_output = 5
        if self.data_type == 'real':
            self.name = f'DES_real_CCSNe_{self.gen_activation}_lr{self.lr}_ld{self.latent_dims}_labels{self.labels}' \
                        f'_GP{self.GP}_{self.generator_type}_err{self.error}'
        elif self.data_type == 'sim':
            self.name = f'DES_sim_CCSNe_{self.gen_activation}_lr{self.lr}_ld{self.latent_dims}_labels{self.labels}' \
                        f'_GP{self.GP}_zlim{self.z_lim}_{self.generator_type}_bn{self.batch_norm}_N{self.units}'
        if self.error:
            self.root = os.path.join('Data', 'Models', 'err', self.name)
        else:
            self.root = os.path.join('Data', 'Models', 'noerr', self.name)
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        if self.data_type == 'real':
            self.dataset_name = f'DES_real_GP{self.GP}_cadence{self.cadence}_{self.generator_type}'
            self.dataset_path = os.path.join('Data', 'Datasets', f'{self.dataset_name}.csv')
        elif self.data_type == 'sim':
            self.dataset_name = f'DES_sim_GP{self.GP}_zlim{self.z_lim}_cadence{self.cadence}_{self.generator_type}' \
                                f'_error{self.error}'
            self.dataset_path = os.path.join('Data', 'Datasets', f'{self.dataset_name}.csv')
        else:
            raise ValueError('Invalid dataset type, must be either real or sim')
        if os.path.exists(self.dataset_path):
            self.train_df = pd.read_csv(self.dataset_path)
            self.scaling_factors = pickle.load(open(os.path.join('Data', 'Datasets', f'{self.dataset_name}'
                                                                                     f'_scaling_factors.pkl'), 'rb'))
        else:
            print('Dataset does not already exist, creating now...')
            self.train_df, self.scaling_factors = self.__prepare_dataset__()
        self.generator_dir = os.path.join(self.root, 'model_weights')

        with tf.device(self.device):
            # Optimizer
            self.optimizer = opt.Adam(learning_rate=self.lr)

            # Build discriminator
            self.discriminator = self.build_discriminator()
            self.discriminator.compile(loss='binary_crossentropy')
            print(self.discriminator.summary())

            # Build generator
            self.generator = self.build_generator()
            self.generator.compile(loss='binary_crossentropy', optimizer=self.optimizer)
            print(self.generator.summary())

            # Build combined model
            if self.generator_type == 'dense':
                i = Input(shape=self.latent_dims)
            elif self.generator_type == 'rnn':
                i = Input(shape=(None, self.latent_dims))
            else:
                raise ValueError('Invalid generator, must choose one of dense or rnn')
            lcs = self.generator(i)

            self.discriminator.trainable = False

            valid = self.discriminator(lcs)

            self.combined = Model(i, valid)
            self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)
            print(self.combined.summary())

    def __prepare_dataset__(self):

        if self.data_type == 'real':
            # Some overall setup stuff----------------------------------------------------
            years = [(56500, 56750), (56850, 57100), (57200, 57450), (57600, 57850),
                     (57900, 58200)]  # rough boundaries of each observing period in MJD
            years_conv = {13: 0, 14: 1, 15: 2, 16: 3, 17: 4}
            zp_ab = {'g': 20.802, 'r': 21.436, 'i': 21.866, 'z': 22.214}
            exp_data = pd.read_pickle('Data/exp_date.pkl')
            R_v = 3.1

            # Start by calculating upper and lower bounds for each band and in time----------------------------

            des_df = pd.read_csv('Data/sn_max_data_all_err_H070_host_colour+mag_ext.csv')
            des_df = des_df[des_df.rest_filter == 'R']

            all_min_xs, all_max_xs, all_x_lims = [], [], {}
            all_max_fluxes = {f: [] for f in ['g', 'r', 'i', 'z']}
            all_min_mags, all_max_mags = {f: [] for f in ['g', 'r', 'i', 'z']}, {f: [] for f in ['g', 'r', 'i', 'z']}
            all_exp_dates = []
            all_keep_data = {}

            for i, row in des_df.reset_index().iterrows():
                sn, snid, z = row.sn, row.sn_id, row.z_sn
                filename = f'des_real_0{snid}.dat'
                data_df = pd.read_csv(os.path.join('Data', 'SN_Data', filename), skiprows=55, usecols=[1, 2, 4, 5],
                                      delim_whitespace=True, header=None, names=['mjd', 'filt', 'flux', 'flux_err'],
                                      comment='#')
                fp = open(os.path.join('Data', 'SN_Data', filename))

                for line in fp:
                    if line[0:4] == 'IAUC':
                        maxyear = years_conv[int(line.split()[1][3:5])]
                        name = line.split()[1]
                    if line[0:4] == 'SNID':
                        id = line.split()[1]
                    if line[0:2] == 'RA':
                        RA = line.split()[1]
                    if line[0:4] == 'DECL':
                        Dec = line.split()[1]
                    if line[0:13] == 'HOSTGAL_SPECZ':
                        z = float(line.split()[1])
                        break

                if z < 0:
                    continue

                fp.close()

                df = data_df[
                    (years[maxyear][0] < data_df.mjd) & (
                            data_df.mjd < years[maxyear][1])]  # Select only for year that SN occurs

                # Identify explosion date----------------------------------------------------
                exp_dates = []

                for f_ind, f in enumerate(['g', 'r', 'i', 'z']):
                    # Open data--------------------------------------------------------------
                    df_filt = df[df.filt == f]

                    mjd = df_filt.mjd.values
                    flux = df_filt.flux.values
                    flux_err = df_filt.flux_err.values

                    z_p = zp_ab[f]

                    # Convert to f_lambda----------------------------------------------------
                    flux = flux * 1e-11 * 10 ** (-0.4 * z_p)
                    flux_err = 1e-11 * 10 ** (-0.4 * z_p) * flux_err

                    # Determine start of explosion and apply cut-----------------------------
                    exp_date = None
                    f_max = flux[np.argmax((flux - flux_err) / flux_err)]
                    if np.argmax(flux) == 0 or flux[0] - 4 * flux_err[0] > 0:
                        if any(exp_data.SN == name):
                            exp_date = exp_data[exp_data.SN == name].values[0][1]
                            exp_dates.append(exp_date)
                        else:
                            continue
                    else:
                        for ind, val in enumerate(flux):
                            try:
                                if val - 4 * flux_err[ind] > 0 and ((val < flux[ind + 1] < flux[ind + 2] and
                                                                     flux[ind + 1] > f_max / 5) or val > f_max / 3):
                                    exp_date = (mjd[ind] + mjd[ind - 1]) / 2
                                    exp_dates.append(exp_date)
                                    break
                            except:
                                break
                    if exp_date is None:
                        continue

                exp_date = np.min(exp_dates)
                all_exp_dates.append(exp_date)

                # Get data epochs------------------------------------------------------------
                min_x, max_x = [], []
                all_times = []

                lens = []
                keep_data = {}

                for f_ind, f in enumerate(['g', 'r', 'i', 'z']):
                    # Open data--------------------------------------------------------------
                    df_filt = df[df.filt == f]
                    mjd = df_filt.mjd.values
                    flux = df_filt.flux.values
                    flux_err = df_filt.flux_err.values

                    flux = flux[mjd > exp_date]
                    flux_err = flux_err[mjd > exp_date]
                    mjd = mjd[mjd > exp_date]

                    z_p = zp_ab[f]

                    # Convert to f_lambda----------------------------------------------------
                    flux = flux * 1e-11 * 10 ** (-0.4 * z_p)
                    flux_err = 1e-11 * 10 ** (-0.4 * z_p) * flux_err

                    # Convert to mags--------------------------------------------------------
                    mag = -2.5 * np.log10(flux) - zp_ab[f]
                    mag_err = (2.5 * flux_err / (flux * np.log(10)))

                    mjd = mjd[~(np.isnan(mag) | np.isinf(mag))]
                    flux = flux[~(np.isnan(mag) | np.isinf(mag))]
                    flux_err = flux_err[~(np.isnan(mag) | np.isinf(mag))]
                    mag_err = mag_err[~(np.isnan(mag) | np.isinf(mag))]
                    mag = mag[~(np.isnan(mag) | np.isinf(mag))]

                    x = (mjd - exp_date) / (1 + z)
                    keep_data[f] = [x, flux, flux_err, mag, mag_err]
                    min_x.append(np.min(x))
                    max_x.append(np.max(x))
                    all_max_fluxes[f].append(np.max(flux))
                    all_min_mags[f].append(np.nanmax(mag))
                    all_max_mags[f].append(np.nanmin(mag))
                all_min_xs.append(np.max(min_x))
                all_max_xs.append(np.min(max_x))
                all_x_lims[sn] = (np.max(min_x), np.min(max_x))
                all_keep_data[sn] = [exp_date, keep_data]

            x_lims = (np.min(all_min_xs), np.max(all_max_xs))
            max_fluxes = {f: np.max(all_max_fluxes[f]) for f in ['g', 'r', 'i', 'z']}
            mag_lims = {f: (np.max(all_min_mags[f]), np.min(all_max_mags[f])) for f in ['g', 'r', 'i', 'z']}

            if self.cadence is not None:
                t = np.arange(x_lims[0], x_lims[1], self.cadence)

            all_df = None

            for i, row in des_df.reset_index().iterrows():
                sn, snid, z = row.sn, row.sn_id, row.z_sn
                if sn not in all_keep_data.keys():
                    continue
                exp_date, data = all_keep_data[sn]

                sn_dict = {'t': (t - t.mean()) / (t.max() / 2)}
                for f_ind, f in enumerate(['g', 'r', 'i', 'z']):
                    x, flux, flux_err, mag, mag_err = data[f]

                    # If mode == GP, GP all data--------------------------------------------
                    if self.GP and self.cadence is not None:
                        if self.mode == 'flux':
                            y, y_err = flux, flux_err
                        elif self.mode == 'mag':
                            y, y_err = mag, mag_err

                        scale = np.max(y)
                        y, y_err = y / scale, y_err / scale

                        gp = george.GP(Matern32Kernel(500))

                        # Gradient function for optimisation of kernel size------------------------------

                        def ll(p):
                            gp.set_parameter_vector(p)
                            return -gp.log_likelihood(y, quiet=True)

                        def grad_ll(p):
                            gp.set_parameter_vector(p)
                            return -gp.grad_log_likelihood(y, quiet=True)

                        gp.compute(x, y_err)
                        p0 = gp.kernel.get_parameter_vector()[0]

                        results = spopt.minimize(ll, p0, jac=grad_ll)

                        mu, cov = gp.predict(y, t)
                        std = np.sqrt(np.diag(cov))

                        mu, std = mu * scale, std * scale

                        if self.mode == 'flux':
                            for ind, val in enumerate(t):
                                if val < all_x_lims[sn][0] or val > all_x_lims[sn][1]:
                                    mu[ind] = 0
                                    std[ind] = 0
                            mu /= (max_fluxes[f] / 2)
                            mu -= 1
                        elif self.mode == 'mag':
                            for ind, val in enumerate(t):
                                if val < all_x_lims[sn][0] or val > all_x_lims[sn][1]:
                                    mu[ind] = mag_lims[f][0]
                                    std[ind] = 0
                            mu -= np.mean(mag_lims[f])
                            # mu *= -1
                            mu /= (mag_lims[f][1] - mag_lims[f][0]) / 2
                        sn_dict[f] = mu
                        sn_dict[f'{f}_err'] = std
                    else:
                        raise ValueError('This setting for mode and cadence has not yet been implemented')
                sn_df = pd.DataFrame(sn_dict)
                sn_df['sn'] = sn
                if all_df is None:
                    all_df = sn_df
                else:
                    all_df = pd.concat([all_df, sn_df])
            scaling_factors = [x_lims, max_fluxes, mag_lims]
            all_df.to_csv(os.path.join('Data', 'Datasets', f'{self.dataset_name}.csv'))
            pickle.dump(scaling_factors,
                        open(os.path.join('Data', 'Datasets', f'{self.dataset_name}_scaling_factors.pkl'), 'wb'))
            return all_df, scaling_factors
        elif self.data_type == 'sim':
            data_dir = 'Data/DESSIMBIAS5YRCC_V19/PIP_MV_GLOBAL_BIASCOR_DESSIMBIAS5YRCC_V19'
            x = os.listdir(data_dir)
            x.sort()

            all_df = None

            print('Reading in files...')
            for file in tqdm(x):
                if 'HEAD' in file:
                    head_file = file
                    phot_file = file.replace('HEAD', 'PHOT')
                    head = Table.read(os.path.join(data_dir, head_file), hdu=1)
                    head_df = head.to_pandas()
                    head_df = head_df[head_df.HOSTGAL_SPECZ < self.z_lim]
                    head_df = head_df[head_df.NOBS > 20]
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
                        if all_df is None:
                            all_df = sn_phot_df.copy()
                        else:
                            all_df = pd.concat([all_df, sn_phot_df])
                else:
                    continue

            # Drop nans and infs, they only cause problems!
            all_df = all_df[~(np.isnan(all_df.mag) | np.isinf(all_df.mag))]

            # Get bounds for light curve scaling and rescale so between 0 and 1
            min_t, max_t = all_df.t.min(), all_df.t.max()
            min_mag, max_mag = all_df.mag.max(), all_df.mag.min()
            all_df = all_df[all_df.mag_err < 1]  # Remove points with magnitude uncertainties greater than 1 mag
            scaled_mag = all_df['mag'].values - np.mean([min_mag, max_mag])
            scaled_mag /= (max_mag - min_mag) / 2
            all_df['mag'] = scaled_mag
            all_df['mag_err'] /= np.abs(max_mag - min_mag) / 2
            all_df = all_df[~(all_df.mag < 0.1)]  # Remove bad points # & (all_df.mag_err > 0.5))]

            if self.generator_type == 'dense':  # Need same length input for all SNe if using this type of generator
                if self.cadence is None:
                    raise ValueError('Please specify a cadence to use for this data')
                all_t = (np.arange(min_t, max_t, self.cadence) - min_t) / (max_t - min_t)
                all_df['t'] = (all_df['t'] - min_t) / (max_t - min_t)

            all_new_df = None
            used_count, skip_count = 0, 0

            sn_list = list(all_df.sn.unique())

            for i, sn in tqdm(enumerate(sn_list), total=len(sn_list)):
                sn_df = all_df[all_df.sn == sn]
                if self.generator_type == 'rnn' and not self.error:
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
                if not self.error:
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
                    fdf = sn_df[sn_df.FLT == f] # .sort_values('t')
                    gp = george.GP(Matern32Kernel(1))
                    if self.error:
                        if use_len < fdf.shape[0]:
                            drop_count = fdf.shape[0] - use_len
                            skip_num = int(np.floor(drop_count / 2))
                            fdf = fdf.iloc[skip_num: skip_num + use_len, :]
                        t = (fdf.t - min_t) / (max_t - min_t)
                        new_sn_df[f'{f}_t'] = t.values
                    else:
                        t = (fdf.t - min_t) / (max_t - min_t)
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
                        if self.error:
                            mu, cov = gp.predict(y, t)
                        else:
                            mu, cov = gp.predict(y, all_t)
                        std = np.sqrt(np.diag(cov))
                    except:
                        skip = True
                        continue
                    if self.generator_type == 'dense':
                        mu[(all_t < sn_t_min) | (all_t > sn_t_max)] = 0
                        std[(all_t < sn_t_min) | (all_t > sn_t_max)] = 0
                    if self.error:
                        test_t, test_mu = t[mu > 0], mu[mu > 0]
                    else:
                        test_t, test_mu = all_t[mu > 0], mu[mu > 0]
                    if len(test_mu) == 0:
                        skip = True
                        continue
                    if np.argmax(test_mu) == 0 or (np.argmin(test_mu) == 0 and np.argmax(test_mu) == len(test_mu) - 1)\
                            or len(mu[mu > 0.01]) < 8:
                        skip = True
                        continue
                    if any(np.isnan(val) for val in mu):
                        skip = True
                    if np.count_nonzero(np.isnan(mu)) > 0:
                        skip = True

                    new_sn_df[f] = mu
                    new_sn_df[f'{f}_err'] = std

                new_sn_df['sn'] = sn
                if skip:
                    skip_count += 1
                    continue
                used_count += 1
                if all_new_df is None:
                    all_new_df = new_sn_df.copy()
                else:
                    all_new_df = pd.concat([all_new_df, new_sn_df])
            all_new_df.to_csv(self.dataset_path)
            scaling_factors = [min_t, max_t, min_mag, max_mag]
            pickle.dump(scaling_factors,
                        open(os.path.join('Data', 'Datasets', f'{self.dataset_name}_scaling_factors.pkl'), 'wb'))
            return all_new_df, scaling_factors

    def build_generator(self):
        with tf.device(self.device):
            if self.generator_type == 'dense':
                input = Input(shape=self.latent_dims)
                dense1 = Dense(200, activation='relu')(input)
                bn1 = BatchNormalization()(dense1)
                ac1 = Activation('relu')(bn1)
                dense2 = Dense(200, activation='relu')(ac1)
                bn2 = BatchNormalization()(dense2)
                ac2 = Activation('relu')(bn2)
                dense3 = Dense(110, activation=self.gen_activation)(ac2)
                output = Reshape((22, 5))(dense3)
                model = Model(input, output)
                return model
            elif self.generator_type == 'rnn':
                input = Input(shape=(None, self.latent_dims))
                gru1 = GRU(self.units, activation='relu', return_sequences=True)(input)
                dr1 = Dropout(self.dropout)(gru1)
                gru2 = GRU(self.units, activation='relu', return_sequences=True)(dr1)
                dr2 = Dropout(self.dropout)(gru2)
                output = GRU(self.n_output, return_sequences=True, activation=self.gen_activation)(dr2)
                model = Model(input, output)
                return model
            else:
                raise ValueError('Invalid generator, must choose one of dense or rnn')

    def build_discriminator(self):
        with tf.device(self.device):
            input = Input(shape=(None, self.n_output))
            gru1 = GRU(self.units, return_sequences=True)(input)
            dr1 = Dropout(self.dropout)(gru1)
            if self.batch_norm:
                bn1 = BatchNormalization()(gru1)
                gru2 = GRU(self.units, return_sequences=True)(bn1)
                bn2 = BatchNormalization()(gru2)
                output = GRU(1, activation='sigmoid')(bn2)
            else:
                gru2 = GRU(self.units, return_sequences=True)(dr1)
                dr2 = Dropout(self.dropout)(gru2)
                output = GRU(1, activation='sigmoid')(dr2)
            model = Model(input, output)
            return model

    def plot_train_sample(self):
        print('Generating plots for training sample...')
        for sn in tqdm(self.train_df.sn.unique()):
            sndf = self.train_df[self.train_df.sn == sn]
            plt.figure(figsize=(12, 8))
            for band in ['g', 'r', 'i', 'z']:
                plt.scatter(sndf['t'], sndf[band], label=band)
            plt.legend()
            plt.savefig(os.path.join('Data', 'Training_sample_plots', f'{sn}.jpg'))
            plt.close('all')

    def train(self, epochs=100, batch_size=1, plot_interval=None):
        print('Starting training...')
        if not os.path.exists(os.path.join(self.root, 'Train_plots')):
            os.mkdir(os.path.join(self.root, 'Train_plots'))
        with tf.device(self.device):
            rng = np.random.default_rng(123)

            sne = self.train_df.sn.unique()
            n_batches = int(len(sne) / batch_size)

            if os.path.exists(self.generator_dir):
                raise ValueError('A weights path already exists for this model, please delete '
                                 'or rename it')
            os.mkdir(self.generator_dir)

            for epoch in range(epochs):
                rng.shuffle(sne)
                g_losses, d_losses = [], []
                t = trange(n_batches)
                for batch in t:
                    if self.labels == 'hard':
                        real = np.zeros((batch_size, 1))
                        fake = np.ones((batch_size, 1))
                    elif self.labels == 'soft':
                        real = np.random.uniform(0.0, 0.1, (batch_size, 1))
                        fake = np.random.uniform(0.9, 1.0, (batch_size, 1))
                    else:
                        raise ValueError('Labels must be one of hard or soft')

                    '''
                    for i in range(int(real.shape[0]/20)):
                      rand_ind = np.random.randint(real.shape[0])
                      real[rand_ind, 0] = np.random.uniform(0.9, 1.0)
    
                    for i in range(int(fake.shape[0]/20)):
                      rand_ind = np.random.randint(fake.shape[0])
                      fake[rand_ind, 0] = np.random.uniform(0.0, 0.1)
                    '''
                    # Select real data
                    sn = sne[batch]
                    sndf = self.train_df[self.train_df.sn == sn]
                    if self.error:
                        # X = sndf[['g_t', 'r_t', 'i_t', 'z_t', 'g', 'r', 'i', 'z', 'g_err',
                        #          'r_err', 'i_err', 'z_err']].values
                        X = sndf[['g_t', 'g', 'g_err', 'r_t', 'r', 'r_err', 'i_t', 'i', 'i_err',
                                  'z_t', 'z', 'z_err']].values
                    else:
                        X = sndf[['t', 'g', 'r', 'i', 'z']].values
                    X = X.reshape((1, *X.shape))

                    if np.count_nonzero(np.isnan(X)) > 0:
                        continue

                    # Generate fake data
                    # If using dense generator, easy to have N latent dimensions describe whole light curve
                    if self.generator_type == 'dense':
                        noise = rand.normal(size=(batch_size, self.latent_dims))
                    # If using RNN generator, need to generate N random points and repeat them for every timestep to
                    # ensure the same random seed is used for each point
                    elif self.generator_type == 'rnn':
                        noise = rand.normal(size=(batch_size, self.latent_dims))
                        noise = np.reshape(noise, (batch_size, 1, self.latent_dims))
                        noise = np.repeat(noise, X.shape[1], 1)

                    test_gen_lcs = self.generator.predict(noise)
                    if np.count_nonzero(np.isnan(test_gen_lcs)) > 0:
                        raise ValueError('NaN generated, check how this happened')
                    gen_lcs = test_gen_lcs

                    # Train discriminator
                    d_loss_real = self.discriminator.train_on_batch(X, real)
                    d_loss_fake = self.discriminator.train_on_batch(gen_lcs, fake)

                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                    # Train generator
                    if self.generator_type == 'dense':
                        noise = rand.normal(size=(2 * batch_size, self.latent_dims))
                    elif self.generator_type == 'rnn':
                        noise = rand.normal(size=(2 * batch_size, self.latent_dims))
                        noise = np.reshape(noise, (2 * batch_size, 1, self.latent_dims))
                        noise = np.repeat(noise, X.shape[1], 1)

                    gen_labels = np.full((2 * batch_size, 1), 0.1)  # np.zeros((2 * batch_size, 1))
                    g_loss = self.combined.train_on_batch(noise, gen_labels)
                    g_losses.append(g_loss)
                    d_losses.append(d_loss)
                    t.set_description(f'g_loss={np.around(np.mean(g_losses), 5)},'
                                      f' d_loss={np.around(np.mean(d_losses), 5)}')
                    t.refresh()
                self.generator.save_weights(os.path.join(self.generator_dir, f'{epoch + 1}.h5'))
                full_g_loss = np.mean(g_losses)
                full_d_loss = np.mean(d_losses)
                print(f'{epoch + 1}/{epochs} g_loss={full_g_loss}, d_loss={full_d_loss}'
                      f' Ranges: x [{np.min(gen_lcs[:, :, 0])}, {np.max(gen_lcs[:, :, 0])}], '
                      f'y [{np.min(gen_lcs[:, :, 1])}, {np.max(gen_lcs[:, :, 1])}]')

                plot_test = gen_lcs[0, :, :]
                fig = plt.figure(figsize=(12, 8))
                x = plot_test[:, 0]
                X = X.reshape((*X.shape[1:],))
                for f_ind, f in enumerate(['g', 'r', 'i', 'z']):
                    ax = fig.add_subplot(2, 2, f_ind + 1)
                    if self.error:
                        x, y, y_err = plot_test[:, 3 * f_ind], plot_test[:, 3 * f_ind + 1], plot_test[:, 3 * f_ind + 2]
                        ax.errorbar(x, y, yerr=y_err, label=f, fmt='x')
                        ax.errorbar(X[:, 3 * f_ind], X[:, 3 * f_ind + 1], yerr=X[:, 3 * f_ind + 2], fmt='x')
                    else:
                        y = plot_test[:, f_ind + 1]
                        ax.scatter(x, y, label=f)
                        ax.scatter(X[:, 0], X[:, f_ind + 1])
                    ax.legend()
                plt.suptitle(f'Epoch {epoch + 1}/{epochs}')
                plt.savefig(os.path.join(self.root, 'Train_plots', f'{epoch + 1}.png'))
                if plot_interval is not None:
                    if (epoch + 1) % plot_interval == 0:
                        plt.show()
                plt.close('all')

    def colour_analysis(self):
        # Estimate peak times for each SN
        # new_df['t'] = new_df['t'] * (self.scaling_factors[1] - self.scaling_factors[0]) + self.scaling_factors[0]
        # for f in ['g', 'r', 'i', 'z']:
        #     new_df[f] = new_df[f] * ((self.scaling_factors[3] - self.scaling_factors[2]) / 2) + \
        #                 np.mean([self.scaling_factors[3], self.scaling_factors[2]])
        all_new_df = None
        for sn in self.train_df.sn.unique():
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

        for f in [('g', 'r'), ('r', 'i'), ('i', 'z')]:
            f1, f2 = f
            all_new_df[f'{f1}-{f2}'] = all_new_df[f1] - all_new_df[f2]

        bins = pd.cut(all_new_df['t'].values, np.arange(all_new_df.t.min(), all_new_df.t.max(), 5))

        ts, cs, c_errs = [], [], []
        for i, cat in enumerate(bins.categories):
            tdf = all_new_df.iloc[np.argwhere(bins.codes == i).flatten(), :]
            c, c_err = tdf['g-r'].mean(), tdf['g-r'].std()
            ts.append(cat.mid)
            cs.append(c)
            c_errs.append(c_err)

        plt.errorbar(ts, cs, yerr=c_errs, fmt='x')
        plt.show()
        return

    def gen_lightcurves(self, n=10, length=20):
        # noise = rand.normal(size=(n, length, self.latent_dims))
        noise = rand.normal(size=(n, self.latent_dims))
        noise = np.reshape(noise, (n, 1, self.latent_dims))
        noise = np.repeat(noise, length, 1)
        self.generator.load_weights(self.generator_path)
        gen_lcs = self.generator.predict(noise)
        gen_lcs[:, :, 0] = gen_lcs[:, :, 0] * (self.scaling_factors[1] - self.scaling_factors[0]) + self.scaling_factors[0]
        gen_lcs[:, :, 1:] = gen_lcs[:, :, 1:] * ((self.scaling_factors[3] - self.scaling_factors[2]) / 2) + np.mean([self.scaling_factors[3], self.scaling_factors[2]])
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
