import os

import numpy as np
import pandas as pd
import scipy.optimize as spopt
import pickle
import numpy.random as rand
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


class Critic(Model):
    def __init__(self, units=50, dropout=0.5, n_output=12):
        super(Critic, self).__init__()
        self.gru1 = GRU(units, return_sequences=True)
        self.dr1 = Dropout(dropout)
        self.gru2 = GRU(units, return_sequences=True)
        self.dr2 = Dropout(dropout)
        self.gru3 = GRU(1, activation=None)

    def call(self, inputs, training=False):
        x = self.gru1(inputs)
        if training:
            x = self.dr1(x, training=training)
        x = self.gru2(x)
        if training:
            x = self.dr2(x, training=training)
        return self.gru3(x)


class Generator(Model):
    def __init__(self, latent_dims=10, units=100, dropout=0.5, n_output=12):
        super(Generator, self).__init__()
        self.gru1 = GRU(units, return_sequences=True, activation='relu')
        self.dr1 = Dropout(dropout)
        self.gru2 = GRU(units, return_sequences=True, activation='relu')
        self.dr2 = Dropout(dropout)
        self.gru3 = GRU(n_output, return_sequences=True, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.gru1(inputs)
        if training:
            x = self.dr1(x, training=training)
        x = self.gru2(x)
        if training:
            x = self.dr2(x, training=training)
        return self.gru3(x)


class WGAN:
    def __init__(self, latent_dims=100, lr=0.0005, device='gpu:0', GP=True, data_type='sim',
                 z_lim=None, batch_norm=False, mode='template', dropout=0.5, gen_units=100, crit_units=100,
                 sn_type='II'):
        self.latent_dims = latent_dims
        self.lr = lr
        self.device = device
        self.GP = GP
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
        self.data_type = data_type
        self.z_lim = z_lim
        self.batch_norm = batch_norm
        self.mode = mode
        if self.mode.lower() not in ['template', 'observed']:
            raise ValueError('mode must be one of template and observed')
        self.dropout = dropout
        self.gen_units = gen_units
        self.crit_units = crit_units
        # WGAN Paper guidance-----------------------------
        self.n_critic = 1
        self.clip_value = 0.01
        self.optimizer = opt.RMSprop(lr=self.lr)
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
        if self.data_type == 'sim':
            self.name = f'WGAN_DES_sim_{sn_type}_CCSNe_{self.mode}_lr{self.lr}_ld{self.latent_dims}_GP{self.GP}' \
                        f'_zlim{self.z_lim}_bn{self.batch_norm}_gN{self.gen_units}_cN{self.crit_units}'
        else:
            raise ValueError('Not implemented for real data')
        self.root = os.path.join('Data', 'Models', 'WGAN', self.mode, self.name)
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        self.checkpoint_dir = os.path.join(self.root, 'checkpoint')
        self.prev_epochs = 0
        if os.path.exists(os.path.join(self.root, 'Train_plots')) and \
                len(os.listdir(os.path.join(self.root, 'Train_plots'))) > 0:
            self.prev_epochs = np.max([int(val.split('.')[0]) for val in
                                       os.listdir(os.path.join(self.root, 'Train_plots'))])
        if self.data_type == 'real':
            self.dataset_name = f'WGAN_DES_real_GP{self.GP}'
            self.dataset_path = os.path.join('Data', 'Datasets', f'{self.dataset_name}.csv')
        elif self.data_type == 'sim':
            self.dataset_name = f'WGAN_DES_sim_{self.mode}_GP{self.GP}_zlim{self.z_lim}'
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
        self.train_df = self.train_df[self.train_df.sn_type == self.class_label_encoder[sn_type]]
        self.generator_dir = os.path.join(self.root, 'model_weights')

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

            types = []
            total, sn_type, z_cut, n_obs = 0, 0, 0, 0

            print('Reading in files...')
            for file in tqdm(x):
                if 'HEAD' in file:
                    head_file = file
                    phot_file = file.replace('HEAD', 'PHOT')
                    head = Table.read(os.path.join(data_dir, head_file), hdu=1)
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
            min_t, max_t = all_df.t.min(), all_df.t.max()
            min_mag, max_mag = all_df.mag.max(), all_df.mag.min()
            all_df = all_df[all_df.mag_err < 1]  # Remove points with magnitude uncertainties greater than 1 mag
            scaled_mag = all_df['mag'].values - np.mean([min_mag, max_mag])
            scaled_mag /= (max_mag - min_mag) / 2
            all_df['mag'] = scaled_mag
            all_df['mag_err'] /= np.abs(max_mag - min_mag) / 2
            all_df = all_df[~(all_df.mag < 0.1)]  # Remove bad points # & (all_df.mag_err > 0.5))]

            all_new_df = None
            used_count, skip_count, gp_error, no_peak, no_points, nans = 0, 0, 0, 0, 0, 0

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
                        if self.mode == 'template':
                            mu, cov = gp.predict(y, all_t)
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
                if skip:
                    skip_count += 1
                    continue
                used_count += 1
                if all_new_df is None:
                    all_new_df = new_sn_df.copy()
                else:
                    all_new_df = pd.concat([all_new_df, new_sn_df])
            # print(total, sn_type, z_cut, n_obs)
            # print(used_count, skip_count, gp_error + no_peak + no_points + nans, gp_error, no_peak, no_points, nans)
            all_new_df.to_csv(self.dataset_path)
            scaling_factors = [min_t, max_t, min_mag, max_mag]
            pickle.dump(scaling_factors,
                        open(os.path.join('Data', 'Datasets', f'{self.dataset_name}_scaling_factors.pkl'), 'wb'))
            return all_new_df, scaling_factors

    def wasserstein_loss(self, y_true, y_pred):
        return tf.reduce_mean(y_true * y_pred)

    def build_generator(self):
        with tf.device(self.device):
            input = Input(shape=(None, self.latent_dims))
            gru1 = GRU(self.gen_units, activation='relu', return_sequences=True)(input)
            dr1 = Dropout(self.dropout)(gru1)
            gru2 = GRU(self.gen_units, activation='relu', return_sequences=True)(dr1)
            dr2 = Dropout(self.dropout)(gru2)
            output = GRU(self.n_output, return_sequences=True, activation='sigmoid')(dr2)
            model = Model(input, output)
            return model

    def build_critic(self):
        with tf.device(self.device):
            input = Input(shape=(None, self.n_output))
            gru1 = GRU(self.crit_units, return_sequences=True)(input)
            dr1 = Dropout(self.dropout)(gru1)
            if self.batch_norm:
                bn1 = BatchNormalization()(gru1)
                gru2 = GRU(self.crit_units, return_sequences=True)(bn1)
                bn2 = BatchNormalization()(gru2)
                output = GRU(1, activation=None)(bn2)
            else:
                gru2 = GRU(self.crit_units, return_sequences=True)(dr1)
                dr2 = Dropout(self.dropout)(gru2)
                output = GRU(1, activation=None)(dr2)
            model = Model(input, output)
            return model

    def plot_train_sample(self):
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

    # @tf.function
    def train_step(self, X, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            noise = tf.random.normal((1, self.latent_dims))
            noise = tf.reshape(noise, (1, 1, self.latent_dims))
            noise = tf.repeat(noise, X.shape[1], 1)

            real_data_real_label = self.critic.call(X, training=True)
            gen_lcs = self.generator(noise, training=True)
            fake_data_fake_label = self.critic(gen_lcs, training=True)

            noise = tf.random.normal((2, self.latent_dims))
            noise = tf.reshape(noise, (2, 1, self.latent_dims))
            noise = tf.repeat(noise, X.shape[1], 1)

            gen_lcs = self.generator(noise)
            fake_data_real_label = self.critic(gen_lcs, training=True)

            # Calculate losses---------------------------------------------------
            gen_loss = self.wasserstein_loss(tf.fill(tf.shape(fake_data_real_label), -1.), fake_data_real_label)
            crit_loss = 0.5 * (self.wasserstein_loss(tf.fill(tf.shape(real_data_real_label), -1.), real_data_real_label) + \
                self.wasserstein_loss(tf.ones_like(fake_data_fake_label), fake_data_fake_label))
        # print('Generator: ', fake_data_real_label, gen_loss)
        # print('Critic: ', fake_data_fake_label, real_data_real_label, crit_loss)
        crit_gradients = disc_tape.gradient(crit_loss, self.critic.trainable_variables)
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.c_optimizer.apply_gradients(zip(crit_gradients, self.critic.trainable_variables))
        self.g_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        return real_data_real_label.numpy().flatten()[0], fake_data_fake_label.numpy().flatten()[0], \
            gen_loss.numpy(), crit_loss.numpy()

    def train(self, epochs=100, batch_size=1, plot_interval=None):
        print('Starting training...')
        if not os.path.exists(os.path.join(self.root, 'Train_plots')):
            os.mkdir(os.path.join(self.root, 'Train_plots'))
        with tf.device(self.device):
            rng = np.random.default_rng(123)

            sne = self.train_df.sn.unique()
            n_batches = int(len(sne) / batch_size)

            # Build models------------------------------------------------------------------
            self.c_optimizer = opt.RMSprop(lr=self.lr)
            self.g_optimizer = opt.RMSprop(lr=self.lr)

            # self.critic = self.build_critic()
            self.critic = Critic(units=self.crit_units, dropout=self.dropout, n_output=self.n_output)

            # Build generator
            # self.generator = self.build_generator()
            self.generator = Generator(self.latent_dims, self.gen_units, self.dropout, self.n_output)
            # self.critic = critic.build((1, 20, self.n_output))
            # print(type(critic), type(self.critic))
            # print(help(critic.build))
            # ------------------------------------------------------------------------------

            checkpoint = tf.train.Checkpoint(g_optimizer=self.g_optimizer,
                                             c_optimizer=self.c_optimizer,
                                             generator=self.generator,
                                             critic=self.critic
                                             )
            if not os.path.exists(self.checkpoint_dir):
                os.mkdir(self.checkpoint_dir)
            ckpt_manager = tf.train.CheckpointManager(checkpoint, self.checkpoint_dir, max_to_keep=3)
            if ckpt_manager.latest_checkpoint:
                print(f'Models already exist, resuming training from epoch {self.prev_epochs + 1}...')
                checkpoint.restore(ckpt_manager.latest_checkpoint)
                # old_generator_weights = pickle.load(open(os.path.join(self.root, 'generator_weights.pkl'), 'rb'))
                # old_critic_weights = pickle.load(open(os.path.join(self.root, 'critic_weights.pkl'), 'rb'))
                # print(old_generator_weights[0] == self.generator.get_weights()[0])
                # print(old_critic_weights[0] == self.critic.get_weights()[0])
                # raise ValueError('Nope')


            real = -np.ones((batch_size, 1))
            fake = np.ones((batch_size, 1))

            for epoch in range(self.prev_epochs, epochs):
                rng.shuffle(sne)
                g_losses, c_losses, real_predictions, fake_predictions = [], [], [], []
                t = trange(n_batches)
                for batch in t:
                    sn = sne[batch]
                    sndf = self.train_df[self.train_df.sn == sn]
                    if self.mode == 'observed':
                        X = sndf[['g_t', 'r_t', 'i_t', 'z_t', 'g', 'r', 'i', 'z', 'g_err',
                                  'r_err', 'i_err', 'z_err']].values
                    elif self.mode == 'template':
                        X = sndf[['t', 'g', 'r', 'i', 'z']].values

                    X_np = X.copy()
                    X = tf.convert_to_tensor(X.reshape((1, *X.shape)))
                    real_pred, fake_pred, g_loss, c_loss = self.train_step(X, epoch)

                    g_losses.append(g_loss)
                    c_losses.append(c_loss)
                    real_predictions.append(real_pred)
                    fake_predictions.append(fake_pred)
                    t.set_description(f'g_loss={np.around(np.mean(g_losses), 5)},'
                                      f' d_loss={np.around(np.mean(c_losses), 5)}')
                    t.refresh()
                    '''
                    # Select real data
                    sn = sne[batch]
                    sndf = self.train_df[self.train_df.sn == sn]
                    if self.mode == 'observed':
                        X = sndf[['g_t', 'r_t', 'i_t', 'z_t', 'g', 'r', 'i', 'z', 'g_err',
                                  'r_err', 'i_err', 'z_err']].values
                    elif self.mode == 'template':
                        X = sndf[['t', 'g', 'r', 'i', 'z']].values

                    X = X.reshape((1, *X.shape))

                    if np.count_nonzero(np.isnan(X)) > 0:
                        continue

                    noise = rand.normal(size=(batch_size, self.latent_dims))
                    noise = np.reshape(noise, (batch_size, 1, self.latent_dims))
                    noise = np.repeat(noise, X.shape[1], 1)

                    test_gen_lcs = self.generator.predict(noise)
                    if np.count_nonzero(np.isnan(test_gen_lcs)) > 0:
                        raise ValueError('NaN generated, check how this happened')
                    gen_lcs = test_gen_lcs
                    real_prediction = self.critic.predict(X)
                    real_predictions.append(real_prediction.flatten()[0])
                    fake_prediction = self.critic.predict(gen_lcs)
                    fake_predictions.append(fake_prediction.flatten()[0])

                    # Train discriminator
                    d_loss_real = self.critic.train_on_batch(X, real)
                    d_loss_fake = self.critic.train_on_batch(gen_lcs, fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                    # for l in self.critic.layers:
                    #    weights = l.get_weights()
                    #    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    #    l.set_weights(weights)

                    if batch % self.n_critic == 0:
                        # Train generator
                        noise = rand.normal(size=(2 * batch_size, self.latent_dims))
                        noise = np.reshape(noise, (2 * batch_size, 1, self.latent_dims))
                        noise = np.repeat(noise, X.shape[1], 1)

                        gen_labels = -np.ones((2 * batch_size, 1))
                        g_loss = self.combined.train_on_batch(noise, gen_labels)
                        g_losses.append(g_loss)
                    d_losses.append(d_loss)
                    t.set_description(f'g_loss={np.around(np.mean(g_losses), 5)},'
                                      f' d_loss={np.around(np.mean(d_losses), 5)}')
                    t.refresh()
                    '''

                '''
                if os.path.exists(os.path.join(self.root, 'generator_weights.pkl')):
                    os.remove(os.path.join(self.root, 'generator_weights.pkl'))
                pickle.dump(self.generator.get_weights(), open(os.path.join(self.root, 'generator_weights.pkl'), 'wb'))
                if os.path.exists(os.path.join(self.root, 'critic_weights.pkl')):
                    os.remove(os.path.join(self.root, 'critic_weights.pkl'))
                pickle.dump(self.critic.get_weights(), open(os.path.join(self.root, 'critic_weights.pkl'), 'wb'))
                '''

                ckpt_manager.save()
                full_g_loss = np.mean(g_losses)
                full_c_loss = np.mean(c_losses)
                print(f'{epoch + 1}/{epochs} g_loss={full_g_loss}, c_loss={full_c_loss}, '
                      f'Real prediction: {np.mean(real_predictions)} +- {np.std(real_predictions)}, '
                      f'Fake prediction: {np.mean(fake_predictions)} +- {np.std(fake_predictions)}')
                # f' Ranges: x [{np.min(gen_lcs[:, :, 0])}, {np.max(gen_lcs[:, :, 0])}], '
                # f'y [{np.min(gen_lcs[:, :, 1])}, {np.max(gen_lcs[:, :, 1])}]')

                noise = rand.normal(size=(batch_size, self.latent_dims))
                noise = np.reshape(noise, (batch_size, 1, self.latent_dims))
                noise = np.repeat(noise, X.shape[1], 1)

                gen_lcs = self.generator.predict(noise)
                if np.count_nonzero(np.isnan(gen_lcs)) > 0:
                    raise ValueError('NaN generated, check how this happened')

                plot_test = gen_lcs[0, :, :]
                fig = plt.figure(figsize=(12, 8))
                x = plot_test[:, 0]
                for f_ind, f in enumerate(['g', 'r', 'i', 'z']):
                    ax = fig.add_subplot(2, 2, f_ind + 1)
                    if self.mode == 'observed':
                        x, y, y_err = plot_test[:, f_ind], plot_test[:, f_ind + 4], plot_test[:, f_ind + 8]
                        ax.errorbar(x, y, yerr=y_err, label=f, fmt='x')
                        ax.errorbar(X_np[:, f_ind], X_np[:, f_ind + 4], yerr=X_np[:, f_ind + 8], fmt='x')
                    elif self.mode == 'template':
                        y = plot_test[:, f_ind + 1]
                        ax.scatter(x, y, label=f)
                        ax.scatter(X_np[:, 0], X_np[:, f_ind + 1])
                    ax.legend()
                plt.suptitle(f'Epoch {epoch + 1}/{epochs}')  #: Type {self.class_label_dict[sn_type]}')
                plt.savefig(os.path.join(self.root, 'Train_plots', f'{epoch + 1}.png'))
                if plot_interval is not None:
                    if (epoch + 1) % plot_interval == 0:
                        plt.show()
                plt.close('all')

    def colour_analysis(self, epoch=-1, n=1):
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
