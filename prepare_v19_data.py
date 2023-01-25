import os
import numpy as np
import pandas as pd
import scipy.optimize as spopt
import pickle
import matplotlib as mpl
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['cmr10']})
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['mathtext.fontset'] = 'cm'
import matplotlib.pyplot as plt
from astropy.table import Table
from tqdm import tqdm, trange
import george
from george.kernels import Matern32Kernel

type_dict = {}

type_dict[1] = [834, 825, 840, 845, 819, 851]  # IIn
type_dict[2] = [830, 829, 822, 847, 865, 849, 850, 848, 832, 859, 804]  # IIb
type_dict[0] = [864, 866, 835, 844, 837, 838, 855, 863, 843, 861, 860, 858, 857, 856, 852, 839,
                801, 802, 867, 824, 808, 811, 831, 817]
type_dict[3] = [833, 807, 854, 813, 815, 816, 803, 827, 821, 842, 841, 828, 818]  # Ib
type_dict[4] = [846, 805, 862, 823, 814, 812, 810]  # Ic
type_dict[5] = [826, 806, 853, 836, 820, 809]  # Ic-BL


def prepare_data(sn_type, z_lim):
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
            if sn_type == 'Ia':
                head_df = head_df[head_df.SNTYPE == 101]
            else:
                head_df = head_df[head_df.SNTYPE == 120]
            sn_type += head_df.shape[0]
            head_df = head_df[head_df.HOSTGAL_SPECZ < z_lim]
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
                if row.SIM_TEMPLATE_INDEX in type_dict.keys():
                    sn_phot_df['sn_type'] = type_dict[row.SIM_TEMPLATE_INDEX]
                    types.append(type_dict[row.SIM_TEMPLATE_INDEX])
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
    min_mag = 27.428694
    all_df = all_df[all_df.mag < min_mag]  # 99th percentile

    #min_t, max_t = all_df.t.min(), all_df.t.max()
    #min_mag, max_mag = all_df.mag.max(), all_df.mag.min()
    #all_df = all_df[all_df.mag_err < 1]  # Remove points with magnitude uncertainties greater than 1 mag
    #scaled_mag = (all_df['mag'] - min_mag) / (max_mag - min_mag)
    #all_df['mag'] = scaled_mag
    #all_df['mag_err'] /= np.abs(max_mag - min_mag)
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
            t = fdf.t
            new_sn_df[f'{f}_t'] = t.values
            sn_t_min, sn_t_max = np.min(t), np.max(t)
            y = fdf.mag
            y_err = fdf.mag_err

            scale = y.max()

            fit_y, fit_y_err = y / scale, y_err / scale

            # Gradient function for optimisation of kernel size------------------------------

            def ll(p):
                gp.set_parameter_vector(p)
                return -gp.log_likelihood(y, quiet=True)

            def grad_ll(p):
                gp.set_parameter_vector(p)
                return -gp.grad_log_likelihood(y, quiet=True)

            try:
                gp.compute(t, fit_y_err)
                p0 = gp.kernel.get_parameter_vector()[0]
                results = spopt.minimize(ll, p0, jac=grad_ll)
                mu, cov = gp.predict(fit_y, t)
                std = np.sqrt(np.diag(cov))
            except:
                if not skip:
                    gp_error += 1
                skip = True
                continue
            test_t, test_mu = t[mu < min_mag], mu[mu < min_mag]
            if len(test_mu) == 0:
                if not skip:
                    no_points += 1
                skip = True
                continue
            if np.argmin(test_mu) == 0 or (np.argmax(test_mu) == 0 and np.argmin(test_mu) == len(test_mu) - 1) \
                    or len(mu[mu < min_mag]) < 8:
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
            new_sn_df = new_sn_df.loc[np.max(new_sn_df[['g', 'r', 'i', 'z']].values, axis=1) <= min_mag, :]
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

    return
    dataset_name = f'DES_sim_colourFalse_GPTrue_zlim{z_lim}_redshift'
    dataset_path = os.path.join('Data', 'Datasets', f'{dataset_name}.csv')
    all_new_df.to_csv(dataset_path, index=False)


prepare_data('Ic', 0.1)
