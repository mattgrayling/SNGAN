import os

import numpy as np
import scipy.optimize as spopt
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
import george
from george.kernels import Matern32Kernel
import pickle
from tqdm import tqdm

'''
tbl = Table.read('Data/DESSIMBIAS5YRCC_V19/PIP_MV_GLOBAL_BIASCOR_DESSIMBIAS5YRCC_V19'
                 '/PIP_MV_GLOBAL_BIASCOR_DES9a92f_NONIaMODEL0-0001_HEAD.FITS.gz', hdu=1)
df = tbl.to_pandas()

for col, dtype in df.dtypes.items():
    if dtype == object:  # Only process byte object columns.
        df[col] = df[col].apply(lambda x: x.decode("utf-8").lstrip().rstrip())
'''

data_dir = 'Data/DESSIMBIAS5YRCC_V19/PIP_MV_GLOBAL_BIASCOR_DESSIMBIAS5YRCC_V19'

x = os.listdir(data_dir)
x.sort()

total = 0
all_df = None

for file in x:
    if 'HEAD' in file:
        head_file = file
        phot_file = file.replace('HEAD', 'PHOT')
        head = Table.read(os.path.join(data_dir, head_file), hdu=1)
        head_df = head.to_pandas()
        head_df = head_df[head_df.HOSTGAL_SPECZ < 0.1]
        head_df = head_df[head_df.NOBS > 20]
        for col, dtype in head_df.dtypes.items():
            if dtype == object:  # Only process byte object columns
                head_df[col] = head_df[col].apply(lambda x: x.decode("utf-8").lstrip().rstrip())
        phot = Table.read(os.path.join(data_dir, phot_file), hdu=1)
        phot_df = phot.to_pandas()
        for col, dtype in phot_df.dtypes.items():
            if dtype == object:  # Only process byte object columns.
                phot_df[col] = phot_df[col].apply(lambda x: x.decode("utf-8").lstrip().rstrip())
        phot_df['mag'] = -2.5 * np.log10(phot_df.FLUXCAL) + 27
        phot_df['mag_err'] = (2.5 / np.log(10)) * phot_df.FLUXCALERR / phot_df.FLUXCAL
        total += head_df.shape[0]
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
            '''
                fdf = sn_phot_df[sn_phot_df.FLT == f]
                plt.errorbar(fdf.t, fdf.mag, yerr=fdf.mag_err, fmt='x', label=f)
            plt.legend()
            # plt.vlines(row.PEAKMJD, -100, 100, ls='-')
            # plt.vlines(row.SIM_PEAKMJD, -100, 100, ls='--')
            plt.ylim(16, 30)
            plt.gca().invert_yaxis()
            plt.show()
            '''
    else:
        continue

cadence = 7

all_df = all_df[~(np.isnan(all_df.mag) | np.isinf(all_df.mag))]

min_t, max_t = all_df.t.min(), all_df.t.max()
min_mag, max_mag = all_df.mag.max(), all_df.mag.min()
all_t = (np.arange(min_t, max_t, cadence) - min_t) / (max_t - min_t)
all_df['t'] = (all_df['t'] - min_t) / (max_t - min_t)
scaled_mag = all_df['mag'].values - np.mean([min_mag, max_mag])
scaled_mag /= (max_mag - min_mag) / 2
all_df['mag'] = scaled_mag
all_df['mag_err'] /= (max_mag - min_mag) / 2

all_new_df = None

for i, sn in tqdm(enumerate(all_df.sn.unique())):
    sn_df = all_df[all_df.sn == sn]
    new_sn_df = pd.DataFrame(all_t, columns=['t'])
    for f in ['g', 'r', 'i', 'z']:
        fdf = sn_df[sn_df.FLT == f]
        gp = george.GP(Matern32Kernel(500))

        t = fdf.t
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
            mu, cov = gp.predict(y, all_t)
            std = np.sqrt(np.diag(cov))
        except:
            continue

        mu[(all_t < sn_t_min) | (all_t > sn_t_max)] = 0
        std[(all_t < sn_t_min) | (all_t > sn_t_max)] = 0

        new_sn_df[f] = mu
        new_sn_df[f'{f}_err'] = std
    new_sn_df['sn'] = sn
    if all_new_df is None:
        all_new_df = new_sn_df.copy()
    else:
        all_new_df = pd.concat([all_new_df, new_sn_df])

new_sn_df.to_csv('Data/Datasets/V19z0.1.csv')
pickle.dump([min_t, max_t, min_mag, max_mag], open('Data/Datasets/V19z0.1lims.pkl', 'wb'))
