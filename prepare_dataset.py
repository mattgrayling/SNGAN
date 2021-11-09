import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astroquery.irsa_dust import IrsaDust
import astropy.units as u
from astropy import coordinates
import extinction
import george
from george.kernels import Matern32Kernel
import scipy.optimize as opt
import pickle


# -------------------------------------------------------------------------------
'''
years = [(56500, 56750), (56850, 57100), (57200, 57450), (57600, 57850),
         (57900, 58200)]  # rough boundaries of each observing period in MJD
years_conv = {13: 0, 14: 1, 15: 2, 16: 3, 17: 4}
zp_ab = {'g': 20.802, 'r': 21.436, 'i': 21.866, 'z': 22.214}
exp_data = pd.read_pickle('Data/exp_date.pkl')

R_v = 3.1

des_df = pd.read_csv('Data/sn_max_data_all_err_H070_host_colour+mag_ext.csv')
des_df = des_df[des_df.rest_filter == 'R']

all_df = None

mode = 'GP'

for i, row in des_df.reset_index().iterrows():
    sn, snid, z = row.sn, row.sn_id, row.z_sn
    # if i < 5:
    #    continue
    filename = f'des_real_0{snid}.dat'
    data_df = pd.read_csv(os.path.join('Data', 'SN_Data', filename), skiprows=55, usecols=[1, 2, 4, 5],
                          delim_whitespace=True, header=None, names=['mjd', 'filt', 'flux', 'flux_err'], comment='#')
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
        (years[maxyear][0] < data_df.mjd) & (data_df.mjd < years[maxyear][1])]  # Select only for year that SN occurs

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
                        mjd_low = ind - 1
                        mjd_up = ind
                        pre_date = mjd[ind - 1]
                        exp_dates.append(exp_date)
                        break
                except:
                    break
        if exp_date is None:
            continue

    exp_date = np.min(exp_dates)

    # Get data epochs------------------------------------------------------------
    min_x, max_x = [], []
    all_times = []

    lens = []

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

        mjd = mjd[~np.isnan(mag)]
        flux = flux[~np.isnan(mag)]
        flux_err = flux_err[~np.isnan(mag)]
        mag_err = mag_err[~np.isnan(mag)]
        mag = mag[~np.isnan(mag)]

        x = (mjd - exp_date) / (1 + z)
        x = np.around(x, 1)
        min_x.append(np.min(x))
        max_x.append(np.max(x))
        for t in x:
            if t not in all_times:
                all_times.append(t)

    all_times = np.array(all_times)

    min_x, max_x = np.max(min_x), np.min(max_x)
    all_times = all_times[all_times >= min_x]
    all_times = all_times[all_times <= max_x]

    ind = 0
    change_dict = {}
    while True:
        if ind == len(all_times):
            break
        val = all_times[-(ind + 1)]
        if any(np.abs(all_times[all_times != val] - val) < 0.11):
            closest = all_times[all_times != val][np.argmin(np.abs(all_times[all_times != val] - val))]
            change_dict[val] = closest
            all_times = np.delete(all_times, -(ind + 1))
        else:
            ind += 1

    # Calculate MW extinction----------------------------------------------------
    co = coordinates.SkyCoord(ra=RA, dec=Dec, unit=(u.deg, u.deg))
    table = np.array(IrsaDust.get_query_table(co, section='ebv'))
    ebv = table[0][7]
    A_v = R_v * ebv

    wav = np.array([3556.52, 4702.50, 6175.58, 7489.98, 8946.71])
    ext = extinction.fm07(wav, A_v)

    data_df = pd.DataFrame(all_times, columns=['t'])

    # Select data----------------------------------------------------------------
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

        mjd = mjd[~np.isnan(mag)]
        flux = flux[~np.isnan(mag)]
        flux_err = flux_err[~np.isnan(mag)]
        mag_err = mag_err[~np.isnan(mag)]
        mag = mag[~np.isnan(mag)]

        x = (mjd - exp_date) / (1 + z)
        x = np.around(x, 1)

        flux = flux[(x >= min_x) & (x <= max_x)]
        flux_err = flux_err[(x >= min_x) & (x <= max_x)]
        mag = mag[(x >= min_x) & (x <= max_x)]
        mag_err = mag_err[(x >= min_x) & (x <= max_x)]
        x = x[(x >= min_x) & (x <= max_x)]

        # Fix times-------------------------------------------------------------
        if any(t not in all_times for t in x):
            new_x = []
            for t in x:
                if t in all_times:
                    new_x.append(t)
                else:
                    new_x.append(change_dict[t])
            x = np.array(new_x)

        # Remove any nans-------------------------------------------------------
        remove_inds = []
        for ind, val in enumerate(mag):
            if np.isnan(val):
                remove_inds.append(ind)
        for ind in reversed(remove_inds):
            x = np.delete(x, ind)
            flux = np.delete(flux, ind)
            flux_err = np.delete(flux_err, ind)
            mag = np.delete(mag, ind)
            mag_err = np.delete(mag_err, ind)

        # If mode == GP, GP all data--------------------------------------------
        if mode == 'GP':
            scale = np.max(flux)
            flux, flux_err = flux / scale, flux_err / scale

            gp = george.GP(Matern32Kernel(500))
            gp.compute(x, flux_err)
            p0 = gp.kernel.get_parameter_vector()[0]
            fl = flux  # Necessary for minimisation functions

            results = opt.minimize(ll, p0, jac=grad_ll)

            mu, cov = gp.predict(flux, all_times)
            std = np.sqrt(np.diag(cov))

            mu, std = mu * scale, std * scale

            mag = -2.5 * np.log10(mu) - zp_ab[f]
            mag_err = (2.5 * std / (mu * np.log(10)))
        else:
            # Fill gaps with GP if necessary----------------------------------------
            if len(x) != len(all_times):
                missing_times = np.array([t for t in all_times if t not in x])
                scale = np.max(flux)
                flux, flux_err = flux / scale, flux_err / scale

                gp = george.GP(Matern32Kernel(500))
                gp.compute(x, flux_err)
                p0 = gp.kernel.get_parameter_vector()[0]
                fl = flux  # Necessary for minimisation functions

                results = opt.minimize(ll, p0, jac=grad_ll)

                mu, cov = gp.predict(flux, missing_times)
                std = np.sqrt(np.diag(cov))

                mu, std = mu * scale, std * scale

                missing_mag = -2.5 * np.log10(mu) - zp_ab[f]
                missing_mag_err = (2.5 * std / (mu * np.log(10)))

                if any(np.isnan(val) for val in missing_mag):
                    print(all_times)
                    full_t = np.arange(min_x, max_x, 0.5)
                    full_mu, full_cov = gp.predict(flux, full_t)
                    full_std = np.sqrt(np.diag(full_cov))
                    full_mu, full_std = full_mu * scale, full_std * scale
                    plt.plot(full_t, full_mu)
                    plt.fill_between(full_t, full_mu-full_std, full_mu+full_std, alpha=0.3)
                    plt.errorbar(missing_times, mu, yerr=std, fmt='x')
                    plt.errorbar(x, flux*scale, yerr=flux_err*scale, fmt='x')
                    plt.title(f'{sn} {f}-band')
                    plt.show()

                new_mag, new_mag_err = [], []
                missing_ind = 0
                for t in all_times:
                    if t in x:
                        new_mag.append(mag[x == t][0])
                        new_mag_err.append(mag_err[x == t][0])
                    else:
                        new_mag.append(missing_mag[missing_ind])
                        new_mag_err.append(missing_mag_err[missing_ind])
                        missing_ind += 1
                mag, mag_err = np.array(new_mag), np.array(new_mag_err)
                x = all_times.copy()

        # Correct for MW extinction---------------------------------------------
        mag -= ext[f_ind]

        # Put into merged array-------------------------------------------------
        data_df[f] = mag
        data_df[f'{f}_err'] = mag_err
    print('---------')
    data_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_df = data_df.dropna()
    data_df['sn'] = sn
    data_df = data_df.sort_values(by='t')
    if all_df is None:
        all_df = data_df.copy()
    else:
        all_df = pd.concat([all_df, data_df])
    print(all_df.shape)

all_df.to_csv(os.path.join('Data', 'Datasets', 'DES_CC_GP.csv'))
'''


def prepare_dataset(mode='flux', GP=True, cadence=None, single_band=None):
    """
    Prepare dataset from real data based on input parameters
    :param mode: Specifies whether to use mag or flux data
    :param GP: Specifies whether to use original data or GP representation
    :param cadence: Specifies cadence in days to use for final observations across all objects, if None then original
    cadence will be used which will be different for each object
    :param single_band: If not None, will select a single specified band for inclusion in the data set
    :return:
    """

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
                              delim_whitespace=True, header=None, names=['mjd', 'filt', 'flux', 'flux_err'], comment='#')
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
            (years[maxyear][0] < data_df.mjd) & (data_df.mjd < years[maxyear][1])]  # Select only for year that SN occurs

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

    if cadence is not None:
        t = np.arange(x_lims[0], x_lims[1], cadence)

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
            if GP and cadence is not None:
                if mode == 'flux':
                    y, y_err = flux, flux_err
                elif mode == 'mag':
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

                results = opt.minimize(ll, p0, jac=grad_ll)

                mu, cov = gp.predict(y, t)
                std = np.sqrt(np.diag(cov))

                mu, std = mu * scale, std * scale

                if mode == 'flux':
                    for ind, val in enumerate(t):
                        if val < all_x_lims[sn][0] or val > all_x_lims[sn][1]:
                            mu[ind] = 0
                            std[ind] = 0
                    mu /= (max_fluxes[f] / 2)
                    mu -= 1
                elif mode == 'mag':
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
    dataset_name = f'DES_real_GP{GP}_{mode}_cadence{cadence}'
    scaling_factors = [x_lims, max_fluxes, mag_lims]
    all_df.to_csv(os.path.join('Data', 'Models', f'{dataset_name}.csv'))
    pickle.dump(open(os.path.join('Data', 'Models', f'{dataset_name}_scaling_factors.pkl'), 'wb'))
    return

    all_df = None

    for i, row in des_df.reset_index().iterrows():
        sn, snid, z = row.sn, row.sn_id, row.z_sn
        filename = f'des_real_0{snid}.dat'
        data_df = pd.read_csv(os.path.join('Data', 'SN_Data', filename), skiprows=55, usecols=[1, 2, 4, 5],
                              delim_whitespace=True, header=None, names=['mjd', 'filt', 'flux', 'flux_err'], comment='#')
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
            (years[maxyear][0] < data_df.mjd) & (data_df.mjd < years[maxyear][1])]  # Select only for year that SN occurs

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
                            mjd_low = ind - 1
                            mjd_up = ind
                            pre_date = mjd[ind - 1]
                            exp_dates.append(exp_date)
                            break
                    except:
                        break
            if exp_date is None:
                continue

        exp_date = np.min(exp_dates)

        # Get data epochs------------------------------------------------------------
        min_x, max_x = [], []
        all_times = []

        lens = []

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

            mjd = mjd[~np.isnan(mag)]
            flux = flux[~np.isnan(mag)]
            flux_err = flux_err[~np.isnan(mag)]
            mag_err = mag_err[~np.isnan(mag)]
            mag = mag[~np.isnan(mag)]

            x = (mjd - exp_date) / (1 + z)
            x = np.around(x, 1)
            min_x.append(np.min(x))
            max_x.append(np.max(x))
            all_max_fluxes[f].append(np.max(flux))
            for t in x:
                if t not in all_times:
                    all_times.append(t)

        all_times = np.array(all_times)

        min_x, max_x = np.max(min_x), np.min(max_x)
        all_times = all_times[all_times >= min_x]
        all_times = all_times[all_times <= max_x]

        ind = 0
        change_dict = {}
        while True:
            if ind == len(all_times):
                break
            val = all_times[-(ind + 1)]
            if any(np.abs(all_times[all_times != val] - val) < 0.11):
                closest = all_times[all_times != val][np.argmin(np.abs(all_times[all_times != val] - val))]
                change_dict[val] = closest
                all_times = np.delete(all_times, -(ind + 1))
            else:
                ind += 1

        # Calculate MW extinction----------------------------------------------------
        co = coordinates.SkyCoord(ra=RA, dec=Dec, unit=(u.deg, u.deg))
        table = np.array(IrsaDust.get_query_table(co, section='ebv'))
        ebv = table[0][7]
        A_v = R_v * ebv

        wav = np.array([3556.52, 4702.50, 6175.58, 7489.98, 8946.71])
        ext = extinction.fm07(wav, A_v)

        data_df = pd.DataFrame(all_times, columns=['t'])

        # Select data----------------------------------------------------------------
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

            mjd = mjd[~np.isnan(mag)]
            flux = flux[~np.isnan(mag)]
            flux_err = flux_err[~np.isnan(mag)]
            mag_err = mag_err[~np.isnan(mag)]
            mag = mag[~np.isnan(mag)]

            x = (mjd - exp_date) / (1 + z)
            x = np.around(x, 1)

            flux = flux[(x >= min_x) & (x <= max_x)]
            flux_err = flux_err[(x >= min_x) & (x <= max_x)]
            mag = mag[(x >= min_x) & (x <= max_x)]
            mag_err = mag_err[(x >= min_x) & (x <= max_x)]
            x = x[(x >= min_x) & (x <= max_x)]

            # Fix times-------------------------------------------------------------
            if any(t not in all_times for t in x):
                new_x = []
                for t in x:
                    if t in all_times:
                        new_x.append(t)
                    else:
                        new_x.append(change_dict[t])
                x = np.array(new_x)

            # Remove any nans-------------------------------------------------------
            remove_inds = []
            for ind, val in enumerate(mag):
                if np.isnan(val):
                    remove_inds.append(ind)
            for ind in reversed(remove_inds):
                x = np.delete(x, ind)
                flux = np.delete(flux, ind)
                flux_err = np.delete(flux_err, ind)
                mag = np.delete(mag, ind)
                mag_err = np.delete(mag_err, ind)

            # If mode == GP, GP all data--------------------------------------------
            if mode == 'GP':
                scale = np.max(flux)
                flux, flux_err = flux / scale, flux_err / scale

                gp = george.GP(Matern32Kernel(500))
                gp.compute(x, flux_err)
                p0 = gp.kernel.get_parameter_vector()[0]
                fl = flux  # Necessary for minimisation functions

                results = opt.minimize(ll, p0, jac=grad_ll)

                mu, cov = gp.predict(flux, all_times)
                std = np.sqrt(np.diag(cov))

                mu, std = mu * scale, std * scale

                mag = -2.5 * np.log10(mu) - zp_ab[f]
                mag_err = (2.5 * std / (mu * np.log(10)))
            else:
                # Fill gaps with GP if necessary----------------------------------------
                if len(x) != len(all_times):
                    missing_times = np.array([t for t in all_times if t not in x])
                    scale = np.max(flux)
                    flux, flux_err = flux / scale, flux_err / scale

                    gp = george.GP(Matern32Kernel(500))
                    gp.compute(x, flux_err)
                    p0 = gp.kernel.get_parameter_vector()[0]
                    fl = flux  # Necessary for minimisation functions

                    results = opt.minimize(ll, p0, jac=grad_ll)

                    mu, cov = gp.predict(flux, missing_times)
                    std = np.sqrt(np.diag(cov))

                    mu, std = mu * scale, std * scale

                    missing_mag = -2.5 * np.log10(mu) - zp_ab[f]
                    missing_mag_err = (2.5 * std / (mu * np.log(10)))

                    '''
                    if any(np.isnan(val) for val in missing_mag):
                        print(all_times)
                        full_t = np.arange(min_x, max_x, 0.5)
                        full_mu, full_cov = gp.predict(flux, full_t)
                        full_std = np.sqrt(np.diag(full_cov))
                        full_mu, full_std = full_mu * scale, full_std * scale
                        plt.plot(full_t, full_mu)
                        plt.fill_between(full_t, full_mu-full_std, full_mu+full_std, alpha=0.3)
                        plt.errorbar(missing_times, mu, yerr=std, fmt='x')
                        plt.errorbar(x, flux*scale, yerr=flux_err*scale, fmt='x')
                        plt.title(f'{sn} {f}-band')
                        plt.show()
                    '''

                    new_mag, new_mag_err = [], []
                    missing_ind = 0
                    for t in all_times:
                        if t in x:
                            new_mag.append(mag[x == t][0])
                            new_mag_err.append(mag_err[x == t][0])
                        else:
                            new_mag.append(missing_mag[missing_ind])
                            new_mag_err.append(missing_mag_err[missing_ind])
                            missing_ind += 1
                    mag, mag_err = np.array(new_mag), np.array(new_mag_err)
                    x = all_times.copy()

            # Correct for MW extinction---------------------------------------------
            mag -= ext[f_ind]

            # Put into merged array-------------------------------------------------
            data_df[f] = mag
            data_df[f'{f}_err'] = mag_err
        print('---------')
        data_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        data_df = data_df.dropna()
        data_df['sn'] = sn
        data_df = data_df.sort_values(by='t')
        if all_df is None:
            all_df = data_df.copy()
        else:
            all_df = pd.concat([all_df, data_df])
        print(all_df.shape)

    all_df.to_csv(os.path.join('Data', 'Datasets', 'DES_CC_GP.csv'))


prepare_dataset(mode='mag', GP=True, cadence=7, single_band='g')
