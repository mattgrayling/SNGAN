import os

import matplotlib.pyplot as plt
from sncosmo import read_snana_ascii

for file in os.listdir(os.path.join('Data', 'YSE', 'SNIbc_sample')):
    meta, lcdata = read_snana_ascii(os.path.join('Data', 'YSE', 'SNIbc_sample', file), default_tablename='OBS')
    data = lcdata['OBS'].to_pandas()
    for filt in ['g', 'r', 'i']:
        filt_df = data[data.FLT == filt]
        plt.errorbar(filt_df.MJD, filt_df.MAG, yerr=filt_df.MAGERR, fmt='x')
        plt.gca().invert_yaxis()
    plt.title(f'{file[:file.find(".")]}: {meta["NOBS_BEFORE_PEAK"]}, {meta["NOBS_AFTER_PEAK"]}')
    plt.show()
