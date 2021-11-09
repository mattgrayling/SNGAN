import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv(os.path.join('Data', 'Datasets', 'DES_CC_GP.csv'))

scaling_factors = {}

# Scale time--------------------------------------------------
max_t = df.t.max()
df['t'] = (df.t - max_t/2) / (max_t/2)
scaling_factors['t'] = max_t

# Scale photometry--------------------------------------------
for f in ['g', 'r', 'i', 'z']:
    df[f] = df[f] # * -1
    max1 = df[f].max()
    df[f] = df[f] - max1
    max2 = df[f].min()/2
    df[f] = (df[f] - max2) / max2
    scaling_factors[f] = [max1, max2]
    max_error = df[f'{f}_err'].max()
    df[f'{f}_err'] = (df[f'{f}_err'] - (max_error/2)) / (max_error/2)
    scaling_factors[f'{f}_err'] = max_error

# Apply cuts to remove bad sections of light curves-----------
cuts = {}
cuts['DES13C3gjd'] = (-1, -0.5)
cuts['DES13X1atuq'] = (-1, -0.6)

for sn, sn_cuts in cuts.items():
    temp_df = df[df.sn == sn]
    print(df.shape)
    df = df[df.sn != sn]
    temp_df = temp_df[(temp_df['t'] > sn_cuts[0]) & (temp_df['t'] < sn_cuts[1])]
    df = pd.concat([df, temp_df])
    print(df.shape)

# Drop bad objects-------------------------------------------
drop = ['DES15C1pkx']

print(df.shape)
df = df[~df.sn.isin(drop)]
print(df.shape)

df.to_csv(os.path.join('Data', 'Datasets', 'DES_CC_GP_scaled.csv'))
pickle.dump(scaling_factors, open(os.path.join('Data', 'Datasets', 'DES_CC_GP_scaling_factors.pkl'), 'wb'))
