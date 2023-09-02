#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calculate average absolute and relative bias correction for estimates"""
import copy as cp

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from utils import get_paths

datapath, resultspath, figurepath = get_paths('paths.json')

font = {# 'family' : 'normal',
        # 'weight' : 'bold',
        'size': 6}

matplotlib.rc('font', **font)

# List of pairs with significant TE/AIS estimates.
all_pairs = [i for i in range(1, 18)]
all_pairs.remove(5)
N = len(all_pairs)

all_stats_abs = {
    'lais_mean': [],
    'lais_sd': [],
    'lais_min': [],
    'lais_max': [],
    'lte_mean': [],
    'lte_sd': [],
    'lte_min': [],
    'lte_max': [],
}
all_stats_frac = cp.deepcopy(all_stats_abs)

fig, ax = plt.subplots(ncols=4, nrows=len(all_pairs), figsize=(7, len(all_pairs)*0.7))
for i, n_pair in enumerate(all_pairs):
    lte_uncor = np.load(resultspath.joinpath('pair_{0:02d}_lte.npy'.format(n_pair)))
    lais_uncor = np.load(resultspath.joinpath('pair_{0:02d}_lais.npy'.format(n_pair)))

    lte_cor = np.load(resultspath.joinpath(f'pair_{n_pair:02d}_lte_corrected.npy'))
    lais_cor = np.load(resultspath.joinpath(f'pair_{n_pair:02d}_lais_corrected.npy'))

    lte_bias_correction = lte_cor - lte_uncor
    lais_bias_correction = lais_cor - lais_uncor
    lte_bias_correction_frac = np.divide(
        lte_bias_correction, lte_uncor, out=np.zeros_like(lte_bias_correction), where=lte_uncor!=0
    )
    lais_bias_correction_frac = np.divide(
        lais_bias_correction, lais_uncor, out=np.zeros_like(lais_bias_correction), where=lais_uncor!=0
    )

    def _print_stats(a, measure):
        print(
            f'\t{measure}: mean: {np.mean(a):.6f} '
            f'SD: {np.std(a, ddof=1):.6f} '
            f'min: {np.min(a):.6f} '
            f'max: {np.max(a):.6f}'
        )
        if '%' in measure:
            all_stats_frac[f'{measure[:4].lower().strip()}_mean'].append(a.mean())
            all_stats_frac[f'{measure[:4].lower().strip()}_sd'].append(a.std(ddof=1))
            all_stats_frac[f'{measure[:4].lower().strip()}_min'].append(a.min())
            all_stats_frac[f'{measure[:4].lower().strip()}_max'].append(a.max())
        else:
            all_stats_abs[f'{measure[:4].lower().strip()}_mean'].append(a.mean())
            all_stats_abs[f'{measure[:4].lower().strip()}_sd'].append(a.std(ddof=1))
            all_stats_abs[f'{measure[:4].lower().strip()}_min'].append(a.min())
            all_stats_abs[f'{measure[:4].lower().strip()}_max'].append(a.max())

    print(f'\nPair {n_pair} bias correction')
    _print_stats(lais_bias_correction, 'lAIS')
    _print_stats(lte_bias_correction, 'lTE ')
    _print_stats(lais_bias_correction_frac, 'lAIS (%)')
    _print_stats(lte_bias_correction_frac, 'lTE  (%)')

    ax[i, 0].hist(lais_bias_correction, bins=20, log=True, color='b')
    ax[i, 0].set(ylabel=f'Pair {n_pair}')
    ax[i, 1].hist(lte_bias_correction, bins=20, log=True, color='r')
    ax[i, 2].hist(lais_bias_correction_frac, bins=20, log=True, color='b')
    ax[i, 3].hist(lte_bias_correction_frac, bins=20, log=True, color='r')

ax[-1, 0].set(xlabel='lAIS')
ax[-1, 1].set(xlabel='lTE')
ax[-1, 2].set(xlabel='lAIS [%]')
ax[-1, 3].set(xlabel='lTE [%]')

plt.tight_layout()
plt.savefig(figurepath.joinpath('bias_correction_magnitude.pdf'))

df_stats_abs = pd.DataFrame(all_stats_abs, index=all_pairs)
df_stats_frac = pd.DataFrame(all_stats_frac, index=all_pairs)
print('\n\nAbsolute bias corrections')
print(df_stats_abs)
print(df_stats_abs.describe())
print('\n\nRelative bias corrections')
print(df_stats_frac)
print(df_stats_frac.describe())
df_stats_abs.to_csv(resultspath.joinpath('bias_correction_magnitude_abs.csv'))
df_stats_frac.to_csv(resultspath.joinpath('bias_correction_magnitude_frac.csv'))

plt.show()
