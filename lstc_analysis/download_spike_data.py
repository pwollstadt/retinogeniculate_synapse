"""Download and bin spike timing data."""
import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymatreader import read_mat

from utils import get_paths

N_PAIRS = 17
BIN_SIZE = 0.001  # used by us, 1 ms
MAX_LAG = 60

# Numbering corresponds to numbering in Rathbun2010
DATA_URL = 'https://github.com/scottiealexander/PairsDB.jl/raw/main/data/'
FILENAMES = {
    1: '20050803_107_msequence-000.mat',
    2: '20050811_101_msequence-000.mat',
    3: '20050811_102_msequence-000.mat',
    4: '20050811_103_msequence-000.mat',
    5: '20051228_108_msequence-000.mat',
    6: '20060119_109_msequence-000.mat',
    7: '20060119_104_msequence-000.mat',
    8: '20060125_110_msequence-000.mat',
    9: '20060223_111_msequence-000.mat',
    10: '20060316_105_msequence-000.mat',
    11: '20060316_106_msequence-000.mat',
    12: '20060417_112_msequence-000.mat',
    13: '20060417_113_msequence-000.mat',
    15: '20060720_115_msequence-000.mat',
    16: '20070215_116_msequence-000.mat',
    14: '20060417_114_msequence-000.mat',
    17: '20070418_117_msequence-000.mat'
    }


def crosscorr(x, y, max_lag):
    """Calculate cross-correlation at a given lag.

    Args:
        x (pandas.Series): Time series 1
        y (pandas.Series): Time series 2
        max_lag (int): Lag by which to shift time series

    Returns:
        numpy.ndarray: Cross-correlation up to maximum lag
    """
    return np.r_[
        np.correlate(x[:-(max_lag-1)], y),
        np.correlate(x[max_lag:], y)
        ]


datapath, _, _ = get_paths('paths.json')
temp_file = datapath.joinpath('temp_data.mat')

lags = np.arange(-MAX_LAG, MAX_LAG+1)

_, ax_all = plt.subplots(ncols=5, nrows=4, figsize=(10, 7))
ax_all[3, 2].axis('off')
ax_all[3, 3].axis('off')
ax_all[3, 4].axis('off')

for cell_pair, a in zip(range(1, N_PAIRS+1), ax_all.flatten()):

    r = requests.get(f'{DATA_URL}{FILENAMES[cell_pair]}', allow_redirects=True)
    open(temp_file, 'wb').write(r.content)
    spike_timings = read_mat(temp_file)
    print(f'\nSuccessfully loaded spike_timings for cell pair {cell_pair}')

    # Bin data and generate spike trains of equal length for both cells.
    last_spike = max(spike_timings['retina'][-1], spike_timings['lgn'][-1])
    lgn, _ = np.histogram(
        spike_timings['lgn'],
        bins=np.arange(0, last_spike + BIN_SIZE, BIN_SIZE)
    )
    rgc, _ = np.histogram(
        spike_timings['retina'],
        bins=np.arange(0, last_spike + BIN_SIZE, BIN_SIZE)
    )

    print('                 RGC\t     LGN')
    print('N spikes    {:8d}\t{:8d}'.format(
        len(spike_timings['retina']),
        len(spike_timings['lgn']))
        )
    print('length [ms] {:8d}\t{:8d}'.format(
        int(max(spike_timings['retina']*1000)),
        int(max(spike_timings['lgn']*1000)))
        )
    np.savez(
        datapath.joinpath(f'pair_{cell_pair:02d}_raw_data.npz'),
        rgc=rgc,
        lgn=lgn
        )

    # Plot cross-correlogram for current pair.
    xcorr = crosscorr(pd.Series(lgn), pd.Series(rgc), max_lag=MAX_LAG)
    a.bar(lags, xcorr, width=1.0)
    a.axvline(lags[np.argmax(xcorr)], linestyle=':', color='k')
    a.set(
        title=f'Pair {cell_pair}',
        xlim=(-MAX_LAG, MAX_LAG),
        ylabel='LGN spike count',
        xlabel='time [ms]'
        )

os.remove(temp_file)
plt.tight_layout()
plt.savefig(datapath.joinpath('cross_correlation.pdf'))
plt.show()
