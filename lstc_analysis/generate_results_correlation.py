#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Genrerate local storage-transfer correlations (LSTC) for each cell pair.

Load locally PT-corrected estimates of LAIS and LTE and calculate the
Pearson correlation.
"""
import sys
import pickle
import itertools
import copy as cp
import numpy as np
import scipy.stats as sc
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from idtxl.data import Data

from utils import load_spike_trains, get_paths, load_mte_ais_estimates, load_local_estimates

# Create a spinner as a busy symbol when performing permutation tests
spinner = itertools.cycle(['-', '\\', '|', '/'])

datapath, resultspath, figurepath = get_paths('paths.json')

for n_pair in range(1, 18):

    print(f'Calculating LSTC for pair {n_pair}')

    lte, lais, delay = load_local_estimates(resultspath, n_pair)
    mte, ais = load_mte_ais_estimates(resultspath, n_pair)
    if lte is None or lais is None:
        print('No estimates for current spike pair')
        continue

    # Load words used for LAIS/LTE estimation
    rgc, lgn = load_spike_trains(datapath, n_pair)
    data = Data(np.vstack((rgc, lgn)), 'ps', normalise=False)
    words = np.load(resultspath.joinpath(f'pair_{n_pair:02d}_lte_words.npz'))
    te_source_past = words['source_past']
    te_target_past = words['target_past']
    te_current_value = data.get_realisations(
        current_value=mte.get_single_target(1, False).current_value,
        idx_list=[mte.get_single_target(1, False).current_value])[0]
    words = np.load(resultspath.joinpath(f'pair_{n_pair:02d}_lais_words.npz'))
    ais_source_past = words['source_past']
    ais_current_value = words['current_value_realisations']

    # Correct for differences in embedding lengths (max_lag of 40 ms for TE and a max_lag of 30 for AIS. Also account
    # for the delay between LAIS and LTE, i.e, the sample in the RGC that is most informative about the current value of
    # the LGN is in the past of the current value due to an information transfer delay. Also account for different max
    # lags in the LTE and LAIS estimation.
    embedding_diff = (mte.get_single_target(1, False).current_value[1] -
                      ais.get_single_process(0, False).current_value[1])
    lais = lais[embedding_diff:-delay]
    lte = lte[delay:]
    ais_source_past = ais_source_past[embedding_diff:-delay]
    ais_current_value = ais_current_value[embedding_diff:-delay]
    te_source_past = te_source_past[delay:]
    te_target_past = te_target_past[delay:]
    te_current_value = te_current_value[delay:]
    rgc_spike_ind = np.squeeze(ais_current_value == 1)
    lgn_spike_ind = np.squeeze(te_current_value == 1)
    relayed_ind = np.squeeze(np.logical_and(
        te_current_value == 1, ais_current_value == 1))

    # Save preprocessed LAIS and LTE values to disk to create histograms and visualizations for the paper.
    savepath = resultspath.joinpath(f'pair_{n_pair:02d}_hist_data')
    print(f'Saving histogram data to\n{savepath}')
    np.savez(
        savepath,
        lais=lais,
        lte=lte,
        rgc_spike_ind=rgc_spike_ind,
        lgn_spike_ind=lgn_spike_ind
        )

    # Calculate correlation coefficients.
    c = np.corrcoef(lais, lte)
    r = sc.spearmanr(lais, lte)
    corr = {'corrcoef': c[0, 1],
            'spearmanr': r[0],
            'n_permutations': 1000}
    print(f'Delay: {delay}, pearson corr: {c[0, 1]:.4f}, spearman r: {r[0]:.4f}')

    # Perform permutation test
    perm_dist_c = np.zeros(corr['n_permutations'])
    perm_dist_r = np.zeros(corr['n_permutations'])
    lais_surrogate = cp.copy(lais)  # make a copy to use numpy's shuffling
    print('Perform permutation test with {0} surrogates'.format(
        corr['n_permutations']))
    for p in range(corr['n_permutations']):
        # Shuffle LAIS in place
        np.random.shuffle(lais_surrogate)
        c = np.corrcoef(lais_surrogate, lte)
        r = sc.spearmanr(lais_surrogate, lte)
        perm_dist_c[p] = c[0, 1]
        perm_dist_r[p] = r[0]
        # Print a busy symbol to the console
        sys.stdout.write(next(spinner))
        sys.stdout.flush()
        sys.stdout.write('\b')

    corr['p_value_c'] = sum(perm_dist_c >= corr['corrcoef']) / corr['n_permutations']
    corr['p_value_r'] = sum(perm_dist_r >= corr['spearmanr']) / corr['n_permutations']
    print('Pearson corr p-val: {0:.4f}, Spearman r p-val: {1:.4f}'.format(
        corr['p_value_c'], corr['p_value_r']))

    # Save results
    with open(resultspath.joinpath(f'pair_{n_pair:02d}_correlation.p'), 'wb') as f:
        pickle.dump(corr, f)

    # Plot correlation as scatter plot, don't save as pdf, this is not working for
    # this many data points.
    n_bins = 25
    axes_linewidth = 0.7
    plt.rc('text', usetex=False)
    plt.rc('font', family='sans-serif')
    plt.rc('axes', titlesize=12, labelsize=10, titleweight='bold',
           linewidth=axes_linewidth)
    plt.rc('xtick', labelsize=8, direction='out')
    plt.rc('xtick.major', size=2.5, width=axes_linewidth)
    plt.rc('ytick', labelsize=8, direction='out')
    plt.rc('ytick.major', size=2.5, width=axes_linewidth)

    plt.figure(figsize=(8.0, 7.0))
    plt.subplots_adjust(left=0.08, right=0.9, wspace=0.5, hspace=0.7)

    plt.subplot(331)
    plt.scatter(lais, lte,
                color='lightgrey', edgecolor='k')
    plt.xlabel('LAIS [bit]')
    plt.ylabel('LTE [bit]')
    plt.title('all samples', y=1.08)
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()

    plt.subplot(334)
    plt.scatter(lais[rgc_spike_ind], lte[rgc_spike_ind],
                color='seagreen', edgecolor='k')
    plt.xlabel('LAIS [bit]')
    plt.ylabel('LTE [bit]')
    plt.title('LGN spike', y=1.08)
    ax = plt.gca()
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)

    plt.subplot(335)
    plt.scatter(lais[lgn_spike_ind], lte[lgn_spike_ind],
                color='royalblue', edgecolor='k')
    plt.xlabel('LAIS [bit]')
    plt.ylabel('LTE [bit]')
    plt.title('RGC spike', y=1.08)
    ax = plt.gca()
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)

    plt.subplot(336)
    plt.scatter(lais[relayed_ind], lte[relayed_ind],
                color='firebrick', edgecolor='k')
    plt.xlabel('LAIS [bit]')
    plt.ylabel('LTE [bit]')
    plt.title('RGC spike relayed', y=1.08)
    ax = plt.gca()
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)

    plt.subplot(337)
    plt.hist2d(lais[rgc_spike_ind], lte[rgc_spike_ind],
               cmap='Greens', bins=n_bins, norm=LogNorm())
    plt.colorbar()
    plt.xlabel('LAIS [bit]')
    plt.ylabel('LTE [bit]')
    ax = plt.gca()
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)

    plt.subplot(338)
    plt.hist2d(lais[lgn_spike_ind], lte[lgn_spike_ind],
               cmap='Blues', bins=n_bins, norm=LogNorm())
    plt.colorbar()
    plt.xlabel('LAIS [bit]')
    plt.ylabel('LTE [bit]')
    ax = plt.gca()
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)

    plt.subplot(339)
    plt.hist2d(lais[relayed_ind], lte[relayed_ind],
               cmap='Reds', bins=n_bins, norm=LogNorm())
    plt.colorbar()
    plt.xlabel('LAIS [bit]')
    plt.ylabel('LTE [bit]')
    ax = plt.gca()
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)

    plt.savefig(figurepath.joinpath(f'pair_{n_pair:02d}_scatter.png'), dpi=600)
    plt.close()
