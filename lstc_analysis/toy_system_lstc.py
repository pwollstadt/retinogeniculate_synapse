#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simulate data to demonstrate negative LSTC."""
import pickle
import copy as cp
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt

from idtxl.data import Data
from idtxl.active_information_storage import ActiveInformationStorage
from idtxl.multivariate_te import MultivariateTE

from utils import (
    get_paths,
    load_spike_trains,
    load_local_estimates,
    load_mte_ais_estimates
)

plt.rc('text', usetex=False)
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

def plot_2d_hist(lais, lte, rgc, ax, n_bins=15):
    """Plot 2D-histogram of local AIS and TE estimates"""
    print('LAIS hist:')
    print('  ', '\t'.join([f'{count}' for count in np.histogram(lais, bins=n_bins)[0]]))
    print('  ', '\t'.join([f'{edge:.2f}' for edge in np.histogram(lais, bins=n_bins)[1]]))
    print('LTE  hist:')
    print('  ', '\t'.join([f'{count}\t' for count in np.histogram(lte, bins=n_bins)[0]]))
    print('  ', '\t'.join([f'{edge:.2f}\t' for edge in np.histogram(lte, bins=n_bins)[1]]))

    rgc_spike_ind = rgc > 0

    _, _, _, h = ax.hist2d(
        lais[rgc_spike_ind.astype(bool)],
        lte[rgc_spike_ind.astype(bool)],
        bins=n_bins,
        cmap='Greys',
        norm=mlp.colors.LogNorm()
    )
    plt.colorbar(h, ax=ax)
    ax.set(xlabel='$lAIS$', ylabel='$lTE$')
    ax.axhline(0, c='k', ls='--', alpha=0.5)
    ax.axvline(0, c='k', ls='--', alpha=0.5)

def _calculate_correlation(lais, lte):
    """Calculate Pearson correlation and perform permutation test"""
    c = np.corrcoef(lais, lte)
    corr = {'corrcoef': c[0, 1],
            'n_permutations': 200}

    # Perform permutation test
    perm_dist = np.zeros(corr['n_permutations'])
    lais_surrogate = cp.copy(lais)  # make a copy to use numpy's shuffling
    print('Perform permutation test with {0} surrogates'.format(
        corr['n_permutations']))
    for p in range(corr['n_permutations']):
        # Shuffle LAIS in place
        np.random.shuffle(lais_surrogate)
        c = np.corrcoef(lais_surrogate, lte)
        perm_dist[p] = c[0, 1]

    # One-sided test against H0: anti-correlation if significantly different from zero
    n_smaller = sum(perm_dist < corr['corrcoef'])
    if n_smaller == 0:
        corr['p_value'] = 1 / corr['n_permutations']
    else:
        corr['p_value'] = n_smaller / corr['n_permutations']
    return corr

def plot_lais_sta(lais, lte, rgc, lgn):
    """Plot spike-triggered averages"""
    max_silence_time = 20
    rgc_spikes = np.where(rgc > 0)[0]
    rgc_spikes = rgc_spikes[rgc_spikes > max_silence_time]
    rgc_spikes = rgc_spikes[rgc_spikes < (len(rgc) - max_silence_time)]
    sta_window = 10
    sta_rgc = []
    sta_lgn = []
    sta_lais_nonrel = []
    sta_lais_rel = []
    sta_lte_nonrel = []
    sta_lte_rel = []
    isi = np.zeros(len(rgc_spikes))
    isi_rel = []
    isi_nonrel = []
    for i in range(1, len(rgc_spikes)):
        current_isi = rgc_spikes[i] - rgc_spikes[i-1]
        isi[i] = current_isi
        spike_ind = rgc_spikes[i]
        sta_rgc.append(rgc[spike_ind-sta_window:spike_ind+sta_window+1])
        sta_lgn.append(lgn[spike_ind-sta_window:spike_ind+sta_window+1])
        if lgn[spike_ind+1]:
            sta_lais_rel.append(lais[spike_ind-sta_window:spike_ind+sta_window+1])
            sta_lte_rel.append(lte[spike_ind-sta_window:spike_ind+sta_window+1])
            isi_rel.append(current_isi)
        else:
            sta_lais_nonrel.append(lais[spike_ind-sta_window:spike_ind+sta_window+1])
            sta_lte_nonrel.append(lte[spike_ind-sta_window:spike_ind+sta_window+1])
            isi_nonrel.append(current_isi)


    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(10, 8))

    ax[0,0].plot(np.array(sta_rgc).mean(axis=0))
    ax[0,0].set(title='RGC')
    ax[0,0].axvline(10, c='k', ls='--', alpha=0.5)
    ax[0,1].plot(np.array(sta_lgn).mean(axis=0))
    ax[0,1].set(title='LGN')
    ax[0,1].axvline(10, c='k', ls='--', alpha=0.5)

    ax[1,0].plot(np.array(sta_lais_rel).mean(axis=0), 'b')
    ax[1,0].set(title='LAIS - relayed')
    ax[1,0].axvline(10, c='k', ls='--', alpha=0.5)
    ax[1,1].plot(np.array(sta_lte_rel).mean(axis=0), 'r')
    ax[1,1].set(title='LTE - relayed')
    ax[1,1].axvline(10, c='k', ls='--', alpha=0.5)

    ax[2,0].plot(np.array(sta_lais_nonrel).mean(axis=0), 'b')
    ax[2,0].set(title='LAIS - non-relayed')
    ax[2,0].axvline(10, c='k', ls='--', alpha=0.5)
    ax[2,1].plot(np.array(sta_lte_nonrel).mean(axis=0), 'r')
    ax[2,1].set(title='LTE - non-relayed')
    ax[2,1].axvline(10, c='k', ls='--', alpha=0.5)
    for a in ax[:2,:2].flatten():
        a.set(xticks=np.arange(0, sta_window*2+1, 2),xticklabels=np.arange(-sta_window, sta_window+1, 2))

    ax[0,2].hist(isi, bins=20, range=(0, 20))
    ax[0,2].set(title='ISI')
    ax[0,3].bar(['rel', 'nonrel'], [len(sta_lais_rel), len(sta_lais_nonrel)])
    ax[1,2].hist(isi_rel, bins=20, range=(0, 20))
    ax[1,2].set(title='ISI relayed')
    ax[1,3].hist(isi_nonrel, bins=20, range=(0, 20))
    ax[1,3].set(title='ISI nonrelayed')

    plot_2d_hist(lais, lte, rgc, ax[2,2])
    plt.tight_layout()
    return fig


def _plot_sta(rgc, lgn, threshold, rgc_spikes):
    # Generates actual figures.
    sta_window = 10
    sta_rgc_bigger_ff = []
    sta_rgc_smaller_ff = []
    sta_lgn_bigger_ff = []
    sta_lgn_smaller_ff = []
    isi = np.zeros(len(rgc_spikes))
    for i in range(1, len(rgc_spikes)):
        current_isi = rgc_spikes[i] - rgc_spikes[i-1]
        isi[i] = current_isi
        spike_ind = rgc_spikes[i]
        if current_isi > threshold:
            sta_rgc_bigger_ff.append(rgc[spike_ind-sta_window:spike_ind+sta_window+1])
            sta_lgn_bigger_ff.append(lgn[spike_ind-sta_window:spike_ind+sta_window+1])
        else:
            sta_rgc_smaller_ff.append(rgc[spike_ind-sta_window:spike_ind+sta_window+1])
            sta_lgn_smaller_ff.append(lgn[spike_ind-sta_window:spike_ind+sta_window+1])

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
    ax[0,0].plot(np.array(sta_rgc_smaller_ff).mean(axis=0))
    ax[0,0].set(title=f'RGC ISI < t={threshold}')
    ax[0,0].axvline(10, c='k', ls='--', alpha=0.5)
    ax[0,1].plot(np.array(sta_lgn_smaller_ff).mean(axis=0))
    ax[0,1].set(title=f'LGN ISI < t={threshold}')
    ax[0,1].axvline(10, c='k', ls='--', alpha=0.5)
    ax[1,0].plot(np.array(sta_rgc_bigger_ff).mean(axis=0))
    ax[1,0].set(title=f'RGC ISI >= t={threshold}')
    ax[0,1].axvline(10, c='k', ls='--', alpha=0.5)
    ax[1,1].plot(np.array(sta_lgn_bigger_ff).mean(axis=0))
    ax[1,1].set(title=f'LGN ISI >= t={threshold}')
    ax[1,1].axvline(10, c='k', ls='--', alpha=0.5)
    for a in ax[:2,:2].flatten():
        a.set(xticks=np.arange(0, sta_window*2+1, 2),xticklabels=np.arange(-sta_window, sta_window+1, 2))
    ax[0,2].hist(isi, bins=20, range=(0, 20))
    plt.tight_layout()
    return fig


def _simulate_lgn_data(resultspath, datapath, pair_toy_system, threshold, n_samples=None, max_silence_time=20):
    """Simulate LGN data by removing responses of RGC spikes with prior ISI below the threshold"""
    # Load raw data and IDTxl AIS and TE results.
    _, _, delay = load_local_estimates(resultspath, pair_toy_system)
    mte, _ = load_mte_ais_estimates(resultspath, pair_toy_system)
    print(f'Loaded delay and STA data for pair {pair_toy_system} - delay: {delay}')
    rgc, lgn = load_spike_trains(datapath, pair_toy_system)

    efficacy = np.genfromtxt(datapath.joinpath('effic.csv'), delimiter=',')
    efficacy = efficacy[pair_toy_system-1] / 100

    # Align data by their delay to make detection of a relayed spike easier
    # After aligning, the delay is one time step.
    rgc = rgc[mte.get_single_target(1, False)['current_value'][1]:-(delay-1)]
    lgn = lgn[mte.get_single_target(1, False)['current_value'][1] + (delay-1):]

    if n_samples is not None:
        rgc = rgc[:n_samples]
        lgn = lgn[:n_samples]

    rgc_spikes = np.where(rgc > 0)[0]
    rgc_spikes = rgc_spikes[rgc_spikes > max_silence_time]
    rgc_spikes = rgc_spikes[rgc_spikes < (len(rgc) - max_silence_time)]
    print(f'Loaded raw spike trains for pair {pair_toy_system}, delay={delay}, effic={efficacy}')

    # Get actual statistics from raw data. Calculate a stricter version of the
    # efficacy that considers a spike relayed only if the reconstructed delay
    # is matched exactly
    relayed_spikes = lgn[rgc_spikes+1].sum()
    relayed_first_spikes = 0
    relayed_second_spikes = 0
    isi = np.zeros(len(rgc_spikes))
    for i in range(1, len(rgc_spikes)):
        current_isi = rgc_spikes[i] - rgc_spikes[i-1]
        isi[i] = current_isi
        if current_isi > threshold:
            relayed_first_spikes += lgn[rgc_spikes[i]+1]
        else:
            relayed_second_spikes += lgn[rgc_spikes[i]+1]
    print('ISI distribution:', np.histogram(isi, bins=10, range=(0, 10)))
    print(f'Found {relayed_spikes} relayed spikes, of which {relayed_first_spikes} were preceeded by an ISI > {threshold}' )

    # Simulate LGN spiking without response to second spike
    print(f'Using an ISI threshold {threshold} ms')
    np.random.seed(0)
    lgn_simulated = lgn.copy()
    removed_spikes = 0
    for i in range(1, len(rgc_spikes)):
        current_isi = rgc_spikes[i] - rgc_spikes[i-1]
        if current_isi <= threshold:
            if lgn_simulated[rgc_spikes[i]+1]:
                removed_spikes += 1
            lgn_simulated[rgc_spikes[i]+1] = 0

    print(f'RGC Spikes: {np.sum(rgc)}')
    print(f'LGN Spikes: {np.sum(lgn)}')
    print(f'LGN Spikes in simulated data: {np.sum(lgn_simulated)} ({removed_spikes} spikes removed)\n')
    fig_orig = _plot_sta(rgc, lgn, threshold, rgc_spikes)
    fig_sim = _plot_sta(rgc, lgn_simulated, threshold, rgc_spikes)
    return rgc, lgn_simulated, delay, fig_orig, fig_sim

def _align_local_estimates(lais, lte, rgc, lgn, res_mte, res_ais, delay=1):
    """Align estimates by accounting for different embedding lengths."""
    embedding_diff = (res_mte.get_single_target(1, False).current_value[1] -
                      res_ais.get_single_process(0, False).current_value[1])
    lais = lais[embedding_diff:-delay]
    lte = lte[delay:]
    rgc = rgc[res_ais.settings['max_lag']+embedding_diff:-delay]
    lgn = lgn[res_ais.settings['max_lag']+embedding_diff:-delay]
    assert len(rgc) == len(lais), 'RGC spike not aligned with LAIS'
    assert len(lte) == len(lais), 'LTE not aligned with LAIS'
    return lais, lte, rgc, lgn, delay


def run_toy_system(pair_toy_system, threshold, n_samples):
    """Simulate data and save results to disk"""
    datapath, resultspath, _ = get_paths('paths.json')

    print('\n\nRunning local AIS and TE estimation\n\n')

    rgc, lgn_simulated, delay, fig_orig, fig_sim = _simulate_lgn_data(
        resultspath, datapath, pair_toy_system, threshold, n_samples
    )
    fig_orig.savefig(resultspath.joinpath(
        'toy_system', f'pair_{pair_toy_system:02d}_thresh_{threshold}_n_{n_samples}_sta_orig.png'
    ))
    fig_sim.savefig(resultspath.joinpath(
        'toy_system', f'pair_{pair_toy_system:02d}_thresh_{threshold}_n_{n_samples}_sta_sim.png'
    ))
    plt.close('all')

    data = Data(np.vstack((rgc, lgn_simulated)), 'ps', normalise=False)

    # Estimate TE from RGC to LGN spike train. Use the same parameters as for
    # the original analysis.
    print('Start TE estimation')
    settings = {
        'cmi_estimator': 'JidtDiscreteCMI',
        'alph1': 2,  # define alphabet size for input data
        'alph2': 2,
        'alphc': 2,
        'n_perm_max_stat': 200,
        'n_perm_min_stat': 200,
        'max_lag_target': 30,
        'max_lag_sources': 40,
        'min_lag_sources': 1,
        'tau': 1,
        'local_values': True
        }
    mte_analysis = MultivariateTE()
    target = 1
    res_mte = mte_analysis.analyse_single_target(settings, data, target=target)
    if len(res_mte.get_single_target(target, fdr=False)['selected_vars_sources']) == 0:
        print('No significant TE - return')
        return {
            'corrcoef': np.nan,
            'n_permutations': 200,
            'p_value': np.nan,
            'pair': pair_toy_system,
            }

    # Estimate AIS within the RGC
    print('Start source AIS estimation')
    settings = {
        'cmi_estimator': 'JidtDiscreteCMI',
        'alph1': 2,
        'alph2': 2,
        'alphc': 2,
        'n_perm_max_stat': 200,
        'n_perm_min_stat': 200,
        'max_lag': 30,
        'tau': 1,
        'local_values': True
        }
    ais_analysis = ActiveInformationStorage()
    res_ais = ais_analysis.analyse_single_process(settings, data, process=0)

    lais = np.squeeze(res_ais.get_single_process(0, False)['ais'])
    lte = np.squeeze(res_mte.get_single_target(1, False)['te'])
    lais, lte, rgc, lgn_simulated, delay = _align_local_estimates(
        lais, lte, rgc, lgn_simulated, res_mte, res_ais
    )
    corr = _calculate_correlation(lais, lte)
    corr['pair'] = pair_toy_system

    fig = plot_lais_sta(lais, lte, rgc, lgn_simulated)
    fig.savefig(resultspath.joinpath(
        'toy_system', f'pair_{pair_toy_system:02d}_thresh_{threshold}_n_{n_samples}_sta_2dhist_lais_lte.png'
    ))
    plt.close('all')
    print(f'\n\nPair {pair_toy_system} -- thresh ISI: {threshold} -- N: {n_samples}:')
    print(f'  Delay: {delay}')
    print(f'  Pearson corr: {corr["corrcoef"]:.4f}')
    print(f'  Pearson p-val: {corr["p_value"]:.4f}\n\n')
    with open(resultspath.joinpath('toy_system', f'pair_{pair_toy_system:02d}_thresh_{threshold}_n_{n_samples}_correlation.p'), 'wb') as f:
        pickle.dump(corr, f)
    np.savez(
        resultspath.joinpath(
            'toy_system',
            f'pair_{pair_toy_system:02d}_thresh_{threshold}_n_{n_samples}_local_estimates'
        ),
        rgc=rgc,
        lgn_simulated=lgn_simulated,
        lais=lais,
        lte=lte,
        delay=np.array([delay])
    )
    pd.DataFrame({
        'pair': [pair_toy_system],
        'threshold': [threshold],
        'N': [n_samples],
        'delay': [delay],
        'Pearson corr': [corr["corrcoef"]],
        'Pearson p-val': [corr["p_value"]],
    }).to_csv(
        resultspath.joinpath(
            'toy_system',
            f'pair_{pair_toy_system:02d}_thresh_{threshold}_n_{n_samples}_results.csv'
        ),
        index=False
    )
    return corr


def main():
    pair = 4
    thresholds = [4, 8, 10, 15]

    # print('Running simulations')
    # results = []
    # for n_samples in [20000, 50000, 100000]:
    #     for threshold in thresholds:
    #         results.append(
    #             run_toy_system(
    #                 pair_toy_system=pair,
    #                 threshold=threshold,
    #                 n_samples=n_samples
    #             )
    #         )
    # df = pd.DataFrame(results)
    # print(df)
    # df.to_csv('scan_params.csv')

    # Plot results.
    _, resultspath, _ = get_paths('paths.json')
    loadpath = resultspath.joinpath('toy_system')

    _, ax = plt.subplots(ncols=2, figsize=(6, 3))
    corr = []
    for n_samples in [20000, 50000, 100000]:
        for threshold in thresholds:
            try:
                corr.append(pd.read_csv(
                    loadpath.joinpath(f'pair_{pair:02d}_thres_{threshold}_n_{n_samples}_results.csv')
                ))
            except FileNotFoundError as err:
                print(err.args)
                continue
    df = pd.concat(corr)
    print(df)

    df.pivot(
        index='threshold', columns='N', values='Pearson corr'
        ).plot(
            ax=ax[0],
        )
    ax[0].set(
        title=f'Pair {pair}',
        xticks=thresholds,
        ylabel='$c(lAIS, lTE)$',
        xlabel='$t_{ISI}$'
    )

    # Plot 2D-histogram for a selected parameter configuration.
    threshold_example = 8
    n_samples_example = 100000
    print(pd.read_csv(
        loadpath.joinpath(
            f'pair_{pair:02d}_thres_{threshold_example}_n_{n_samples_example}_results.csv'
        )
    ))
    f = np.load(loadpath.joinpath(
        f'pair_{pair:02d}_thres_{threshold_example}_n_{n_samples_example}_local_estimates.npz'
    ))
    plot_2d_hist(f['lais'], f['lte'], f['rgc'], ax[1])
    ax[1].set(title=f'Pair {pair}, '+'$t_{ISI}$=' + f'{threshold_example}')

    plt.tight_layout()
    plt.savefig(loadpath.joinpath(f's2_fig_negative_lstc_pair_{pair}.png'))
    plt.show()


if __name__=='__main__':
    main()
