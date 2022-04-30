#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot LAIS and LTE as a function of the inter-spike interval (ISI)."""
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    load_spike_trains,
    get_paths,
    load_mte_ais_estimates,
    load_local_estimates
)


datapath, resultspath, figurepath = get_paths('paths.json')

for n_pair in range(1, 18):

    print(f'Analyzing ISIs for pair {n_pair}')

    rgc, lgn = load_spike_trains(datapath, n_pair)
    lte, lais, delay = load_local_estimates(resultspath, n_pair)
    mte, ais = load_mte_ais_estimates(resultspath, n_pair)
    if lte is None or lais is None:
        print('No estimates for current spike pair')
        continue

    # Align data
    embedding_diff = (mte.get_single_target(1, False).current_value[1] -
                      ais.get_single_process(0, False).current_value[1])
    lais = lais[embedding_diff:-delay]
    lte = lte[delay:]
    rgc = rgc[mte.get_single_target(1, False).current_value[1]:-delay]
    lgn = lgn[mte.get_single_target(1, False).current_value[1] + delay:]
    assert len(lais) == len(lte)
    assert len(lais) == len(rgc)
    assert len(lais) == len(lgn)

    # Find indices of spikes
    rgc_spikes = np.where(rgc > 0)[0]
    lgn_spikes = np.where(lgn > 0)[0]
    # Remove the first spike, it has no isi
    isi = (rgc_spikes[1:] - rgc_spikes[:-1]) - 1
    relayed = lgn[rgc > 0]
    relayed = relayed[1:].astype(bool)
    lais_spikes = lais[rgc > 0]
    lte_spikes = lte[rgc > 0]
    lais_spikes = lais_spikes[1:]
    lte_spikes = lte_spikes[1:]

    # Get mean LAIS/LTE per ISI according to.
    isi_unique, idx, isi_counts = np.unique(isi, return_inverse=True, return_counts=True)
    sum_lais = np.bincount(idx, weights=lais_spikes)
    lais_by_isi = sum_lais / isi_counts
    sum_lte = np.bincount(idx, weights=lte_spikes)
    lte_by_isi = sum_lte / isi_counts
    # Relayed spikes.
    isi_unique_relayed, idx, isi_counts_relayed = np.unique(isi[relayed], return_inverse=True, return_counts=True)
    sum_lais = np.bincount(idx, weights=lais_spikes[relayed])
    lais_by_isi_relayed = sum_lais / isi_counts_relayed
    sum_lte = np.bincount(idx, weights=lte_spikes[relayed])
    lte_by_isi_relayed = sum_lte / isi_counts_relayed
    # Non-relayed spikes.
    isi_unique_nonrelayed, idx, isi_counts_nonrelayed = np.unique(
        isi[np.invert(relayed)], return_inverse=True, return_counts=True)
    sum_lais = np.bincount(idx, weights=lais_spikes[np.invert(relayed)])
    lais_by_isi_nonrelayed = sum_lais / isi_counts_nonrelayed
    sum_lte = np.bincount(idx, weights=lte_spikes[np.invert(relayed)])
    lte_by_isi_nonrelayed = sum_lte / isi_counts_nonrelayed

    axes_linewidth = 0.7
    plt.rc('text', usetex=False)
    plt.rc('font', family='sans-serif')
    plt.rc('axes', titlesize=12, labelsize=10, titleweight='bold', linewidth=axes_linewidth)
    plt.rc('xtick', labelsize=8, direction='out')
    plt.rc('xtick.major', size=2.5, width=axes_linewidth)
    plt.rc('ytick', labelsize=8, direction='out')
    plt.rc('ytick.major', size=2.5, width=axes_linewidth)
    plt.rc('legend', fontsize=10)

    plt.figure(figsize=(8.0, 7.0))
    width = 0.5
    plt.subplots_adjust(left=0.12, right=0.9, wspace=0.5, hspace=0.7)
    max_isi = 40
    plt.subplot(221)
    plt.hist(isi, bins=np.arange(max_isi), linewidth=0.5)
    plt.xlabel('ISI [ms]')
    plt.ylabel('abs. frequency')
    plt.title('frequency of ISIs', y=1.08)

    # LAIS by ISI, total and divided into relayed and non-relayed.
    plt.subplot(223)
    plt.plot(isi_unique, lais_by_isi, label='all RGC spikes')
    plt.plot(isi_unique_relayed, lais_by_isi_relayed, 'b--',
             label='relayed RGC spikes')
    plt.plot(isi_unique_nonrelayed, lais_by_isi_nonrelayed, 'b:',
             label='non-relayed RGC spikes')
    plt.xlabel('ISI [ms]')
    plt.ylabel('LAIS [bits]')
    plt.title('LAIS by ISI', y=1.08)
    plt.legend(loc='lower right')
    ax = plt.gca()
    ax.set_xlim(-0.5, max_isi)

    # LTE by ISI, total and divided into relayed and non-relayed.
    plt.subplot(224)
    plt.plot(isi_unique, lte_by_isi, 'r', label='all RGC spikes')
    plt.plot(isi_unique_relayed, lte_by_isi_relayed, 'r--', label='relayed RGC spikes')
    plt.plot(isi_unique_nonrelayed, lte_by_isi_nonrelayed, 'r:', label='non-relayed RGC spikes')
    plt.xlabel('ISI [ms]')
    plt.ylabel('LTE [bits]')
    plt.title('LTE by ISI', y=1.08)
    ax = plt.gca()
    ax.set_xlim(-0.5, max_isi)

    plt.savefig(figurepath.joinpath(f'pair_{n_pair:02d}_isi.png'), dpi=600)
    plt.close()

    np.savez(
        resultspath.joinpath(f'pair_{n_pair:02d}_isi'),
        isi=isi, relayed=relayed,
        rgc_spikes=rgc_spikes, lgn_spikes=lgn_spikes,
        isi_unique=isi_unique, isi_counts=isi_counts,
        lais_by_isi=lais_by_isi, lte_by_isi=lte_by_isi,
        isi_counts_relayed=isi_counts_relayed,
        isi_counts_nonrelayed=isi_counts_nonrelayed,
        isi_unique_relayed=isi_unique_relayed,
        lais_by_isi_relayed=lais_by_isi_relayed,
        lte_by_isi_relayed=lte_by_isi_relayed,
        isi_unique_nonrelayed=isi_unique_nonrelayed,
        lais_by_isi_nonrelayed=lais_by_isi_nonrelayed,
        lte_by_isi_nonrelayed=lte_by_isi_nonrelayed
        )
