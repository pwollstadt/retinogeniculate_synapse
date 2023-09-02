#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plote correlation between PID estimates and LSTC"""
import pickle
import numpy as np

import matplotlib.pyplot as plt

from utils import load_mte_ais_estimates, get_paths, load_local_estimates

datapath, resultspath, figurepath = get_paths('paths.json')
# List of pairs with significant TE/AIS estimates.
all_pairs = [i for i in range(1, 18)]
all_pairs.remove(5)
N = len(all_pairs)

fig_ext = 'pdf'


def correlate_synergy_lstc():
    with open(resultspath.joinpath('all_pairs_correlation_contr.p'), 'rb') as f:
        lstc = pickle.load(f)

    syn_all = np.zeros(N)
    unq_rgc_all = np.zeros(N)
    te_all = np.zeros(N)
    ais_all = np.zeros(N)
    for i, n_pair in enumerate(all_pairs):
        with open(resultspath.joinpath(f'pair_{n_pair:02d}_pid.p'), 'rb') as f:
            pid = pickle.load(f)
        syn_all[i] = pid['syn_s1_s2']
        lte, lais, delay = load_local_estimates(resultspath, n_pair)
        mte, ais = load_mte_ais_estimates(resultspath, n_pair)
        embedding_diff = (mte.get_single_target(1, False).current_value[1] -
                          ais.get_single_process(0, False).current_value[1])
        lais = lais[embedding_diff:-delay]
        lte = lte[delay:]

        syn_all[i] = pid['syn_s1_s2']
        unq_rgc_all[i] = pid['unq_s1']
        te_all[i] = np.mean(lte)
        ais_all[i] = np.mean(lais)

    syn_all_norm = syn_all / te_all
    rgc_unq_all_norm = unq_rgc_all / te_all

    fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(6, 7))
    ax = ax.flatten()
    ax[0].scatter(lstc['c_all'], syn_all_norm, c='gray')
    ax[1].scatter(lstc['c_all'], rgc_unq_all_norm, c='gray')
    ax[2].scatter(te_all, syn_all_norm, c='r')
    ax[3].scatter(te_all, rgc_unq_all_norm, c='r')
    ax[4].scatter(ais_all, syn_all_norm, c='b')
    ax[5].scatter(ais_all, rgc_unq_all_norm, c='b')

    xlabels = ['$c$', '$c$', '$TE$', '$TE$', '$AIS$', '$AIS$']
    ylabels = ['$I_{syn}/TE$', '$I_{unq}/TE$', '$I_{syn}$', '$I_{unq}$', '$I_{syn}$', '$I_{unq}$']
    for a, xl, yl in zip(ax, xlabels, ylabels):
        a.set(xlabel=xl, ylabel=yl)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    correlate_synergy_lstc()
