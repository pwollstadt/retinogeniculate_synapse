#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot spike-triggered averages of LAIS and LTE for tuples.

Tuples are isolated spike pairs with a prior window of no spiking activity,
termed silence period.
"""
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    load_spike_trains,
    get_paths,
    load_mte_ais_estimates,
    load_local_estimates
)

datapath, resultspath, figurepath = get_paths('paths.json')

# Colors
col_light_gray = 'lightgray'
col_dark_gray = 'darkgray'
col_lte_dark = 'firebrick'
col_lte_light = 'salmon'
col_lais_dark = 'darkblue'
col_lais_light = 'slateblue'

# Set the silence time to one sample further than the max embedding for AIS. We
# don't "see" anything further in the past anyways, in terms of the estimated
# measures. This value is also used as maximum ISI to test for, because,
# larger ISIs mean the second spike's AIS estimate is not affected by the first
# spike.
max_silence_time = 20
max_isi = max_silence_time

for n_pair in range(1, 18):

    print(f'Analyzing spike tuples for pair {n_pair}')

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
    rgc = rgc[mte.get_single_target(1, False)['current_value'][1]:-delay]
    lgn = lgn[mte.get_single_target(1, False)['current_value'][1] + delay:]

    # Find indices of spikes
    rgc_spikes = np.where(rgc > 0)[0]
    lgn_spikes = np.where(lgn > 0)[0]
    # Remove spikes that are too close to the beginning and end of the
    # recording to fit into a window. Multiply the  max lag by two to also
    # account for the silence period before the first spike.
    rgc_spikes = rgc_spikes[rgc_spikes > max_silence_time]
    rgc_spikes = rgc_spikes[rgc_spikes < (len(rgc) - max_silence_time)]
    isi = np.zeros(len(rgc_spikes))
    sta_lais = np.zeros((len(rgc_spikes), 2 * max_silence_time + 1))
    sta_lte = np.zeros((len(rgc_spikes), 2 * max_silence_time + 1))
    sta_rgc = np.zeros((len(rgc_spikes), 2 * max_silence_time + 1))
    sta_lgn = np.zeros((len(rgc_spikes), 2 * max_silence_time + 1))
    relayed = np.zeros(len(rgc_spikes), dtype=bool)

    # Get relayed and non-relayed RGC spikes
    for i in range(1, len(rgc_spikes)-1):
        spike_ind = rgc_spikes[i]
        assert rgc[spike_ind]
        if lgn[spike_ind+1]:  # a tuple is relayed if the second spike in the tuple triggers an LGN response
            relayed[i] = True
        current_isi = rgc_spikes[i+1] - rgc_spikes[i]
        if i == 1:
            silence_period = spike_ind-1
        else:
            silence_period = rgc_spikes[i] - rgc_spikes[i-1]
        if current_isi < max_isi and silence_period >= max_silence_time:
            isi[i] = current_isi
            sta_lais[i, :] = lais[spike_ind - max_silence_time:spike_ind + max_silence_time + 1]
            sta_lte[i, :] = lte[spike_ind - max_silence_time:spike_ind + max_silence_time + 1]
            sta_rgc[i, :] = rgc[spike_ind - max_silence_time:spike_ind + max_silence_time + 1]
            sta_lgn[i, :] = lgn[spike_ind - max_silence_time:spike_ind + max_silence_time + 1]
        else:
            isi[i] = np.nan
    isi[0] = np.nan
    isi[1] = np.nan

    # find STAs triggered on second spike
    plt.figure(figsize=(12.0, 12.0))
    plt_count = 1
    width = 0.1
    for i in range(1, max_isi):
        ind = isi == i

        ax = plt.subplot(max_silence_time, 3, plt_count)
        plt.bar([-i, 0], [1, 1], width=width, color=col_light_gray)
        plt.plot([-(max_silence_time + 1), 1], [0, 0], 'k')
        ax.set(xlim=[-(max_silence_time + 1), 1], ylim=[-0.5, 1.5],
               xticks=(np.arange(-(max_silence_time + 1), 1) + width/2),
               xticklabels=(np.arange(-(max_silence_time + 1), 1)))

        ax = plt.subplot(max_silence_time, 3, plt_count + 1)
        plt.plot(np.arange(-max_silence_time, max_silence_time + 1),
                 np.mean(sta_lais[ind, :], axis=0), color=col_lais_dark)
        if plt_count == 1:
            ylim = [-5, 1.5]
        else:
            ylim = [-1, 1.5]
        ax.set_ylim(ylim)
        plt.plot([0, 0], ylim, ':', color=col_light_gray, linewidth=2)
        plt.plot([i, i], ylim, ':', color=col_light_gray, linewidth=2)

        ax = plt.subplot(max_silence_time, 3, plt_count + 2)
        plt.plot(np.arange(-max_silence_time, max_silence_time + 1),
                 np.mean(sta_lte[ind, :], axis=0), color=col_lte_dark)
        ylim = [-0.1, 1.0]
        ax.set_ylim(ylim)
        plt.plot([0, 0], ylim, ':', color=col_light_gray, linewidth=2)
        plt.plot([-i, -i], ylim, ':', color=col_light_gray, linewidth=2)

        plt_count += 3

    plt.subplots_adjust(left=0.08, right=0.9, wspace=0.5, hspace=0.7)
    plt.savefig(figurepath.joinpath(f'pair_{n_pair:02d}_tuple.png'), dpi=600)
    plt.close()
    np.savez(
        resultspath.joinpath(f'pair_{n_pair:02d}_tuple'),
        sta_lais=sta_lais, sta_lte=sta_lte, sta_rgc=sta_rgc, sta_lgn=sta_lgn,
        relayed=relayed, isi=isi)
