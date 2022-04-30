#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot spike-triggered averages for LAIS and LTE values."""
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

    print(f'Analyzing STAs for pair {n_pair}')

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

    # Find indices of spikes
    rgc_spikes = np.where(rgc > 0)[0]
    lgn_spikes = np.where(lgn > 0)[0]
    # Remove spikes that are too close to the beginning and end of the recording to
    # fit into a window
    max_lag = mte.settings['max_lag_sources']
    rgc_spikes = rgc_spikes[rgc_spikes > max_lag]
    rgc_spikes = rgc_spikes[rgc_spikes < (len(rgc) - max_lag)]
    sta_lais = np.zeros((len(rgc_spikes), 2 * max_lag + 1))
    sta_lte = np.zeros((len(rgc_spikes), 2 * max_lag + 1))
    sta_rgc = np.zeros((len(rgc_spikes), 2 * max_lag + 1))
    sta_lgn = np.zeros((len(rgc_spikes), 2 * max_lag + 1))
    relayed = np.zeros(len(rgc_spikes), dtype=bool)
    # Get relayed and non-relayed RGC spikes
    i = 0
    for s in rgc_spikes:
        assert rgc[s]
        if lgn[s]:
            relayed[i] = True
        sta_lais[i, :] = lais[s - max_lag:s + max_lag + 1]
        sta_lte[i, :] = lte[s - max_lag:s + max_lag + 1]
        sta_rgc[i, :] = rgc[s - max_lag:s + max_lag + 1]
        sta_lgn[i, :] = lgn[s - max_lag:s + max_lag + 1]
        i += 1

    # Save raw data for lSTC plotting
    np.savez(resultspath.joinpath(f'pair_{n_pair:02d}_lstc_data'),
             lais=lais, lte=lte, rgc=rgc, lgn=lgn,
             relayed=relayed, rgc_spikes=rgc_spikes, lgn_spikes=lgn_spikes
             )

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
    plt.subplots_adjust(left=0.08, right=0.9, wspace=0.5, hspace=0.7)

    plt.subplot(431)
    plt.plot(np.arange(-max_lag, max_lag + 1), np.mean(sta_lais[relayed], axis=0),
             color='royalblue', linewidth=1.5)
    plt.plot(
        np.arange(-max_lag, max_lag + 1),
        np.mean(sta_lais[relayed], axis=0) + np.std(sta_lais[relayed], axis=0),
        '--', color='royalblue')
    plt.plot(
        np.arange(-max_lag, max_lag + 1),
        np.mean(sta_lais[relayed], axis=0) - np.std(sta_lais[relayed], axis=0),
        '--', color='royalblue')
    plt.xlabel('t [ms]')
    plt.ylabel('LAIS [bit]')
    plt.title('LAIS STA relayed', y=1.08)
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()

    plt.subplot(432)
    plt.plot(
        np.arange(-max_lag, max_lag + 1),
        np.mean(sta_lais[np.invert(relayed)], axis=0),
        color='royalblue', linewidth=1.5)
    plt.plot(
        np.arange(-max_lag, max_lag + 1),
        np.mean(sta_lais[relayed], axis=0) + np.std(sta_lais[np.invert(relayed)], axis=0),
        '--', color='royalblue')
    plt.plot(
        np.arange(-max_lag, max_lag + 1),
        np.mean(sta_lais[relayed], axis=0) - np.std(sta_lais[np.invert(relayed)], axis=0),
        '--', color='royalblue')
    plt.xlabel('t [ms]')
    plt.ylabel('LAIS [bit]')
    plt.title('LAIS STA non-relayed', y=1.08)
    ax = plt.gca()
    ax.set_ylim(ymin, ymax)

    plt.subplot(433)
    plt.bar(
        [1, 2],
        [np.mean(sta_lais[relayed, max_lag]), np.mean(sta_lais[np.invert(relayed), max_lag])],
        width,
        color='royalblue', linewidth=0.5)
    plt.ylabel('mean LAIS [bit]')
    ax = plt.gca()
    ax.set_xticks(np.array([1, 2]) + width / 2)
    ax.set_xticklabels(['relayed', 'non-relayed'])

    plt.subplot(434)
    plt.plot(np.arange(-max_lag, max_lag + 1), np.mean(sta_lte[relayed], axis=0),
             color='firebrick', linewidth=1.5)
    plt.plot(
        np.arange(-max_lag, max_lag + 1),
        np.mean(sta_lte[relayed], axis=0) + np.std(sta_lte[relayed], axis=0),
        '--', color='firebrick')
    plt.plot(
        np.arange(-max_lag, max_lag + 1),
        np.mean(sta_lte[relayed], axis=0) - np.std(sta_lte[relayed], axis=0),
        '--', color='firebrick')
    plt.xlabel('t [ms]')
    plt.ylabel('LTE [bit]')
    plt.title('LTE STA relayed', y=1.08)
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()

    plt.subplot(435)
    plt.plot(np.arange(-max_lag, max_lag + 1),
             np.mean(sta_lte[np.invert(relayed)], axis=0),
             color='firebrick', linewidth=1.5)
    plt.plot(
        np.arange(-max_lag, max_lag + 1),
        np.mean(sta_lte[relayed], axis=0) + np.std(sta_lte[np.invert(relayed)], axis=0),
        '--', color='firebrick')
    plt.plot(
        np.arange(-max_lag, max_lag + 1),
        np.mean(sta_lte[relayed], axis=0) - np.std(sta_lte[np.invert(relayed)], axis=0),
        '--', color='firebrick')
    plt.xlabel('t [ms]')
    plt.ylabel('LTE [bit]')
    plt.title('LTE STA non-relayed', y=1.08)
    ax = plt.gca()
    ax.set_ylim(ymin, ymax)

    plt.subplot(436)
    plt.bar(
        [1, 2],
        [np.mean(sta_lte[relayed, max_lag]), np.mean(sta_lte[np.invert(relayed), max_lag])],
        width,
        color='firebrick', linewidth=0.5)
    plt.ylabel('mean LTE [bit]')
    ax = plt.gca()
    ax.set_xticks(np.array([1, 2]) + width / 2)
    ax.set_xticklabels(['relayed', 'non-relayed'])

    plt.subplot(437)
    plt.plot(np.arange(-max_lag, max_lag + 1), np.mean(sta_rgc[relayed], axis=0),
             color='slategray', linewidth=1.5)
    plt.plot(
        np.arange(-max_lag, max_lag + 1),
        np.mean(sta_rgc[relayed], axis=0) + np.std(sta_rgc[relayed], axis=0),
        '--', color='slategray')
    plt.plot(
        np.arange(-max_lag, max_lag + 1),
        np.mean(sta_rgc[relayed], axis=0) - np.std(sta_rgc[relayed], axis=0),
        '--', color='slategray')
    plt.xlabel('t [ms]')
    plt.ylabel('RGC [bit]')
    plt.title('RGC STA relayed', y=1.08)
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()

    plt.subplot(438)
    plt.plot(np.arange(-max_lag, max_lag + 1),
             np.mean(sta_rgc[np.invert(relayed)], axis=0),
             color='slategray', linewidth=1.5)
    plt.plot(
        np.arange(-max_lag, max_lag + 1),
        np.mean(sta_rgc[relayed], axis=0) + np.std(sta_rgc[np.invert(relayed)], axis=0),
        '--', color='slategray')
    plt.plot(
        np.arange(-max_lag, max_lag + 1),
        np.mean(sta_rgc[relayed], axis=0) - np.std(sta_rgc[np.invert(relayed)], axis=0),
        '--', color='slategray')
    plt.xlabel('t [ms]')
    plt.ylabel('RGC [bit]')
    plt.title('RGC STA non-relayed', y=1.08)
    ax = plt.gca()
    ax.set_ylim(ymin, ymax)

    plt.subplot(439)
    plt.bar(
        [1, 2],
        [np.mean(sta_rgc[relayed, max_lag]), np.mean(sta_rgc[np.invert(relayed), max_lag])],
        width,
        color='slategray', linewidth=0.5)
    plt.ylabel('mean RGC [bit]')
    ax = plt.gca()
    # ax.set_ylim(ymin, ymax)
    ax.set_xticks(np.array([1, 2]) + width / 2)
    ax.set_xticklabels(['relayed', 'non-relayed'])

    plt.subplot(4, 3, 10)
    plt.plot(np.arange(-max_lag, max_lag + 1), np.mean(sta_lgn[relayed], axis=0),
             color='seagreen', linewidth=1.5)
    plt.plot(
        np.arange(-max_lag, max_lag + 1),
        np.mean(sta_lgn[relayed], axis=0) + np.std(sta_lgn[relayed], axis=0),
        '--', color='seagreen')
    plt.plot(
        np.arange(-max_lag, max_lag + 1),
        np.mean(sta_lgn[relayed], axis=0) - np.std(sta_lgn[relayed], axis=0),
        '--', color='seagreen')
    plt.xlabel('t [ms]')
    plt.ylabel('LGN [bit]')
    plt.title('LGN STA relayed', y=1.08)
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()

    plt.subplot(4, 3, 11)
    plt.plot(np.arange(-max_lag, max_lag + 1),
             np.mean(sta_lgn[np.invert(relayed)], axis=0),
             color='seagreen', linewidth=1.5)
    plt.plot(
        np.arange(-max_lag, max_lag + 1),
        np.mean(sta_lgn[relayed], axis=0) + np.std(sta_lgn[np.invert(relayed)], axis=0),
        '--', color='seagreen')
    plt.plot(
        np.arange(-max_lag, max_lag + 1),
        np.mean(sta_lgn[relayed], axis=0) - np.std(sta_lgn[np.invert(relayed)], axis=0),
        '--', color='seagreen')
    plt.xlabel('t [ms]')
    plt.ylabel('LGN [bit]')
    plt.title('LGN STA non-relayed', y=1.08)
    ax = plt.gca()
    ax.set_ylim(ymin, ymax)

    plt.subplot(4, 3, 12)
    plt.bar(
        [1, 2],
        [np.mean(sta_lgn[relayed, max_lag]), np.mean(sta_lgn[np.invert(relayed), max_lag])],
        width,
        color='seagreen', linewidth=0.5)
    plt.ylabel('mean LGN [bit]')
    ax = plt.gca()
    ax.set_xticks(np.array([1, 2]) + width / 2)
    ax.set_xticklabels(['relayed', 'non-relayed'])

    plt.savefig(figurepath.joinpath(f'pair_{n_pair:02d}_sta.png'), dpi=600)
    plt.close()

    np.savez(resultspath.joinpath(f'pair_{n_pair:02d}_sta'),
             sta_lais=sta_lais, sta_lte=sta_lte, sta_rgc=sta_rgc, sta_lgn=sta_lgn,
             relayed=relayed)
