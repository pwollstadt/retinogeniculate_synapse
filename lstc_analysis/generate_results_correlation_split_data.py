#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Genrerate local storage-transfer correlations (LSTC) for split data.

Load locally PT-corrected estimates of LAIS and LTE and calculate the
Pearson correlation. Compare results to estimates from full data sets.
"""
import pickle

import numpy as np
import matplotlib.pyplot as plt

from idtxl.estimators_jidt import JidtDiscreteCMI

from utils import get_paths, load_mte_ais_estimates, load_local_estimates
from generate_results_correlation import calculate_lstc


def results_split_data(pairs_to_split):

    datapath, resultspath, figurepath = get_paths('paths.json')

    JidtDiscreteCMI({})

    # LTE (red) and LAIS (blue)
    col_lte_dark = 'indianred'  # alt.: 'firebrick', 'darkred'
    col_lte_light = 'salmon'
    col_lais_dark = 'steelblue'  # 'darkblue'
    col_lais_light = 'lightsteelblue'  # 'slateblue'

    fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(9, 8))

    print('Results on full data sets')
    i = 0
    for n_pair in pairs_to_split:

        # Load and print results from full data set for comparison.
        lte, lais, delay = load_local_estimates(resultspath, n_pair)
        with open(resultspath.joinpath(f'pair_{n_pair:02d}_correlation.p'), 'rb') as f:
            corr_full = pickle.load(f)
        print(f'\nPair {n_pair}')
        print(f'\tlAIS: mean: {np.mean(lais):.4f} SD: {np.std(lais):.4f} min: {np.min(lais):.4f} max: {np.max(lais):.4f}')
        print(f'\tlTE:  mean: {np.mean(lte):.4f} SD: {np.std(lte):.4f} min: {np.min(lte):.4f} max: {np.max(lte):.4f}')
        ax[i, 0].hist(lais, bins=20, log=True, color=col_lais_dark)
        ax[i, 0].set(title=f'Pair {n_pair} - all data', xlabel='$lAIS$', ylabel='count')
        ax[i+1, 0].hist(lte, bins=20, log=True, color=col_lte_dark)
        ax[i+1, 0].set(title=f'Pair {n_pair} - all data', xlabel='$lTE$', ylabel='count')
        print('\tPearson corr: {0:.4f} (p={1:.4f}), spearman r: {2:.4f} (p={3:.4f})'.format(
            corr_full['corrcoef'], corr_full['p_value_c'], corr_full['spearmanr'], corr_full['p_value_r']))

        # Generate and print results for first and second half of each data set
        n_pair_split_1 = n_pair*3
        n_pair_split_2 = n_pair*3+1
        j = 0
        for split in [n_pair_split_1, n_pair_split_2]:
            lte_split, lais_split, delay = load_local_estimates(resultspath, split)

            print(f'Results split {j+1}')
            print(f'\tlAIS: mean: {np.mean(lais_split):.4f} SD: {np.std(lais_split):.4f} min: {np.min(lais_split):.4f} max: {np.max(lais_split):.4f}')
            print(f'\tlTE:  mean: {np.mean(lte_split):.4f} SD: {np.std(lte_split):.4f} min: {np.min(lte_split):.4f} max: {np.max(lte_split):.4f}')
            print(f'\tDiff lAIS mean: {(np.mean(lais_split)-np.mean(lais))/np.mean(lais):.4f}')
            print(f'\tDiff lTE mean:  {(np.mean(lte_split)-np.mean(lte))/np.mean(lte):.4f}')

            ax[i, 1+j].hist(lais_split, bins=20, log=True, color=col_lais_light)
            ax[i, 1+j].set(title=f'Pair {n_pair} - split {j+1}' , xlabel='$lAIS$', ylabel='count')
            ax[i+1, 1+j].hist(lte_split, bins=20, log=True, color=col_lte_light)
            ax[i+1, 1+j].set(title=f'Pair {n_pair} - split {j+1}' , xlabel='$lTE$', ylabel='count')
            j += 1

            mte, ais = load_mte_ais_estimates(resultspath, split)
            if lte_split is None or lais_split is None:
                print('No estimates for current spike pair')
                continue

            # Correct for differences in embedding lengths (max_lag of 40 ms for TE and a max_lag of 30 for AIS. Also account
            # for the delay between LAIS and LTE, i.e, the sample in the RGC that is most informative about the current value of
            # the LGN is in the past of the current value due to an information transfer delay. Also account for different max
            # lags in the LTE and LAIS estimation.
            embedding_diff = (mte.get_single_target(1, False).current_value[1] -
                              ais.get_single_process(0, False).current_value[1])
            lais_split = lais_split[embedding_diff:-delay]
            lte_split = lte_split[delay:]

            # Calculate correlation coefficients.
            corr_split = calculate_lstc(lais_split, lte_split, n_permutations=100)
            print('\tPearson corr: {0:.4f} (p={1:.4f})'.format(corr_split['corrcoef'], corr_split['p_value_c']))
            print('\tDiff Pearson corr: {0:.4f}'.format(corr_full['corrcoef']-corr_split['corrcoef']))

            # Save results
            with open(resultspath.joinpath(f'pair_{split:02d}_correlation.p'), 'wb') as f:
                pickle.dump(corr_split, f)

        i += 2

    plt.tight_layout()
    plt.savefig(figurepath.joinpath('split_data_comparison.png'))
    plt.show()


def main():
    results_split_data(pairs_to_split=[10, 6])


if __name__ == '__main__':
    main()
