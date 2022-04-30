#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Estimate PID using the Tartu estimator.

Usage: python3 rathbun_analysis_idtxl_pid.py

References:
    Makkeh, A., Theis, D. O., R. S., Vicente, R. (2018) "BROJA-2PID: A Robust
    Estimator for Bivariate Partial Information Decomposition", Entropy 2018,
    20(4), 271; https://doi.org/10.3390/e20040271
"""
import argparse
import pickle
import numpy as np

from idtxl.data import Data
from idtxl.estimators_pid import TartuPID

from utils import load_spike_trains, get_paths, load_mte_ais_estimates


def estimate(n_pair):
    """Estimate PID between TE past states and current value."""

    datapath, resultspath, _ = get_paths('paths.json')

    rgc, lgn = load_spike_trains(datapath, n_pair)
    data = Data(np.vstack((rgc, lgn)), 'ps', normalise=False)

    # Load LTE variables and realisations.
    mte, _ = load_mte_ais_estimates(resultspath, n_pair)
    words = np.load(resultspath.joinpath(f'pair_{n_pair:02d}_lte_words.npz'))

    source_past = words['source_past']
    target_past = words['target_past']
    target = 1
    current_value = mte.get_single_target(target, fdr=False)['current_value']
    current_value = data.get_realisations(current_value=current_value, idx_list=[current_value])[0]

    # PID estimation
    settings = {'alph_s1': np.max(source_past) + 1,
                'alph_s2': np.max(target_past) + 1,
                'alph_t': 2,
                'verbose': True}

    est_tartu = TartuPID(settings)
    pid_tartu = est_tartu.estimate(s1=source_past, s2=target_past, t=current_value)
    with open(resultspath.joinpath(f'pair_{n_pair:02d}_pid.p'), 'wb') as f:
        pickle.dump(pid_tartu, f)


def main():
    parser = argparse.ArgumentParser(description='Run PID estimation')
    parser.add_argument(
        'pair',
        type=int,
        help='ID of cell pair to analyze')

    args = parser.parse_args()
    estimate(args.pair)


if __name__ == '__main__':
    main()
