#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calculate TE/AIS using the non-uniform embedding in IDTxl.

Usage: python3 estimate_te_ais.py
"""
import argparse
import pickle
import numpy as np

from idtxl.active_information_storage import ActiveInformationStorage
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data

from utils import get_paths, load_spike_trains


def estimate(n_pair):
    """Estimate mTE and AIS for spike trains."""
    datapath, resultspath, _ = get_paths('paths.json')

    # Load data and create IDTxl data object.
    rgc, lgn = load_spike_trains(datapath, n_pair)
    data = Data(np.vstack((rgc, lgn)), 'ps', normalise=False)

    # Estimate TE from RGC to LGN spike train.
    print('Start TE estimation')
    settings = {
        'cmi_estimator': 'JidtDiscreteCMI',
        'alph1': 2,  # define alphabet size for input data
        'alph2': 2,
        'alphc': 2,
        'n_perm_max_stat': 200,
        'n_perm_min_stat': 200,
        'max_lag_target': 30,  # spikes up to 30 ms in the past influence spiking behavior (Rathbun, 2010)
        'max_lag_sources': 40,  # we expect delays up to 10 ms based on cross-correlation reported in Rathbun (2010)
        'min_lag_sources': 1,
        'tau': 1
        }
    # The toolbox creates analytical surrogates for the discrete estimator.
    mte_analysis = MultivariateTE()
    res_mte = mte_analysis.analyse_single_target(settings, data, target=1)
    with open(resultspath.joinpath(f'pair_{n_pair:02d}_mte.p'), 'wb') as f:
        pickle.dump(res_mte, f)

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
        'tau': 1
        }
    ais_analysis = ActiveInformationStorage()
    res_ais = ais_analysis.analyse_single_process(settings, data, process=0)
    with open(resultspath.joinpath(f'pair_{n_pair:02d}_ais.p'), 'wb') as f:
        pickle.dump(res_ais, f)


def main():
    parser = argparse.ArgumentParser(description='Run TE and AIS estimation')
    parser.add_argument(
        'pair',
        type=int,
        help='ID of cell pair to analyze')

    args = parser.parse_args()
    estimate(args.pair)


if __name__ == '__main__':
    main()
