#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calculate local TE/AIS, encode and save past states for local PT correction.

Usage: python3 estimate_local_values.py
"""
import argparse
import numpy as np

from idtxl.data import Data
from idtxl.network_analysis import NetworkAnalysis
from idtxl.estimators_jidt import JidtDiscreteCMI

from utils import load_spike_trains, get_paths, load_mte_ais_estimates


def encode_words(x, alphabet_size):
    """Encode words in data strings as decimal numbers.

    Encode words of length x.shape[1] in the data array x as decimal numbers.
    Symbols in x are assumed to come from an alphabet with the size
    given in alphabet_size.

    Args:
        x (2D np array): string of symbols with a max. value < alphabet_size
        alphabet_size (int): size of the alphabet of symbols in x

    Returns:
        np array, int: encoded words

    Raises:
        ValueError: if the actual  alphabet_size of x exceeds the number
            given in alphabet_size
    """
    if np.unique(x).shape[0] > alphabet_size:
        raise ValueError('The number of unique values in x is larger than the alphabet size!')
    n_words = x.shape[0]
    k = x.shape[1]
    print(f'Returning encoded history of {n_words} observations')
    # Create a lookup table to save time when looping over words.
    lookup = np.power(alphabet_size, np.arange(k - 1, -1, -1))
    paststate = np.empty(n_words).astype(int)
    for n in range(n_words):
        for i in range(k-1, -1, -1):
            paststate[n] += x[n, i] * lookup[i]
    return paststate


def estimate(n_pair):
    """Estimate local values using optimized mTE and AIS embedding parameters.
    """
    datapath, resultspath, _ = get_paths('paths.json')

    settings = {'local_values': True}
    cmi_est = JidtDiscreteCMI(settings)

    # Load raw data and IDTxl AIS and TE results.
    rgc, lgn = load_spike_trains(datapath, n_pair)
    data = Data(np.vstack((rgc, lgn)), 'ps', normalise=False)
    mte, ais = load_mte_ais_estimates(resultspath, n_pair)
    if mte is None or ais is None:
        raise RuntimeError('Did not find results from prior analysis')

    target = 1
    mte_selected_sources = mte.get_single_target(
        target, fdr=False)['selected_sources_te']
    mte_selected_vars_sources = mte.get_single_target(
        target, fdr=False)['selected_vars_sources']
    mte_selected_vars_target = mte.get_single_target(
        target, fdr=False)['selected_vars_target']
    ais_selected_vars = ais.get_single_process(0, fdr=False)['selected_vars']

    if len(mte_selected_sources) == 0:
        RuntimeError('No sources selected during mTE analysis!')

    # Get information-transfer delay as the past variable with the max. CMI
    delay_ind = np.argmax(mte_selected_sources)

    if len(mte_selected_vars_sources) == 0:
        raise RuntimeError('No significant source variables, no TE.')
    delay = mte_selected_vars_sources[delay_ind][1]
    np.save(resultspath.joinpath(f'pair_{n_pair:02d}_delay'), delay)

    # Calculate local TE
    network_analysis = NetworkAnalysis()
    print(f'\nCalculate local TE for pair {n_pair}')
    cv_mte = mte.get_single_target(target, fdr=False)['current_value']
    current_value_realisations = data.get_realisations(
        current_value=cv_mte,
        idx_list=[cv_mte])[0]
    source_realisations = data.get_realisations(
        current_value=cv_mte,
        idx_list=network_analysis._lag_to_idx(
            mte_selected_vars_sources, cv_mte[1]))[0]
    target_past_realisations = data.get_realisations(
        current_value=cv_mte,
        idx_list=network_analysis._lag_to_idx(
            mte_selected_vars_target, cv_mte[1]))[0]

    local_te = cmi_est.estimate(var1=current_value_realisations,
                                var2=source_realisations,
                                conditional=target_past_realisations)
    print('Omnibus mTE: {0:.4f}, mean local TE: {1:.4f}'.format(
        mte.get_single_target(target, fdr=False)['omnibus_te'],
        np.mean(local_te)))
    assert np.isclose(
        mte.get_single_target(target, fdr=False)['omnibus_te'],
        np.mean(local_te),
        atol=0.0005
        ), 'TE and mean LTE diverge'
    np.save(resultspath.joinpath(f'pair_{n_pair:02d}_lte'), local_te)

    # Encode and save past states used for TE estimation.
    k_source = source_realisations.shape[1]
    k_target = target_past_realisations.shape[1]
    source_past = encode_words(source_realisations, 2)
    print('Source past words:', np.unique(source_past))
    target_past = encode_words(target_past_realisations, 2)
    print('Target past words:', np.unique(target_past))
    target_comb = encode_words(np.hstack(
        (target_past_realisations, current_value_realisations)), 2)
    print('Target combined words:', np.unique(target_comb))
    np.savez(resultspath.joinpath(f'pair_{n_pair:02d}_lte_words'),
             source_past=source_past, target_past=target_past,
             target_comb=target_comb, k_source=k_source, k_target=k_target)

    # Calculate local AIS.
    print(f'Calculate local AIS for pair {n_pair}')
    cv_ais = ais.get_single_process(0, fdr=False)['current_value']
    current_value_realisations = data.get_realisations(
        current_value=cv_ais,
        idx_list=[cv_ais])[0]
    source_realisations = data.get_realisations(
        current_value=cv_ais,
        idx_list=network_analysis._lag_to_idx(ais_selected_vars, cv_ais[1]))[0]
    local_ais = cmi_est.estimate(var1=current_value_realisations,
                                 var2=source_realisations,
                                 conditional=None)
    print('AIS: {0:.4f}, mean local AIS: {1:.4f}'.format(
        ais.get_single_process(0, fdr=False)['ais'], np.mean(local_ais)))
    assert np.isclose(ais.get_single_process(0, fdr=False)['ais'],
                      np.mean(local_ais)), (
        'AIS and mean LAIS diverge')
    np.save(resultspath.joinpath(f'pair_{n_pair:02d}_lais'), local_ais)

    # Encode and save past states used for AIS estimation.
    k_source = source_realisations.shape[1]
    source_past = encode_words(source_realisations, 2)
    print('Source past words:', np.unique(source_past))
    np.savez(resultspath.joinpath(f'pair_{n_pair:02d}_lais_words'),
             source_past=source_past,  k_source=k_source,
             current_value_realisations=current_value_realisations)


def main():
    parser = argparse.ArgumentParser(description='Run local TE and AIS estimation')
    parser.add_argument(
        'pair',
        type=int,
        help='ID of cell pair to analyze')

    args = parser.parse_args()
    estimate(args.pair)


if __name__ == '__main__':
    main()
