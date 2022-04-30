"""Utility functions for LSTC analysis."""
import json
import pickle
from pathlib import Path
import numpy as np


def get_paths(filename):
    """Read paths from json config file

    Args:
        filename (str): File name

    Returns:
        pathlib.Path: Path to input data
        pathlib.Path: Path to generated results
        pathlib.Path: Path to generated figures
    """
    with open(filename) as f:
        paths = json.load(f)
    datapath = Path(paths['datapath'])
    datapath.mkdir(parents=True, exist_ok=True)
    resultspath = Path(paths['resultspath'])
    resultspath.mkdir(parents=True, exist_ok=True)
    figurepath = Path(paths['figurepath'])
    figurepath.mkdir(parents=True, exist_ok=True)
    return datapath, resultspath, figurepath


def load_spike_trains(datapath, pair):
    """Load raw spike trains for a cell pair

    Args:
        datapath (pathlib.Path): Folder containing raw data
        pair (int): Cell pair ID

    Returns:
        numpy.ndarray: Spike train RGC
        numpy.ndarray: Spike train LGN
    """
    raw_data = np.load(datapath.joinpath(f'pair_{pair:02d}_raw_data.npz'))
    rgc = raw_data['rgc'].astype(int)
    lgn = raw_data['lgn'].astype(int)
    # Remove bins with 2s (happens very rarely).
    rgc[rgc > 1] = 1
    lgn[lgn > 1] = 1
    assert max(rgc) == 1, 'RGC spike train not binary'
    assert max(lgn) == 1, 'LGN spike train not binary'
    return rgc, lgn


def load_mte_ais_estimates(resultspath, n_pair):
    """Load mTE and AIS estimates.

    Args:
        resultspath (pathlib.Path): Path to results files
        n_pair (int): Cell pair ID

    Returns:
        idtxl.NetworkInferenceResults: mTE estimation results
        idtxl.NetworkInferenceResults: AIS estimation results
    """
    try:
        filename = resultspath.joinpath(f'pair_{n_pair:02d}_mte.p')
        with open(filename, 'rb') as f:
            mte = pickle.load(f)
    except FileNotFoundError as e:
        print(f'{e}')
        mte = None
    try:
        filename = resultspath.joinpath(f'pair_{n_pair:02d}_ais.p')
        with open(filename, 'rb') as f:
            ais = pickle.load(f)
    except FileNotFoundError as e:
        print(f'{e}')
        ais = None
    return mte, ais


def load_local_estimates(resultspath, n_pair):
    """Load local mTE and AIS estimates.

    Args:
        resultspath (pathlib.Path): Path to results files
        n_pair (int): Cell pair ID

    Returns:
        numpy.ndarray: local mTE estimation results
        numpy.ndarray: local AIS estimation results
        int: TE delay
    """
    try:
        filename = resultspath.joinpath(f'pair_{n_pair:02d}_lte_corrected.npy')
        lte = np.load(filename)
    except FileNotFoundError as e:
        print(f'{e}')
        lte = None
    try:
        filename = resultspath.joinpath(f'pair_{n_pair:02d}_lais_corrected.npy')
        lais = np.load(filename)
    except FileNotFoundError as e:
        print(f'{e}')
        lais = None
    try:
        filename = resultspath.joinpath(f'pair_{n_pair:02d}_delay.npy')
        delay = np.load(filename)
    except FileNotFoundError as e:
        print(f'{e}')
        delay = None
    return lte, lais, delay
