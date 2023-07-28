import numpy as np

from utils import get_paths, load_spike_trains


def split_data(n_pair):
    """Load raw spike trains and split in half."""
    datapath, _, _ = get_paths('paths.json')
    rgc, lgn = load_spike_trains(datapath, n_pair)
    assert len(rgc) == len(lgn)
    n_half = len(rgc) // 2
    print(f'Splitting pair {n_pair} at {n_half} samples (total length: {len(rgc)} samples)')
    rgc_half_1 = rgc[:n_half]
    lgn_half_1 = lgn[:n_half]
    rgc_half_2 = rgc[n_half:]
    lgn_half_2 = lgn[n_half:]
    savename_half_1 = f'pair_{n_pair*3:02d}_raw_data.npz'
    savename_half_2 = f'pair_{n_pair*3+1:02d}_raw_data.npz'
    print(f'Saving split data as {savename_half_1}, {savename_half_2}')
    np.savez(
        datapath.joinpath(savename_half_1),
        rgc=rgc_half_1,
        lgn=lgn_half_1
        )
    np.savez(
        datapath.joinpath(savename_half_2),
        rgc=rgc_half_2,
        lgn=lgn_half_2
        )


def main():
    split_data(n_pair=10)
    split_data(n_pair=6)


if __name__ == '__main__':
    main()
