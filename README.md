# Information-theoretic analysis of spike trains from the retino-geniculate synapse

## Introduction & References

The scripts in this repository perform information-theoretic analyses of spike
trains recorded from the retinogeniculate synapse of the cat. Scripts produce
results on the correlation of local information storage and information
transfer at the synapse, which are presented in

- Wollstadt, P., Rathbun, D. L., Usrey, W. M., Moraes Bastos, A., Lindner, M.,
Priesemann, V., Wibral, M. (2022). Information-theoretic analyses of neural
data to minimize the effect of researchers' assumptions in predictive coding
studies. ArXiv preprint [arXiv:2203.10810
[q-bio.NC]](https://arxiv.org/abs/2203.10810)

The spike data used by the scripts is described in detail in

- Rathbun, D. L., Warland, D. K., Usrey, W. M. (2010). Spike Timing and
Information Transmission at Retinogeniculate Synapses. _Journal of
Neuroscience, 30_(41) 13558-13566.
https://doi.org/10.1523/JNEUROSCI.0909-10.2010

and can be obtained from

https://github.com/scottiealexander/PairsDB.jl/raw/main/data/

using the script `lstc_analysis/download_spike_data.py`.

The code uses the following python packages to perform information-theoretic
analyses:

- `IDTxl`: Wollstadt, P., Lizier, J. T., Vicente, R., Finn, C.,
Martinez-Zarzuela, M., Mediano, P., Novelli, L., Wibral, M. (2019). IDTxl: The
Information Dynamics Toolkit xl: a Python package for the efficient analysis of
multivariate information dynamics in networks. _Journal of Open Source
Software, 4_(34), 1081.
https://github.com/pwollstadt/IDTxl
- `pyentropy`: Ince, R. A. A., Petersen, R. S., Swan, D. C. and Panzeri, S.
(2009). Python for Information Theoretic Analysis of Neural Data",
_Frontiers in Neuroinformatics 3_(4)
https://code.google.com/archive/p/pyentropy/

## Requirements

Code in the folder `lstc_analysis` requires Python 3 and packages described in
the `lstc_analysis/requirements.txt`. Additionally, download and install
IDTxl [v1.4](https://github.com/pwollstadt/IDTxl/releases/tag/v1.4) (not yet on
pypi).

Code in the folder `local_bias_correction` requires Python 2 and packages
described in the `lstc_analysis/requirements.txt`. Additionally, the folder
contains a slightly modified version of the pyentropy toolbox, which is called
by the respective analysis scripts (see
https://code.google.com/archive/p/pyentropy/ for details and original source
code).

## Data

Raw data are obtained from the online repository

https://github.com/scottiealexander/PairsDB.jl/raw/main/data/

by calling
`python3 lstc_analysis/download_spike_data.py`. The script downloads raw spike
timings from the repository, generates spike train data, and saves it into the
data folder specified in `paths.json`. For details on the data see Rathbun et
al. (2010).

## Running the analysis

Data and output paths are defined in `paths.json`. All analysis scripts read in
this file. To change input- or output paths, simply modify the paths provided
in `paths.json`.

Prior to starting the analysis, data have to be downloaded via
`python3 lstc_analysis/download_spike_data.py`.

To estimate local, bias-corrected active information storage (AIS) and transfer
entropy, run the following analysis in the described order (note that step 3
calls an older version of pyentropy and requires __Python2__):

1. `python3 lstc_analysis/estimate_te_ais.py $cell_pair`: Run active
   information storage (AIS) and transfer entropy (TE) algorithms on input data
   to optimize non-uniform past-state embeddings. Estimate AIS in the RGC spike
   train and TE from RGC to LGN spike train.
2. `python3 lstc_analysis/estimate_local_values.py $cell_pair`: Calculate local
   TE/AIS, encode and save past states for local PT correction in the next step
3. `python2 local_bias_correction/estimate_local_bias_correction.py $cell_pair`:
   Calculate bias correction for local TE/AIS using pyentropy

To estimate the partial information decomposition of the TE source and target
past states, and the current value run

`python3 lstc_analysis/estimate_pid.py $cell_pair`

Each call to any of the scripts performs the respective analysis step for the
cell pair ID provided. To run all analyses over all cell pairs call the scripts

`./lstc_analysis/run_estimation.sh`
`./local_bias_correction/run_bias_correction.sh`

To generate figures and results from estimated information-theoretic quantities
for individual pairs, run:

- `python3 generate_results_correlation.py`: Calculate and plot local
  storage-transfer correlations (LSTC)
- `python3 generate_results_isi.py`: Calculate inter-spike intervals (ISI) and
  plot corresponding local AIS and TE estimates
- `python3 generate_results_sta.py`: Calculate spike-triggered averages (STA)
  for estimated local AIS and TE estimates
- `python3 generate_results_tuples.py`: Generate results for spike tuples

When all results have been generated, run
- `python plots_for_paper.py`: Generate plots and outputs from paper
- `python plot_classification_relayed.py`: Run classification of RGC spikes
  into relayed and non-relayed from AIS and spike stats

For convenience, the script

`./lstc_analysis/generate_all_results.sh`

runs all scripts to generate the results and Figures shown in the paper.

For details on the analysis refer to Wollstadt et al. (2022).
