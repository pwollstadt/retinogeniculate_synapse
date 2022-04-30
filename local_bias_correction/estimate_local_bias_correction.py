#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calculate bias correction for local TE/AIS using pyentropy.

Usage: python2 rathbun_analysis_idtxl_bias_corr.py

References:
    Ince, R. A. A., Petersen, R. S., Swan, D. C. and Panzeri, S. (2009)
    "Python for Information Theoretic Analysis of Neural Data",
    Frontiers in Neuroinformatics 3:4
    https://code.google.com/archive/p/pyentropy/
"""
import sys
import json
from pathlib import Path
import numpy as np

# import pyentropy_modified.pyentropy_utils as utils
import pyentropy_modified.pypt as pypt

# Get cell pair from command line
try:
    n_pair = int(sys.argv[1])
except IndexError:
    raise IndexError('ERROR: enter the pair number as int')

# Set paths
with open('../lstc_analysis/paths.json') as f:
    paths = json.load(f)
datapath = Path(paths['datapath'])
resultspath = Path(paths['resultspath'])
if not resultspath.is_dir():
    resultspath.mkdir(parents=True)


# #############################################################################
#
# Calculate LTE correction
#
# #############################################################################

# load LTE estimate and encoded words used for LTE estimation (individual outcomes of variables).
lte = np.load(resultspath.joinpath('pair_{0:02d}_lte.npy'.format(n_pair)))
loadpath = resultspath.joinpath('pair_{0:02d}_lte_words.npz'.format(n_pair))

print '\nLoading data from ' + str(loadpath)
words = np.load(loadpath)
source_past = words['source_past']
target_past = words['target_past']
target_comb = words['target_comb']
k_source = words['k_source']  # length of past state embeddings
k_target = words['k_target']
base = 2

# Use pyentropy to calculate PT-corrected entropy terms. Pyentropy saves
# intermediate results which are later used to calculate local PT-correction.
s1 = pypt.DiscreteSystem(source_past, (1, base ** k_source),
                        target_comb, (1, base ** (k_target + 1)))
s2 = pypt.DiscreteSystem(source_past, (1, base ** k_source),
                        target_past, (1, base ** k_target))
s1.calculate_entropies(method='pt', calc=['HX', 'HY', 'HXY'],
                    methods=['plugin', 'pt'])
s2.calculate_entropies(method='pt', calc=['HX', 'HY', 'HXY'],
                    methods=['plugin', 'pt'])

# Calculate TE and *mean* correction terms from the pyentropy estimate. The
# mean correction terms are not used.
te = s1.I('plugin') - s2.I('plugin')
print '\tTE(RGC -> LGN) = ' + str(s1.I('plugin') - s2.I('plugin'))
print 'calculating corrected LTE (mean correction terms)'
te_correction_term1 = s1.I('plugin') - s1.I('pt')
te_correction_term2 = s2.I('plugin') - s2.I('pt')
print ('\tcorrection terms TE are ' + str(te_correction_term1) + ' and ' +
    str(te_correction_term2))
lte_corr_mean = lte - te_correction_term1 + te_correction_term2

# Calculate *local* correction terms from the pyentropy estimate.
print 'calculating local correction'

# term 1: I(X-;Y+,Y-)
pt_corr = lambda R: (R - 1) / (2 * s1.N * np.log(2))  # the actual correction
corr_hx_1 = pt_corr(pypt.pt_bayescount(s1.PX, s1.N))  # constant term for X
corr_hxy_1 = np.zeros(len(s1.Y_occurrences))  # y-dependent term
r_x = pypt.pt_bayescount(s1.PX, s1.N)  # estimated alphabet size of X
for y in xrange(len(s1.Y_occurrences)):   # loop over all y in Y
    r_y = pypt.pt_bayescount(s1.PXY[y], s1.Ny[y]) # local correction term for Y
    corr_hxy_1[y] = pt_corr(r_y) / float(r_x)  # divide term by alphabet size of X

# term 2: I(X-;Y-) (analogous to term 1)
pt_corr = lambda R: (R - 1) / (2 * s2.N * np.log(2))
corr_hx_2 = pt_corr(pypt.pt_bayescount(s2.PX, s2.N))
corr_hxy_2 = np.empty(len(s2.Y_occurrences))
r_x = pypt.pt_bayescount(s2.PX, s2.N)
for y in xrange(len(s2.Y_occurrences)):
    r_y = pypt.pt_bayescount(s2.PXY[y], s2.Ny[y])
    corr_hxy_2[y] = pt_corr(r_y) / float(r_x)

print 'applying local correction to every data point in the lte estimate'
# Collect local corrections for both terms for all samples in the spike train.
corr_terms = np.zeros(lte.shape[0])
term_1_alphabet = s1.Y_occurrences
term_2_alphabet = s2.Y_occurrences
for n in range(lte.shape[0]):
    # Get the index of the current occurrences in the alphabet taken from the
    # pyentropy analysis. The correction terms are in the same order as the
    # alphabet variables.
    target_comb_ind = term_1_alphabet == target_comb[n]
    target_past_ind = term_2_alphabet == target_past[n]
    # Combine local correction terms (constant and variable terms).
    corr_terms[n] = (corr_hx_1 - corr_hxy_1[target_comb_ind] +
                    corr_hx_2 - corr_hxy_2[target_past_ind])

# Apply local correction to LTE estimate.
lte_corr_local = lte + corr_terms
np.save(resultspath.joinpath('pair_{0:02d}_lte_corrected'.format(n_pair)),
        lte_corr_local)


# #############################################################################
#
# Calculate LAIS correction
#
# #############################################################################
lais = np.load(resultspath.joinpath('pair_{0:02d}_lais.npy'.format(n_pair)))
words = np.load(resultspath.joinpath('pair_{0:02d}_lais_words.npz'.format(n_pair)))
source_past = words['source_past']
current_value = np.squeeze(words['current_value_realisations'])
k_source = words['k_source']

print 'pyentropy AIS estimation'
s3 = pypt.DiscreteSystem(source_past, (1, base ** k_source),
                        current_value, (1, base))
s3.calculate_entropies(method='pt',
                    calc=['HX', 'HY', 'HXY'],  # 'HXY' = H(X|Y)
                    methods=['plugin', 'pt'])  # if this is provided, the method argument is overwritten
print('\tI(X;Y) = ' + str(s3.I('plugin')) + '\n\tH(X) = ' + str(s3.H['HX']) +
    '\n\tH(Y) = ' + str(s3.H['HY']))

# Calculate *mean* correction terms from the pyentropy estimate.
print 'calculating corrected LAIS (mean correction terms)'
ais_correction = s3.I('plugin') - s3.I('pt')
print 'correction term AIS is ' + str(ais_correction)
lais_corr_mean = lais - ais_correction  # substract mean correction

# Calculate *local* correction terms from the pyentropy estimate.
pt_corr = lambda R: (R - 1) / (2 * s3.N * np.log(2))  # the actual correction
corr_hx = pt_corr(pypt.pt_bayescount(s3.PX, s3.N))  # constant term for X
# corr_hy =  pt_corr(pt_bayescount(s3.PY, s3.N))
corr_hxy = np.zeros(len(s3.Y_occurrences))  # y-dependent term
r_x = pypt.pt_bayescount(s3.PX, s3.N)  # estimated alphabet size of X
for y in xrange(len(s3.Y_occurrences)):  # loop over all y in Y
    r_y = pypt.pt_bayescount(s3.PXY[y], s3.Ny[y])  # local correction term for Y
    corr_hxy[y] = pt_corr(r_y) / float(r_x)  # divide term by alphabet size of X

# Collect local corrections for all samples in the spike train.
corr_terms = np.zeros(lais.shape[0])
for n in range(lais.shape[0]):
    current_value_ind = s3.Y_occurrences == current_value[n]
    corr_terms[n] = corr_hx - corr_hxy[current_value_ind]

# Apply local correction to LAIS estimate.
lais_corr_local = lais + corr_terms
np.save(resultspath.joinpath('pair_{0:02d}_lais_corrected'.format(n_pair)),
        lais_corr_local)
