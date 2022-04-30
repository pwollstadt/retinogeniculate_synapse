"""Estimate entropies and MI using a modified version of the pyentropy toolbox.

Modified version of the pyentropy toolbox published in

Ince, R. A. A., Petersen, R. S., Swan, D. C. and Panzeri, S. (2009)
"Python for Information Theoretic Analysis of Neural Data",
Frontiers in Neuroinformatics 3:4

https://github.com/robince/pyentropy

Some modifications for memory efficiency, functionality stays the same.


Copyright (C) 2013
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Patricia Wollstadt
11/01/2016
"""

from __future__ import division
import numpy as np


def ent(p):
    """Estimate entropy for an array of probabilities.

    Estimate entropies, there are two possible use cases:
        (1) simple entropy H(X): p represents probabilites of a single
        RV, where each entry is the probability of one symbol p(X=x)
        (2) conditional entropy H(X|Y): p is a list of arrays, where
        each array holds the conditional probabilities p(X=x|y) for one
        fixed y in Y

    Args:
        p (np array): array of probabilities (case 1)
        p (list, np arrays): list array of probabilities (case 2)

    Returns:
        float: entropy H(X) (case 1)
        np array, floats: entropy H(X|Y=y) for each y in Y (case 2)

    """
    # original code in pyentropy, needed much more memory
    # mp = np.ma.array(p,copy=False,mask=(p<=np.finfo(np.float).eps))
    # h1 =  -(mp*malog2(mp)).sum(axis=0)
    if type(p) is list:
        h = np.empty(len(p))
        for i in range(len(p)):
            mp = p[i][p[i] >= np.finfo(np.float).eps]
            h[i] = -(mp * np.log2(mp)).sum(axis=0)
    else:
        mp = p[np.finfo(np.float).eps <= p]
        h = -(mp * np.log2(mp)).sum(axis=0)
    return h



def prob(x, m, method='naive'):
    """Sample probability of integer sequence.

    Args:
        x (array, int): integer input sequence
        m (int): alphabet size of input sequence (max(x)<m)
        method: {'naive', 'kt', 'beta:x','shrink'}
            Sampling method to use.

    Returns:
        np array, float: array representing probability distribution
            Pr[i] = P(x=i)

    """
    if not np.issubdtype(x.dtype, np.integer):
        raise ValueError, "Input must be of integer type"
    if x.max() > m-1:
        raise ValueError, "Input contains values that are too large"

    C = np.bincount(x)
    if C.size < m:   # resize if any responses missed
        C.resize((m,))
    return _probcount(C, x.size, method)


def _probcount(C, N, method='naive'):
    """Estimate probability from a vector of bin counts

    Args:
        C (int array): integer vector of bin counts
        N (int): number of trials
        method: {'naive', 'kt', 'beta:x','shrink'}
            Sampling method to use.

    """
    N = float(N)
    if method.lower() == 'naive':
        # normal estimate
        P = C / N
    elif method.lower() == 'kt':
        # KT (constant addition) estimate
        P = (C + 0.5) / (N + (C.size/2.0))
    elif method.lower() == 'shrink':
        # James-Stein shrinkage
        # http://www.strimmerlab.org/software/entropy/index.html
        Pnaive = C / N
        target = 1./C.size
        lam = _get_lambda_shrink(N, Pnaive, target)
        P = (lam * target) + ((1 - lam) * Pnaive)
    elif method.split(':')[0].lower() == 'beta':
        beta = float(method.split(':')[1])
        # general add-constant beta estimate
        P = (C + beta) / (N + (beta*C.size))
    else:
        raise ValueError, 'Unknown sampling method: '+str(est)
    return P


def pt_bayescount(Pr, Nt):
    """Compute the support for analytic bias correction using the
    Bayesian approach of Panzeri and Treves (1996).

    Args:
        Pr (np array): Probability vector
        Nt (int): Number of trials

    Returns:
      int: Bayesian estimate of support
    """

    # dimension of space
    dim = Pr.size

    # non zero probs only
    PrNZ = Pr[Pr > np.finfo(np.float).eps]
    Rnaive = PrNZ.size

    R = Rnaive
    if Rnaive < dim:
        Rexpected = Rnaive - ((1.0-PrNZ)**Nt).sum()
        deltaR_prev = dim
        deltaR = np.abs(Rnaive - Rexpected)
        xtr = 0.0
        while (deltaR < deltaR_prev) and ((Rnaive+xtr)<dim):
            xtr = xtr+1.0
            Rexpected = 0.0
            # occupied bins
            gamma = xtr*(1.0 - ((Nt/(Nt+Rnaive))**(1.0/Nt)))
            Pbayes = ((1.0-gamma) / (Nt+Rnaive)) * (PrNZ*Nt+1.0)
            Rexpected = (1.0 - (1.0-Pbayes)**Nt).sum()
            # non-occupied bins
            Pbayes = gamma / xtr
            Rexpected = Rexpected + xtr*(1.0 - (1.0 - Pbayes)**Nt)
            deltaR_prev = deltaR
            deltaR = np.abs(Rnaive - Rexpected)
        Rnaive = Rnaive + xtr - 1.0
        if deltaR < deltaR_prev:
            Rnaive += 1.0
    return Rnaive


class BaseSystem:
    """Base functionality for entropy calculations common to all systems

    Provides basic functionality for the calculation of entropies and
    Panzeri-Treves correction.
    """


    def _calc_ents(self, method, sampling, methods):
        """Main entropy calculation function for non-QE methods"""

        self._sample(method=sampling)
        pt = (method == 'pt') or ('pt' in methods)
        plugin = (method == 'plugin') or ('plugin' in methods)
        calc = self.calc

        if (pt or plugin):
            self._calc_pt_plugin(pt)
        if method == 'plugin':
            self.H = self.H_plugin
        elif method == 'pt':
            self.H = self.H_pt

    def _calc_pt_plugin(self, pt):
        """Calculate direct entropies and apply PT correction if required """
        calc = self.calc
        pt_corr = lambda R: (R-1)/(2*self.N*np.log(2))
        self.H_plugin = {}
        if pt: self.H_pt = {}
        # compute basic entropies
        if 'HX' in calc:
            H = ent(self.PX)
            self.H_plugin['HX'] = H
            if pt:
                self.H_pt['HX'] = H + pt_corr(pt_bayescount(self.PX, self.N))
        if 'HY' in calc:
            H = ent(self.PY)
            self.H_plugin['HY'] = H
            if pt:
                self.H_pt['HY'] = H + pt_corr(pt_bayescount(self.PY, self.N))
        if 'HXY' in calc:
            H = (self.PY[self.PY >= np.finfo(np.float).eps] * ent(self.PXY)).sum()
            self.H_plugin['HXY'] = H
            if pt:
                for i in xrange(self.Y_occurrences.shape[0]):
                    H += pt_corr(pt_bayescount(self.PXY[i], self.Ny[i]))
                self.H_pt['HXY'] = H


    def calculate_entropies(self, method='plugin', sampling='naive',
                            calc=['HX','HXY'], **kwargs):
        """Calculate entropies of the system.

        :Parameters:
          method : {'plugin', 'pt', 'qe', 'nsb', 'nsb-ext', 'bub'}
            Bias correction method to use
          sampling : {'naive', 'kt', 'beta:x'}, optional
            Sampling method to use. 'naive' is the standard histrogram method.
            'beta:x' is for an add-constant beta estimator, with beta value
            following the colon eg 'beta:0.01' [1]_. 'kt' is for the
            Krichevsky-Trofimov estimator [2]_, which is equivalent to
            'beta:0.5'.

          calc : list of strs
            List of entropy values to calculate from ('HX', 'HY', 'HXY',
            'SiHXi', 'HiX', 'HshX', 'HiXY', 'HshXY', 'ChiX', 'HXY1','ChiXY1')

        :Keywords:
          qe_method : {'plugin', 'pt', 'nsb', 'nsb-ext', 'bub'}, optional
            Method argument to be passed for QE calculation ('pt', 'nsb').
            Allows combination of QE with other corrections.
          methods : list of strs, optional
            If present, method argument will be ignored, and all corrections
            in the list will be calculated. Use to comparing results of
            different methods with one calculation pass.

        :Returns:
          self.H : dict
            Dictionary of computed values.
          self.H_method : dict
            Dictionary of computed values using 'method'.

        Notes
        -----
        * If the PT method is chosen with outputs 'HiX' or 'ChiX' no bias
          correction will be performed for these terms.

        References
        ----------
        .. [1] T. Schurmann and P. Grassberger, "Entropy estimation of
           symbol sequences," Chaos,vol. 6, no. 3, pp. 414--427, 1996.
        .. [2] R. Krichevsky and V. Trofimov, "The performance of universal
           encoding," IEEE Trans. Information Theory, vol. 27, no. 2,
           pp. 199--207, Mar. 1981.


        """
        self.calc = calc
        self.methods = kwargs.get('methods',[])
        for m in (self.methods + [method]):
            if m not in ('plugin','pt','qe','nsb','nsb-ext','bub'):
                raise ValueError, 'Unknown correction method : '+str(m)
        methods = self.methods

        # allocate memory for requested calculations
        self.PX = np.zeros(self.X_dim)
        self.PY = np.zeros(self.Y_dim)
        self.PXY = [0] * self.Y_occurrences.shape[0]  # create a simple list for actual occurences only
        self._calc_ents(method, sampling, methods)

    def I(self, corr=None):
        """Convenience function to compute mutual information

        Must have already computed required entropies ['HX', 'HXY']

        :Parameters:
          corr : str, optional
            If provided use the entropies from this correction rather than
            the default values in self.H
        """
        try:
            if corr is not None:
                H = getattr(self,'H_%s'%corr.replace('-',''))
            else:
                H = self.H
            I = H['HX'] - H['HXY']
        except (KeyError, AttributeError):
            print "Error: must have computed HX and HXY for " + \
            "mutual information"
            return
        return I


class DiscreteSystem(BaseSystem):
    """Class to hold probabilities and calculate entropies of
    a discrete stochastic system.

    Attributes:
      PXY : (X_dim, Y_dim)
        Conditional probability vectors on decimalised space P(X|Y).
        ``PXY[:,i]`` is X probability distribution conditional on ``Y==i``.
      PX : (X_dim,)
        Unconditional decimalised X probability.
      PY : (Y_dim,)
        Unconditional decimalised Y probability.
      PXi : (X_m, X_n)
        Unconditional probability distributions for individual X components.
        ``PXi[i,j] = P(X_i==j)``
      PXiY : (X_m, X_n, Y_dim)
        Conditional probability distributions for individual X compoenents.
        ``PXiY[i,j,k] = P(X_i==j | Y==k)``
      PiX : (X_dim,)
        ``Pind(X) = <Pind(X|y)>_y``

    """

    def __init__(self, X, X_dims, Y, Y_dims, qe_shuffle=True):
        """Check and assign inputs.

        Args:
          X : (X_n, t)  int array
            Array of measured input values. X_n variables in X space, t trials
          X_dims : tuple (n, m)
            Dimension of X (input) space; length n, base m words
          Y : (Y_n, t) int array
            Array of corresponding measured output values. Y_n variables in Y
            space, t trials
          Y_dims : tuple (n ,m)
            Dimension of Y (output) space; length n, base m words
          qe_shuffle : {True, False}, optional
            Set to False if trials already in random order, to skip shuffling
            step in QE. Leave as True if trials have structure (ie one stimuli
            after another).

        """
        self.X_dims = X_dims
        self.Y_dims = Y_dims
        self.X_n = X_dims[0]
        self.X_m = X_dims[1]
        self.Y_n = Y_dims[0]
        self.Y_m = Y_dims[1]
        self.X_dim = self.X_m ** self.X_n
        self.Y_dim = self.Y_m ** self.Y_n
        self.X = np.atleast_2d(X)
        self.Y = np.atleast_2d(Y)
        self._check_inputs(self.X, self.Y)
        self.X_occurrences = np.unique(self.X)
        self.Y_occurrences = np.unique(self.Y)
        self.N = self.X.shape[1]
        # self.Ny = np.zeros(self.Y_dim)
        self.Ny = np.zeros(self.Y_occurrences.shape[0])
        self.qe_shuffle = qe_shuffle
        self.sampled = False
        self.calc = []


    def _check_inputs(self, X, Y):
        if (not np.issubdtype(X.dtype, np.int)) \
        or (not np.issubdtype(Y.dtype, np.int)):
            raise ValueError, "Inputs must be of integer type"
        if (X.max() >= self.X_m) or (X.min() < 0):
            raise ValueError, "X values must be in [0, X_m)"
        if (Y.max() >= self.Y_m) or (Y.min() < 0):
            raise ValueError, "Y values must be in [0, Y_m)"
        if (X.shape[0] != self.X_n):
            raise ValueError, "X.shape[0] must equal X_n"
        if (Y.shape[0] != self.Y_n):
            raise ValueError, "Y.shape[0] must equal Y_n"
        if (Y.shape[1] != X.shape[1]):
            raise ValueError, "X and Y must contain same number of trials"


    def _sample(self, method='naive'):
        """Sample probabilities of system.

       Args:
        method (string): {'naive', 'beta:x', 'kt'}, optional
            Sampling method to use. 'naive' is the standard histrogram method.
            'beta:x' is for an add-constant beta estimator, with beta value
            following the colon eg 'beta:0.01' [1]_. 'kt' is for the
            Krichevsky-Trofimov estimator [2]_, which is equivalent to
            'beta:0.5'.

        References
        ----------
        .. [1] T. Schurmann and P. Grassberger, "Entropy estimation of
           symbol sequences," Chaos,vol. 6, no. 3, pp. 414--427, 1996.
        .. [2] R. Krichevsky and V. Trofimov, "The performance of universal
           encoding," IEEE Trans. Information Theory, vol. 27, no. 2,
           pp. 199--207, Mar. 1981.

        """
        calc = self.calc

        # decimalise if necessary
        if self.X_n > 1:
            d_X = decimalise(self.X, self.X_n, self.X_m)
        else:
            # make 1D
            d_X = self.X.reshape(self.X.size)
        if self.Y_n > 1:
            d_Y = decimalise(self.Y, self.Y_n, self.Y_m)
        else:
            # make 1D
            d_Y = self.Y.reshape(self.Y.size)

        # unconditional probabilities
        self.PX = prob(d_X, self.X_dim, method=method)
        self.PY = prob(d_Y, self.Y_dim, method=method)

        # conditional probabilities
        """
        for i in xrange(self.Y_dim):
            indx = np.where(d_Y==i)[0]
            self.Ny[i] = indx.size
            if 'HXY' in calc:
                # output conditional ensemble
                oce = d_X[indx]
                if oce.size == 0:
                    print 'Warning: Null output conditional ensemble for ' + \
                      'output : ' + str(i)
                else:
                    # TODO this requires that the output of prob has fixed dim -> dict?, list of np arrays?
                    self.PXY[:,i] = prob(oce, self.X_dim, method=method)
        """
        for i in xrange(self.Y_occurrences.shape[0]):
            y = self.Y_occurrences[i]
            indx = np.where(d_Y == y)[0]
            self.Ny[i] = indx.size
            if 'HXY' in calc:
                # output conditional ensemble
                oce = d_X[indx]
                self.PXY[i] = prob(oce, self.X_dim, method=method)

        self.sampled = True


