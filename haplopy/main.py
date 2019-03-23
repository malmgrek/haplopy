# -*- coding: utf-8 -*-

"""
haplopy.main
~~~~~~~~~~~~

This module implements the HaploPy

:copyright: (c) 2019 by Stratos Staboulis
:license: MIT
"""

import collections
import itertools
import logging

import numpy as np
from scipy.sparse import dok_matrix


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


__version__ = '0.0.1.dev'
__author__ = 'Stratos Staboulis'


# TODO Logging info
# TODO Log-likelihood conditiion in iteration
# TODO Randomized initial guess in iteration
# TODO Imputation example
# TODO Unit-tests
# TODO Sphinx documentation to GitHub


def log_binomial(n, m):
    """Logarithm of binomial coefficient using Stirling approximation

    """
    return n * np.log(n) - m * np.log(m) - (n - m) * np.log(n - m) + \
           0.5 * (np.log(n) - np.log(m) - np.log(n - m) - np.log(2 * np.pi))


def log_multinomial(*args):
    """Logarithm of the multinomial coefficient

    """
    if len(args) == 1:
        return 1
    return log_binomial(sum(args), args[-1]) + log_multinomial(args[:-1])


class PhenotypeData(object):
    """Container for phenotype data

    Pre-determines useful attributes for applying the EM-algorithm for
    haplotype frequency reconstruction based on unphased genotype (phenotype)
    data.

    # TODO code example

    :param sample: Measured unphased genotype data
    :type sample:

    """
    def __init__(self, sample=None):
        self.sample = sample
        self._haplotypes = None
        self._analysis = None
        self._genotypes = None
        self._genotype_matrix = None
        self._log_likelihood_const = None

    @classmethod
    def simulate(cls, proba_haplotypes, nobs=100):
        """Simulate a set of phenotype observations

        Draws from multinomial distribution using pre-determined haplotype
        frequencies.

        # TODO code example

        :param proba_haplotypes: Known haplotype frequencies
        :type proba_haplotypes: :class:`numpu.ndarray`-like
        :param nobs: Number of generated observations
        :type nobs: int
        :return: Instance of :class:`PhenotypeData` with the simulated data
                 in :attr:`sample`

        """
        proba_haplotypes /= proba_haplotypes.sum()
        # random set of indices
        inds = np.dot(
            np.random.multinomial(1, proba_haplotypes, 2 * nobs),
            np.arange(proba_haplotypes.size)
        )
        sample = []
        for i, j in inds.reshape(nobs, 2):
            sample.append(
                tuple('{}{}'.format(*sorted(x))
                      for x in zip(proba_haplotypes.index[i],
                                   proba_haplotypes.index[j]))
            )
        logger.info('Simulated phenotype data of {0} observations'.format(
            nobs))
        return cls(sample=sample)

    @property
    def haplotypes(self):
        """All haplotypes compatible with :attr:`sample`

        :return: Sequence of haplotypes in unspecified order.
        :rtype: list

        """
        if self._haplotypes is None:
            unique_phenotypes = set(self.sample)
            haplotypes = set()
            for phenotype in unique_phenotypes:
                haplotypes = haplotypes.union(
                    set(itertools.product(*[set(loc) for loc in phenotype]))
                )
            self._haplotypes = list(haplotypes)
            logger.info('Constructed haplotypes data')
        return self._haplotypes

    @property
    def analysis(self):
        """Analysis of the phenotype sample

        :return: Useful information deduced from the phenotype data in
                 :attr:`sample`.
        :rtype: dict

        """
        if self._analysis is None:
            counter = collections.Counter(self.sample)
            unique_phenotypes = list(counter.keys())
            genotype_expansion = list()
            for phenotype in unique_phenotypes:
                # the below call returns a list of haplotypes
                # aligning the top and (flipped) bottom halves gives directly
                # the admissible genotype combinations
                factors = list(itertools.product(*[set(loc)
                                                   for loc in phenotype]))
                mid = len(factors) // 2
                if mid == 0:
                    genotype_expansion.append(
                        [(self.haplotypes.index(factors[0]),
                          self.haplotypes.index(factors[0]))]
                    )
                else:
                    genotype_expansion.append(
                        [(self.haplotypes.index(x), self.haplotypes.index(y))
                         for x, y in zip(factors[:mid], factors[mid:][::-1])]
                    )
            self._analysis = {'counts': list(counter.values()),
                              'phenotypes': unique_phenotypes,
                              'genotype_expansion': genotype_expansion}
            logger.info('Constructed phenotype analysis data')
        return self._analysis

    @property
    def genotypes(self):
        """List all genotypes compatible with the phenotype sample

        :return: List of genotypes compatible with the phenotype data in
                 :attr:`sample`. The genotypes are referred to in terms of the
                 natural index of :attr:`haplotypes`
        :rtype: list

        """
        if self._genotypes is None:
            genotypes = [x for y in self.analysis['genotype_expansion']
                         for x in y]
            self._genotypes = genotypes
            logger.info('Constructed genotypes data')
        return self._genotypes

    @property
    def genotype_matrix(self):
        """Count haplotype occurrences in each compatible genotype

        :return: Sparse matrix of shape (num haplotypes) x (num genotypes)
        :rtype: scipy.sparse.dok_matrix

        """
        if self._genotype_matrix is None:
            genotype_matrix = dok_matrix(
                (len(self.haplotypes), len(self.genotypes)),
                dtype=np.int
            )
            for i, genotype in enumerate(self.genotypes):
                genotype_matrix[genotype[0], i] += 1
                genotype_matrix[genotype[1], i] += 1
            self._genotype_matrix = genotype_matrix
            logger.info('Constructed genotype occurrence matrix')
        return self._genotype_matrix

    @property
    def log_likelihood_const(self):
        if self._log_likelihood_const is None:
            log_likelihood_const = log_multinomial(*self.analysis['counts'])
            self._log_likelihood_const = log_likelihood_const
        return self._log_likelihood_const


def expectation_maximization_update(phenotype_data, proba_haplotypes):
    """Progress the EM algorithm one step forward

    :param phenotype_data: Pre-calculated data structure of the phenotype data
    :type phenotype_data: :class:`~haplopy.PhenotypeData`
    :param proba_haplotypes: Current estimate of the haplotype frequencies
    :type proba_haplotypes: :class:`~numpy.array`
    :return: Next estimate of the haplotype probabilities and _previous_
             Log-Likelihood
    :rtype: (:class:`numpy.ndarray`, float)

    """
    nobs = sum(phenotype_data.analysis['counts'])  # number of observations
    proba_genotypes = np.array([])
    log_likelihood = phenotype_data.log_likelihood_const
    for i, genotypes in enumerate(
            phenotype_data.analysis['genotype_expansion']):
        proba = np.array(
            [2 ** (i != j) * proba_haplotypes[i] * proba_haplotypes[j]
             for (i, j) in genotypes]
        )
        log_likelihood += phenotype_data.analysis['counts'][i] * \
                          np.log(proba.sum())
        proba /= sum(proba)
        proba *= phenotype_data.analysis['counts'][i] / nobs
        proba_genotypes = np.concatenate([proba_genotypes, proba])
    proba_haplotypes = .5 * phenotype_data.genotype_matrix.dot(
        proba_genotypes)
    return proba_haplotypes, log_likelihood


def expectation_maximization(phenotype_data, proba_haplotypes=None,
                             max_iter=20, tol=1.0e-5,
                             logging_threshold=1.0e-5):
    """EM iteration for haplotype frequency estimation

    Calculates the Maximum Likelihood estimate of the haplotype frequencies.

    TODO Math formula of likelihood
    TODO Code example

    The source data is unphased genotype data (phenotype).

    :param phenotype_data: Pre-calculated data structure of the phenotype data
    :type phenotype_data: :class:`~haplopy.PhenotypeData`
    :param proba_haplotypes: Initial estimate of the haplotype frequencies
    :type proba_haplotypes: :class:`~numpy.array`
    :param max_iter: Maximum number of iterations
    :type max_iter: int
    :param tol: Log Likelihood convergence tolerance
    :type tol: float
    :param logging_threshold: Threshold for logging small results
    :type logging_threshold: float
    :return: Estimated haplotype frequencies
    :rtype: :class:`~numpy.array`

    """
    logger.info('Start EM algorithm')
    if proba_haplotypes is None:  # equal frequencies as initial guess
        logger.info('Auto-generated uniform initial guess')
        nhaplotypes = len(phenotype_data.haplotypes)
        proba_haplotypes = np.ones(nhaplotypes) / nhaplotypes
    objective_values = list()
    i = 0
    delta = np.inf
    while i < max_iter:
        proba_haplotypes, log_likelihood = expectation_maximization_update(
            phenotype_data, proba_haplotypes
        )
        objective_values.append(log_likelihood)
        logger.info(
            'Iteration {0}: log(L) = {1}'.format(i, log_likelihood)
        )
        if i > 0:
            delta = objective_values[-1] - objective_values[-2]
        if delta < tol:
            logger.info('Convergence up to tolerance level {0}'.format(tol))
            break
        i += 1
    res = {''.join(k): v for k, v in zip(phenotype_data.haplotypes,
                                         proba_haplotypes)}
    if i == max_iter:
        logger.info('EM algorithm terminated after maximum number of {0} '
                    'iterations.'.format(i + 1))
    # Additional log message for results
    msg = '\n\nResults\n' +\
              '=======\n' +\
          ''.join(['{0}: {1}\n'.format(k, v)
                   for k, v in res.items() if v > logging_threshold])
    logger.debug(msg)
    return res
