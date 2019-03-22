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
# TODO Documentation


class PhenotypeData(object):
    """Container for phenotype data

    Pre-determines useful attributes for applying the EM-algorithm for
    haplotype frequency reconstruction based on unphased genotype (phenotype)
    data.

    """
    def __init__(self, sample=None):
        self.sample = sample
        self._haplotypes = None
        self._analysis = None
        self._genotypes = None
        self._genotype_matrix = None

    @classmethod
    def simulate(cls, proba_haplotypes, nobs=100):
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
        return self._genotype_matrix


def expectation_maximization_update(phenotype_data, proba_haplotypes):
    """Take one step of the EM algorithm

    :param phenotype_data: Pre-calculated data structure of the phenotype data
    :type phenotype_data: :class:`~haplopy.PhenotypeData`
    :param proba_haplotypes: Current estimate of the haplotype frequencies
    :type proba_haplotypes: :class:`~numpy.array`
    :return: Next estimate of the haplotype probabilities
    :rtype: :class:`~numpy.array`

    """
    nobs = sum(phenotype_data.analysis['counts'])  # number of observations
    proba_genotypes = np.array([])
    for i, genotypes in enumerate(
            phenotype_data.analysis['genotype_expansion']):
        proba = np.array(
            [2 ** (i == j) * proba_haplotypes[i] * proba_haplotypes[j]
             for (i, j) in genotypes]
        )
        proba /= sum(proba)
        proba *= phenotype_data.analysis['counts'][i] / nobs
        proba_genotypes = np.concatenate([proba_genotypes, proba])
    return .5 * phenotype_data.genotype_matrix.dot(proba_genotypes)


def expectation_maximization(phenotype_data, proba_haplotypes=None,
                             max_iter=20):
    """EM iteration for haplotype frequency estimation

    Calculates the Maximum Likelihood estimate of the haplotype frequencies.

    TODO Math formula of likelihood

    The source data is unphased genotype data (phenotype).

    :param phenotype_data: Pre-calculated data structure of the phenotype data
    :type phenotype_data: :class:`~haplopy.PhenotypeData`
    :param proba_haplotypes: Initial estimate of the haplotype frequencies
    :type proba_haplotypes: :class:`~numpy.array`
    :param max_iter: Maximum number of iterations
    :type max_iter: int
    :return: Estimated haplotype frequencies
    :rtype: :class:`~numpy.array`

    """
    if proba_haplotypes is None:  # equal frequencies as initial guess
        nhaplotypes = len(phenotype_data.haplotypes)
        proba_haplotypes = np.ones(nhaplotypes) / nhaplotypes
    for i in range(max_iter):
        proba_haplotypes = expectation_maximization_update(phenotype_data,
                                                           proba_haplotypes)
    res = {''.join(k): v for k, v in zip(phenotype_data.haplotypes,
                                         proba_haplotypes)}
    return res
