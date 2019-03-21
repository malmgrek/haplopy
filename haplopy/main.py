import itertools
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


__version__ = '0.0.1.dev'
__author__ = 'Stratos Staboulis'


# TODO Strings to numbers


def simulate(haplotypes, nobs=100):
    """Simulate unphased genotype data

    :param haplotypes: Population haplotypes and their relative frequencies.
    :type haplotypes: pandas.Series
    :param nobs: Number of observations to be generated
    :type nobs: int
    :return: List of unphased genotypes

    TODO testme

    """
    haplotypes /= haplotypes.sum()  # must sum up to 1
    samples = np.random.multinomial(1, haplotypes, 2 * nobs)
    inds = np.dot(samples, np.arange(haplotypes.size))  # random set of indices
    phenotypes = []
    for i, j in inds.reshape(nobs, 2):
        phenotypes.append(
            tuple('{}{}'.format(*sorted(x))
                  for x in zip(haplotypes.index[i], haplotypes.index[j]))
        )
    return pd.Series(phenotypes)


def expand(phenotypes):
    """Find possible haplotypes and genotypes

    The genotype information is returned as a look-up to the haplotypes list.

    :param phenotypes: List of observed phenotypes of same length
    :type phenotypes: list
    :return: (haplotypes, genotypes)
    :rtype: tuple

    TODO testme

    """
    # retrieve all possible constituent haplotypes
    haplotypes = set()
    for phenotype in phenotypes:
        haplotypes = haplotypes.union(
            set(itertools.product(*[set(loc) for loc in phenotype]))
        )
    haplotypes = list(haplotypes)
    # lookup from haplotypes to genotype pairs
    genotypes = list()
    for phenotype in phenotypes:
        factors = list(itertools.product(*[set(loc) for loc in phenotype]))
        mid = len(factors) // 2
        if mid == 0:
            genotypes.append(
                [(haplotypes.index(factors[0]), haplotypes.index(factors[0]))]
            )
        else:
            genotypes.append(
                [(haplotypes.index(x), haplotypes.index(y))
                 for x, y in zip(factors[:mid], factors[mid:][::-1])]
            )
    return pd.Series(haplotypes), genotypes


def expectation(proba_haplotypes, genotypes):
    """Expectation step of the EM algorithm

    TODO: Math formula here

    :param proba_haplotypes: Haplotype frequency estimates
    :param genotypes: Representation as haplotype indices
    :return:

    TODO testme

    """
    proba_genotypes = list()
    for genotype in genotypes:
        proba_genotypes.append(
            [2 ** (i == j) * proba_haplotypes[i] * proba_haplotypes[j]
             for (i, j) in genotype]
        )
    return proba_genotypes
