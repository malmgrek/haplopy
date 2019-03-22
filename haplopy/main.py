import collections
import itertools
import logging

import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


__version__ = '0.0.1.dev'
__author__ = 'Stratos Staboulis'


# TODO Strings to numbers
# TODO Independence of Pandas


class PhenotypeData(object):
    """Container for phenotype data

    All integer index-like values refer to the natural index of
    :attr:`PhenotypeData.haplotypes`.

    """
    def __init__(self, sample=None, **kwargs):
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

    @classmethod
    def read_csv(cls, **kwargs):
        raise NotImplementedError

    @property
    def haplotypes(self):
        """The haplotypes present in the sample

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

        """
        if self._analysis is None:
            counter = collections.Counter(self.sample)
            unique_phenotypes = list(counter.keys())
            genotypes = list()
            for phenotype in unique_phenotypes:
                # the below call returns a list of haplotypes
                # aligning the top and (flipped) bottom halves gives directly
                # the admissible genotype combinations
                factors = list(itertools.product(*[set(loc)
                                                   for loc in phenotype]))
                mid = len(factors) // 2
                if mid == 0:
                    genotypes.append(
                        [(self.haplotypes.index(factors[0]),
                          self.haplotypes.index(factors[0]))]
                    )
                else:
                    genotypes.append(
                        [(self.haplotypes.index(x), self.haplotypes.index(y))
                         for x, y in zip(factors[:mid], factors[mid:][::-1])]
                    )
            self._analysis = {'counts': list(counter.values()),
                              'phenotypes': unique_phenotypes,
                              'genotypes': genotypes}
        return self._analysis

    @property
    def genotypes(self):
        """List all genotypes compatible with the phenotype sample

        """
        if self._genotypes is None:
            genotypes = [x for y in self.analysis['genotypes'] for x in y]
            self._genotypes = genotypes
        return self._genotypes

    @property
    def genotype_matrix(self):
        """Count haplotype occurrences in each compatible genotype

        """
        if self._genotype_matrix is None:
            genotype_matrix = dok_matrix(
                (len(self.genotypes), len(self.haplotypes)),
                dtype=np.int
            )
            for i, genotype in enumerate(self.genotypes):
                genotype_matrix[i, genotype[0]] += 1
                genotype_matrix[i, genotype[1]] += 1
            self._genotype_matrix = genotype_matrix
        return self._genotype_matrix


def proba_em(proba_haplotypes, phenotype_data):
    """Expectation step of the EM algorithm

    TODO: Math formula here

    :param proba_haplotypes: Haplotype frequency estimates
    :param genotypes: Representation as haplotype indices
    :return:

    TODO testme

    """
    num_phenotypes = phenotype_data['phenotype_count'].sum()
    proba_genotypes = list()
    for i, genotypes in enumerate(phenotype_data['genotypes']):
        proba = [2 ** (i == j) * proba_haplotypes[i] * proba_haplotypes[j]
                 for (i, j) in genotypes]
        proba = np.divide(proba, sum(proba))
        proba = np.multiply(
            proba,
            phenotype_data['phenotype_count'].iloc[i] / num_phenotypes
        )
        proba_genotypes.append(proba)  # TODO fix proba_genotypes to relevant
    return proba_genotypes
