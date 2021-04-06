"""Haplotype imputation model

"""

import collections
from functools import reduce
import itertools
import logging
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.sparse import dok_matrix

from haplopy import datautils


# TODO Randomized initial guess in iteration
# TODO Imputation example
# TODO Try BayesPy: We can directly use analysis content:
#      - genotype_expansion gives the "draws" such that each
#        index is a "color". Sample numbers are obtained by
#        multiplying each index occurrences with "counts". Perhaps
#        collections.Counter can be used more directly.
#      - Maybe reduce the itertools.product thingy.
# TODO Unit-tests


def log_binomial(n: int, k: int):
    """Logarithm of binomial coefficient using Stirling approximation

    TODO: Unit test

    """
    return (
        n * np.log(n) - k * np.log(k) - (n - k) * np.log(n - k)
        + 0.5 * (np.log(n) - np.log(k) - np.log(n - k) - np.log(2 * np.pi))
    )


def log_multinomial(*args):
    """Logarithm of the multinomial coefficient

    TODO: Unit test

    """
    if len(args) == 1:
        return 1
    return log_binomial(sum(args), args[-1]) + log_multinomial(args[:-1])


def expectation_maximization(
        phenotypes: List[Tuple[str]],
        randomize_init: bool=False,
        max_iter: int=100,
        tol: float=1.0e-6,
        logging_threshold: float=1.0e-6
):

    # TODO: Set initial p_haplotypes
    # TODO: Make sure that haplotype order (dict) is not lost

    n_obs = len(phenotypes)

    (
        parent_haplotypes,
        counter,
        genotype_expansion,
    ) = datautils.describe_phenotypes(phenotypes)

    # Genotype counting matrix for fast multiplication
    genotype_matrix = datautils.build_genotype_matrix(
        genotype_expansion, parent_haplotypes
    )

    # Log-likelihood constant
    C = log_multinomial(*counter.values)

    def update_haplotype(genotypes, counts):
        # Update step for a single phenotype
        # TODO / FIXME: At the end of the day this is just inside a loop.
        #               So do we really need the complex description / expansion
        #               methods? Could we just do everything once in here?
        # NOTE: It makes sense to pre-compute the indexation and counting stuff.
        #       Otherwise we would be computing it repetitively within the loop.

        return

    def update(p_haplotypes):
        # Calculates the next estimate of haplotype probabilities and previous
        # Log-likelihood
        #
        # TODO: Update haplotype probabilities
        #
        p_haplotypes = None
        return p_haplotypes

    return


class Model():

    def __init__(self, p_haplotypes: Dict[str, float]=None):
        (haplotypes, ps) = zip(*p_haplotypes.items())
        assert sum(ps) == 1, "Probabilities must sum to one"
        self.p_haplotypes = p_haplotypes
        return

    def fit(self, phenotypes: Dict[str, float]):
        """Expectation Maximization

        """
        p_haplotypes = None
        return HaplotypeMultinomial(p_haplotypes)

    def random(self, n_obs: int) -> List[Tuple[str]]:
        """Random generate phenotypes

        """
        (haplotypes, ps) = zip(*self.p_haplotypes.items())
        parent_inds = np.dot(
            np.random.multinomial(1, ps, 2 * n_obs),
            np.arange(len(ps))
        ).reshape(n_obs, 2)
        return [
            tuple(
                "".join(sorted(diplo))
                for diplo in zip(haplotypes[i], haplotypes[j])
            )
            for (i, j) in parent_inds
        ]


class PhenotypeData(object):
    """Container for phenotype data

    Pre-determines useful attributes for applying the EM-algorithm for
    haplotype frequency reconstruction based on unphased genotype (phenotype)
    data.

    # TODO code example

    Parameters
    ----------

    sample : List[Tuple[str]]
        Measured unphased genotype data

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

        Parameters
        ----------

        proba_haplotypes : dict[str, float]
            Known haplotype frequencies
        nobs : int
            Number of generated observations

        """
        haplotypes = list(proba_haplotypes)
        proba = np.array(list(proba_haplotypes.values()))
        proba = proba / proba.sum()
        # random set of indices
        inds = np.dot(
            np.random.multinomial(1, proba, 2 * nobs),
            np.arange(proba.size)
        )
        sample = []
        for i, j in inds.reshape(nobs, 2):
            sample.append(
                tuple('{}{}'.format(*sorted(x))
                      for x in zip(haplotypes[i],
                                   haplotypes[j]))
            )
        logging.info('Simulated phenotype data of {0} observations'.format(
            nobs))
        return cls(sample=sample)

    @property
    def haplotypes(self) -> list:
        """All haplotypes compatible with :attr:`sample`

        """
        if self._haplotypes is None:
            unique_phenotypes = set(self.sample)
            haplotypes = set()
            for phenotype in unique_phenotypes:
                haplotypes = haplotypes.union(
                    set(itertools.product(*[set(loc) for loc in phenotype]))
                )
            self._haplotypes = list(haplotypes)
            logging.info('Constructed haplotypes data')
        return self._haplotypes

    @property
    def analysis(self) -> dict:
        """Useful information deduced from the sample phenotype data

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
            logging.info('Constructed phenotype analysis data')
        return self._analysis

    @property
    def genotypes(self) -> list:
        """List all genotypes compatible with the phenotype sample

        Genotypes are referred to in terms of the natural index of haplotypes

        """
        if self._genotypes is None:
            genotypes = [x for y in self.analysis['genotype_expansion']
                         for x in y]
            self._genotypes = genotypes
            logging.info('Constructed genotypes data')
        return self._genotypes

    @property
    def genotype_matrix(self):
        """Count haplotype occurrences in each compatible genotype

        Returns an instance of scipy.sparse.dok_matrix

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
            logging.info('Constructed genotype occurrence matrix')
        return self._genotype_matrix

    @property
    def log_likelihood_const(self):
        if self._log_likelihood_const is None:
            log_likelihood_const = log_multinomial(*self.analysis['counts'])
            self._log_likelihood_const = log_likelihood_const
        return self._log_likelihood_const


def expectation_maximization_update(
        phenotype_data,
        proba_haplotypes
) -> np.ndarray:
    """Progress the EM algorithm one step forward

    Calculates the next estimate of haplotype probabilities and _previous_
    Log-likelihood

    Parameters
    ----------

    phenotype_data : PhenotypeData
        Pre-calculated data structure of the phenotype data
    proba_haplotypes : Dict[str, float]
        Current estimate of the individual haplotype frequencies

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


def expectation_maximization(
        phenotype_data,
        proba_haplotypes=None,
        max_iter=20,
        tol=1.0e-5,
        logging_threshold=1.0e-5
) -> Dict[str, float]:
    """EM iteration for haplotype frequency estimation

    Calculates the Maximum Likelihood estimate of the haplotype frequencies.
    The source data is unphased genotype data (phenotype).

    TODO Math formula of likelihood
    TODO Code example

    Parameters
    ----------

    phenotype_data : PhenotypeData
        Pre-calculated data structure of the phenotype data
    proba_haplotypes : np.ndarray
        Initial estimate of the individual haplotype frequencies
    max_iter : int
        Maximum number of iterations
    tol : float
        Log-likelihood convergence tolerance
    logging_threshold : float
        Threshold for logging small results

    """
    logging.info('Start EM algorithm')
    if proba_haplotypes is None:  # equal frequencies as initial guess
        logging.info('Auto-generated uniform initial guess')
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
        logging.info(
            'Iteration {0} | log(L) = {1:.4e} | delta = {2:.4e}'.format(
                i, log_likelihood, delta)
        )
        if i > 0:
            delta = objective_values[-1] - objective_values[-2]
        if delta < tol:
            logging.info('Convergence up to tolerance level {0}'.format(tol))
            break
        i += 1
    res = {''.join(k): v for k, v in zip(phenotype_data.haplotypes,
                                         proba_haplotypes)}
    if i == max_iter:
        logging.info('EM algorithm terminated after maximum number of {0} '
                    'iterations.'.format(i))
    # Additional log message for results
    msg = '\n\nResults\n' +\
              '=======\n' +\
          ''.join(['{0} | {1:.6f}\n'.format(k, v)
                   for k, v in res.items() if v > logging_threshold])
    logging.debug(msg)
    return res
