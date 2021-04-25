"""Haplotype frequency inference model

TODO: Batch learning + tests
      - Should normalize probability in loop or at end?
      - Case: parent_haplotypes and given haplotypes partly separate sets
      - Weighting of incoming contributions
TODO: Random generate EM initial values
TODO: Log-likelihood method
TODO: Save and load method for model

"""


from __future__ import annotations

from functools import reduce
import logging
from typing import Dict, List, Tuple

import numpy as np

from haplopy import datautils


def log_multinomial(*ns: int):
    """Logarithm of the multinomial coefficient

        (n1 + n2 + ... + nk)!
        ---------------------
        n1!  n2!  ...   nk!

    log(n!) = log(n) + log(n-1) + log(n-2) + ... + log(1)

    """

    def log_factorial(n):
        return np.log(np.arange(1, n + 1)).sum()

    return log_factorial(sum(ns)) - sum(map(log_factorial, ns))


def expectation_maximization(
        phenotypes: List[Tuple[str]],
        proba_haplotypes: Dict[Tuple[str], float]=None,
        max_iter: int=100,
        tol: float=1.0e-12
):
    """Expectation Maximization algorithm for haplotype probabilities

    Tries to iteratively find the Maximum Likelihood estimate.

    Philosophically, builds upon the hypotheses of Hardy-Weinberg equilibrium
    and random mating.

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations
    tol : float
        Stopping criterion with respect to Euclidean norm of
        probabilities vector

    """

    N = len(phenotypes)

    # If haplotype probabilities are not given, a uniform initial probability
    # is assumed, and only admissible haplotypes are considered
    if proba_haplotypes is None:
        haplotypes = datautils.find_parent_haplotypes(phenotypes)
        Nh = len(haplotypes)
        probas = np.ones(Nh) / Nh
    else:
        (haplotypes, probas) = zip(*proba_haplotypes.items())

    (counter, diplotype_expansion) = datautils.build_diplotype_expansion(
        haplotypes, phenotypes
    )

    # Diplotype count matrix for faster multiplication
    diplotype_matrix = datautils.build_diplotype_matrix(
        haplotypes, diplotype_expansion
    )

    #
    # Separation into expectation and maximization
    #
    # This is not a mathematically accurate separation
    # but loosely based on [1]. The update formulae are non-trivial to
    # derive. A clear mathematical proof is found, e.g., in [2].
    #
    # [1] J. Polanska, The EM algorithm and its implementation for the
    #     estimation of frequencies of SNP-haplotypes, Int. J. Appl. Math.
    #     Comput. Sci., 2003, Vol. 13, No. 3, 419-429.
    #
    # [2] Mikko Koivisto, 582673 Computational Genotype Analysis (lecture 2
    #     notes), Uni. Helsinki
    #

    def expectation(probas: np.ndarray):
        """Calculate probability for each pair and evaluate log-likelihood

        """

        def calculate(ds):
            return np.array([
                2 ** (i != j) * probas[i] * probas[j] for (i, j) in ds
            ])

        Ps_raws = [calculate(ds) for ds in diplotype_expansion]
        Ps_units = np.hstack([
            Ps * n / Ps.sum() / N for (Ps, n) in zip(Ps_raws, counter.values())
        ])
        log_likelihood = sum([
            n * np.log(Ps.sum()) for (Ps, n) in zip(Ps_raws, counter.values())
        ]) + log_multinomial(*counter.values())

        return (Ps_units, log_likelihood)

    def maximization(Ps):
        """Calculate next estimates of haplotype probabilities

        """
        return 0.5 * diplotype_matrix.dot(Ps)

    n_iter = 0
    step = np.inf
    logging.info("Start EM algorithm")
    while (n_iter <= max_iter) and (step > tol):
        (Ps, log_likelihood) = expectation(probas)
        delta = maximization(Ps) - probas
        step = np.linalg.norm(delta)
        probas = probas + delta
        logging.info(
            "Iteration {0} | log(L) = {1:.4e} | step = {2: .4e}".format(
                n_iter, log_likelihood, step
            )
        )
        n_iter += 1

    return (
        dict(zip(haplotypes, probas)),
        log_likelihood
    )


class Model():
    """Haplotype multinomial distribution estimator

    Example
    -------

    .. code-block:: python

        import haplopy as hp

        model = hp.multinomial.Model({("a", "b"): 0.5, ("A", "B"): 0.5})
        phenotypes = model.random(10)
        model_est = hp.multinomial.Model.fit(phenotypes)

    """

    def __init__(self, proba_haplotypes: Dict[Tuple[str], float]):
        (haplotypes, probas) = zip(*proba_haplotypes.items())
        assert abs(sum(probas) - 1) < 1e-8, "Probabilities must sum to one"
        self.proba_haplotypes = proba_haplotypes
        self.haplotypes = haplotypes
        return

    def random(self, n_obs: int) -> List[Tuple[str]]:
        """Random generate phenotypes

        """
        (haplotypes, probas) = zip(*self.proba_haplotypes.items())
        parent_inds = np.dot(
            np.random.multinomial(1, probas, 2 * n_obs),
            np.arange(len(probas))
        ).reshape(n_obs, 2)
        return [
            tuple(
                "".join(sorted(diplo))
                for diplo in zip(haplotypes[i], haplotypes[j])
            )
            for (i, j) in parent_inds
        ]

    @classmethod
    def fit(cls, phenotypes: Dict[Tuple[str], float], **kwargs) -> Model:
        """Fit maximum likelihood haplotype probabilities using EM algorithm

        Implemented as a classmethod so that we can use given phenotypes
        to initialize the model conveniently.

        """
        (proba_haplotypes, _) = expectation_maximization(phenotypes, **kwargs)
        return cls(proba_haplotypes=proba_haplotypes)

    def calculate_proba_diplotypes(
            self,
            phenotypes: Dict[Tuple[str], float]
    ) -> List[Dict[Tuple[str], float]]:
        """Calculate admissible diplotypes' conditional probabilities

        """

        # We need parent haplotypes to handle cases where admissible diplotypes
        # contain haplotypes missing from model
        parent_haplotypes = datautils.find_parent_haplotypes(phenotypes)
        (counter, diplotype_expansion) = datautils.build_diplotype_expansion(
            parent_haplotypes, phenotypes
        )

        # Model's haplotypes might not contain all of the constituent haplotypes
        # of the given set of phenotypes. The probability of such haplotypes
        # will be considered NaN. Note that the implied ambiguity in unit
        # summability of the NaN-probability distribution doesn't ruin the
        # calculation of diplotype probabilities (with the existing haplotypes)
        # because each conditional probability is normalized.
        proba_haplotypes = {
            h: self.proba_haplotypes.get(h, np.NaN) for h in parent_haplotypes
        }
        (haplotypes, probas) = zip(*proba_haplotypes.items())

        def calculate(ds):
            return np.array([
                2 ** (i != j) * probas[i] * probas[j] for (i, j) in ds
            ])

        def normalize(x):
            return x / x.sum()

        def to_dict(ds):
            keys = [(haplotypes[i], haplotypes[j]) for (i, j) in ds]
            values = normalize(calculate(ds))
            return dict(zip(keys, values))

        # Calculate a dict of diplotype probabilities for each unique
        # phenotype
        lookup_probas = list(map(to_dict, diplotype_expansion))
        unique_phenotypes = list(counter)

        # Retrieve a probability dict for each of the phenotypes
        return [
            lookup_probas[unique_phenotypes.index(phenotype)]
            for phenotype in phenotypes
        ]
