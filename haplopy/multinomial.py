"""Haplotype frequency inference model

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
        # randomize: bool=False,
        max_iter: int=100,
        tol: float=1.0e-12
):
    """Expectation Maximization search for haplotype probabilities

    Philosophically, builds upon the hypotheses of Hardy-Weinberg equilibrium
    and random mating.

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations
    tol : float
        Stopping criterion with respect to Euclidean norm of
        probabilities vector

    TODO: Random generate initial values and run many optimization threads
    TODO: Make sure that haplotype order (dict) is not lost

    """


    N = len(phenotypes)

    (
        parent_haplotypes,
        counter,
        genotype_expansion,
    ) = datautils.describe_phenotypes(phenotypes)

    # Genotype count matrix for faster multiplication
    genotype_matrix = datautils.build_genotype_matrix(
        genotype_expansion, parent_haplotypes
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

    def expectation(ps: np.ndarray):
        """Calculate probability for each pair and evaluate log-likelihood

        """

        def calculate(gs):
            return np.array([2 ** (i != j) * ps[i] * ps[j] for (i, j) in gs])

        Ps_raws = [calculate(gs) for gs in genotype_expansion]
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
        return 0.5 * genotype_matrix.dot(Ps)

    Nh = len(parent_haplotypes)
    ps = np.ones(Nh) / Nh
    n_iter = 0
    step = np.inf
    logging.info("Start EM algorithm")
    while (n_iter <= max_iter) and (step > tol):
        (Ps, log_likelihood) = expectation(ps)
        delta = maximization(Ps) - ps
        step = np.linalg.norm(delta)
        ps = ps + delta
        logging.info(
            "Iteration {0} | log(L) = {1:.4e} | step = {2: .4e}".format(
                n_iter, log_likelihood, step
            )
        )

    return (
        dict(zip(parent_haplotypes, ps)),
        log_likelihood
    )


class Model():

    def __init__(self, p_haplotypes: Dict[Tuple[str], float]=None):
        if p_haplotypes is None:
            return
        (haplotypes, ps) = zip(*p_haplotypes.items())
        assert abs(sum(ps) - 1) < 1e-8, "Probabilities must sum to one"
        self.p_haplotypes = p_haplotypes
        return

    def random(self, n_obs: int) -> List[Tuple[str]]:
        """Random generate phenotypes

        """
        if self.p_haplotypes is None:
            raise ValueError("Model probabilities unspecified, cannot randomize.")
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

    def fit(self, phenotypes: Dict[Tuple[str], float], **kwargs) -> Model:
        """Fit maximum likelihood haplotype probabilities using EM algorithm

        """
        (p_haplotypes, _) = expectation_maximization(phenotypes, **kwargs)
        return Model(p_haplotypes)
