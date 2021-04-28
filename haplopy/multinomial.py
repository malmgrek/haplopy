"""Haplotype frequency inference model

TODO: Remove excess type hints
TODO: Flag for NaN-mode or doctor-mode in proba_diplotypes
TODO: Random generate EM initial values
TODO: Log-likelihood method
TODO: Save and load method for model

Phenotype imputation thoughts
-----------------------------

Assume a phenotype x has some missing loci. Mark missing value with "."

1. Factorize x
2. For each haplotype pair in the factorization,
   use string matching to "expand" the pair to all feasible pairs.
3. Filter to unique list of pairs.
   - Remember that pairs in different order are still same.
4. Map diplotypes to index pairs
   - Use NaN-extended `proba_haplotypes` so that presence of unseen haplotypes
     will result in NaN probability for all of the diplotypes. NOTE: there may
     also be dots as diplotype filling may have failed

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
        haplotypes = datautils.find_admissible_haplotypes(phenotypes)
        Nh = len(haplotypes)
        probas = np.ones(Nh) / Nh
    else:
        (haplotypes, probas) = zip(*proba_haplotypes.items())

    counter = datautils.count_distinct(phenotypes)
    diplotype_representation = datautils.build_diplotype_representation(
        counter, haplotypes
    )

    # Diplotype count matrix for faster multiplication
    diplotype_matrix = datautils.build_diplotype_matrix(
        diplotype_representation, haplotypes
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

        def calculate(inds):
            return np.array([
                2 ** (i != j) * probas[i] * probas[j] for (i, j) in inds
            ])

        Ps_raws = [calculate(inds) for inds in diplotype_representation]
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

    TODO: probas and haplotypes as separate attributes

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
                "".join(sorted(snp)) for snp in zip(haplotypes[i], haplotypes[j])
            )
            for (i, j) in parent_inds
        ]

    @classmethod
    def fit(cls, phenotypes: Dict[Tuple[str]], **kwargs) -> Model:
        """Fit maximum likelihood haplotype probabilities using EM algorithm

        Implemented as a classmethod so that we can use given phenotypes
        to initialize the model conveniently.

        """
        (proba_haplotypes, _) = expectation_maximization(phenotypes, **kwargs)
        return cls(proba_haplotypes=proba_haplotypes)

    def calculate_proba_diplotypes(
            self,
            phenotype: Tuple[str],
            fill_proba=np.NaN,
    ) -> Dict[Tuple[int], float]:
        """Calculate admissible diplotypes' conditional probabilities

        """

        # Extended diplotypes where missing data is filled if possible
        diplotypes = reduce(
            lambda ds, d: ds + datautils.fill(d, self.haplotypes),
            datautils.factorize(phenotype),
            []
        )

        # Model's haplotypes might not contain all of the constituent haplotypes
        # of the given set of phenotypes. The probability of such haplotypes
        # will be considered `fill_proba`. Note that the implied ambiguity in
        # unit summability of the NaN-probability distribution doesn't ruin the
        # calculation of diplotype probabilities (with the existing haplotypes)
        # because each conditional probability is normalized.
        proba_haplotypes = {
            h: self.proba_haplotypes.get(h, fill_proba)
            for diplotype in diplotypes for h in diplotype
        }

        def calculate(ds):
            return np.array([
                2 ** (d1 == d2) * proba_haplotypes[d1] * proba_haplotypes[d2]
                for (d1, d2) in ds
            ])

        def normalize(x):
            return x / x.sum() if any(x) else x

        def to_dict(ds):
            return dict(zip(ds, normalize(calculate(ds))))

        return to_dict(diplotypes)

    def impute(self, phenotype: Tuple[str], **kwargs):
        """Impute by selecting the most probable diplotype

        """
        # TODO: Impute with the most probable values defined by
        #       `calculate_proba_diplotypes`.
        #
        #       - If there are no missing values, returns original phenotype.
        #       - If cannot be imputed, log warning and return original.
        #
        proba_diplotypes = self.calculate_proba_diplotypes(phenotype, **kwargs)
        most_probable = max(proba_diplotypes, key=proba_diplotypes.get)
        least_probable = min(proba_diplotypes, key=proba_diplotypes.get)
        return (
            datautils.unphase(diplotype) if most_probable > least_probable
            else phenotype
        )
