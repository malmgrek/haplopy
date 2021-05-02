"""Multinomial distribution haplotype frequency inference model

TODO: Random generate EM initial values

"""


from __future__ import annotations

from functools import reduce
import json
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

    phenotypes : List[Tuple[str]]
        Observation data; list of phenotypes
    proba_haplotypes : Dict[Tuple[str], float]
        Initial haplotype probabilities
    max_iter : int
        Maximum number of iterations
    tol : float
        Stopping criterion with respect to Euclidean norm of probabilities vector

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

    proba_haplotypes = dict(zip(haplotypes, probas))

    return (proba_haplotypes, log_likelihood)


class Model():
    """Haplotype multinomial distribution estimator

    Examples
    --------
    >>> model = Model({("a", "b"): 0.5, ("A", "B"): 0.5})
    >>> phenotypes = model.random(10)
    >>> model_fitted = Model.fit(phenotypes)
    >>> model_fitted.proba_haplotypes
    {('A', 'B'): 0.6,
     ('A', 'b'): 0.0,
     ('a', 'B'): 0.0,
     ('a', 'b'): 0.4}


    """

    def __init__(self, proba_haplotypes: Dict[Tuple[str], float]):
        (haplotypes, probas) = zip(*proba_haplotypes.items())
        assert abs(sum(probas) - 1) < 1e-8, "Probabilities must sum to one"
        self.proba_haplotypes = proba_haplotypes
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
    def fit(cls, phenotypes: Tuple[str], **kwargs) -> Model:
        """Fit maximum likelihood haplotype probabilities using EM algorithm

        Implemented as a classmethod so that we can use given phenotypes
        to initialize the model conveniently.

        """
        (proba_haplotypes, _) = expectation_maximization(phenotypes, **kwargs)
        return cls(proba_haplotypes=proba_haplotypes)

    def calculate_proba_diplotypes(
            self,
            phenotype: Tuple[str],
            fill_proba: float=np.NaN,
    ) -> Dict[Tuple[int], float]:
        """Calculate admissible diplotypes' conditional probabilities

        Examples
        --------
        >>> model = Model({("a", "b"): 0.2, ("A", "B"): 0.5, ("a", "B"): 0.3})
        >>> model.calculate_proba_diplotypes(("Aa", "B."))
        {(('A', 'B'), ('a', 'B')): 0.6, (('A', 'B'), ('a', 'b')): 0.4}

        """

        # Extended diplotypes where missing data is filled if possible
        #
        # At the moment we have decided not to implement a tailored data type
        # for diplotypes, we need to use this complicated and messy composite
        # function to try to ensure there are no duplicate permutations in the
        # filled list. If we had a hashable and symmetric data type for
        # diplotypes, we could just use set (or frozenset) to trivially reduce
        # to unique diplotypes.
        diplotypes = datautils.factorize_fill(phenotype, self.proba_haplotypes)

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
                2 ** (h1 != h2) * proba_haplotypes[h1] * proba_haplotypes[h2]
                for (h1, h2) in ds
            ])

        def normalize(x):
            return x / x.sum() if any(x) else x

        def to_dict(ds):
            return dict(zip(ds, normalize(calculate(ds))))

        return to_dict(diplotypes)

    def calculate_proba_phenotypes(self, phenotype: Tuple[str], **kwargs):
        """Compute probabilities of different options that fill a given phenotype

        Examples
        --------
        >>> model = Model({("a", "b"): 0.2, ("A", "B"): 0.5, ("a", "B"): 0.3})
        >>> model.calculate_proba_phenotypes(("Aa", ".."))
        {("Aa", "BB"): 0.6, ("Aa", "Bb": 0.4)
        """
        proba_diplotypes = self.calculate_proba_diplotypes(phenotype, **kwargs)

        def agg(cum, xs):
            (diplotype, proba) = xs
            phenotype = datautils.unphase(diplotype)
            # There can be multiple different diplotypes that result into the
            # same phenotype
            return (
                {**cum, **{phenotype: proba}} if phenotype not in cum
                else {**cum, **{phenotype: proba + cum[phenotype]}}
            )

        return reduce(agg, proba_diplotypes.items(), dict())

    def impute(self, phenotype: Tuple[str], **kwargs):
        """Impute by selecting the most probable diplotype

        Examples
        --------
        >>> model = Model({("a", "b"): 0.2, ("A", "B"): 0.5, ("a", "B"): 0.3})
        >>> model.impute(("Aa", "BB"))

        """
        proba_phenotypes = self.calculate_proba_phenotypes(phenotype)
        (argmax, m) = datautils.dmax(proba_phenotypes)
        return argmax if argmax else phenotype

    def to_json(self, fp):
        """Save existing model to hard disk

        """

        def jsonify(x):
            return {"".join(k): v for (k, v) in x.items()}

        with open(fp, "w+") as f:
            json.dump(jsonify(self.proba_haplotypes), f)

        return

    @classmethod
    def from_json(cls, fp):
        """Instantiate a new model from a JSON on disk

        """

        def unjsonify(x):
            return {tuple(k): v for (k, v) in x.items()}

        with open(fp, "r") as f:
            raw = json.load(f)

        return cls(proba_haplotypes=unjsonify(raw))
