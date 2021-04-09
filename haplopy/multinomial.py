"""Haplotype imputation model

"""

import logging
from typing import Dict, List, Tuple

import numpy as np

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
# TODO Formulate data as a binary string -> phenotyes as locus sums
#      - This would be more restrictive without clear benefits
# TODO Imputation of missing locus observations e.g. ("A*", "TT", "GC")


def log_binomial(n: int, k: int):
    """Logarithm of binomial coefficient using Stirling approximation

    Binomial coefficient easily gets very large. Thus let's approximate
    directly it's logarithm.

    This is only used as the calibration term in log-likelihood, so accuracy
    isn't super critical.

    Stirling formula:

                       n
            _______ ⎛n⎞
      n = ╲╱ 2 π n  ⎜─⎟
                    ⎝e⎠

    TODO: Unit test

    """
    d = n - k
    return (
        n * np.log(n) - k * np.log(k) - d * np.log(d)
        + 0.5 * (np.log(n) - np.log(k) - np.log(d) - np.log(2 * np.pi))
    )

def log_multinomial(*args):
    """Logarithm of the multinomial coefficient

    Based on the formula

    (n1 + n2 + ... + nk)!   ⎛n1⎞ ⎛n1 + n2⎞     ⎛n1 + n2 + ... + nk⎞
    --------------------- = ⎜  ⎟ ⎜       ⎟ ... ⎜                  ⎟
     n1!  n2!  ...   nk!    ⎝n1⎠ ⎝   n1  ⎠     ⎝        nk        ⎠

    TODO: Unit test

    """
    return (
        1 if len(args) == 1 else
        log_binomial(sum(args), args[-1]) + log_multinomial(args[:-1])
    )


def expectation_maximization(
        phenotypes: List[Tuple[str]],
        randomize: bool=False,
        max_iter: int=100,
        tol: float=1.0e-12,
        logging_threshold: float=1.0e-6
):
    """Expectation Maximization search for haplotype probabilities

    Parameters
    ----------

    References
    ----------
    [1] Polanska article for practical implementation
    [2] Uni. Helsinki course material for proof

    """

    # TODO: Random generate multiple initial values and run many threads
    # TODO: Make sure that haplotype order (dict) is not lost

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
    # Non-rigorous separation into expectation and maximization
    #

    def expectation(ps: np.ndarray):
        # Calculate (normalized) probability for each genotype (haplotype
        # pair) and log likelihood

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
        # Calculates the next estimate of haplotype probabilities and previous
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

    # TODO / FIXME: Combine tuple of strings to string
    #               Requires modifications throughout the code
    return dict(zip(parent_haplotypes, ps))


class Model():

    def __init__(self, p_haplotypes: Dict[str, float]=None):
        (haplotypes, ps) = zip(*p_haplotypes.items())
        assert abs(sum(ps) - 1) < 1e-8, "Probabilities must sum to one"
        self.p_haplotypes = p_haplotypes
        return

    def fit(self, phenotypes: Dict[str, float]):
        """Fit haplotype probabilities using EM algorithm

        """
        # TODO Call expectation_maximization
        p_haplotypes = None
        return Model(p_haplotypes)

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
