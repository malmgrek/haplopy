"""Tests for the Multinomial module"""

from functools import reduce

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_raises
import pytest

from haplopy.multinomial import log_multinomial, expectation_maximization, Model


np.random.seed(666)


def assert_dicts_almost_equal(x, y, **kwargs):
    (x_keys, x_values) = zip(*x.items())
    (y_keys, y_values) = zip(*y.items())
    assert x_keys == y_keys
    assert_almost_equal(x_values, y_values)


@pytest.mark.parametrize("args,expected", [
    ((1,), 0),
    # Note that this is not binomial(n, n)
    ((13, 13), np.log(10400600)),
    ((1, 2, 3, 4), np.log(12600)),
    # Large numbers
    ((666, 42, 313, 13), 849.87355176),
    # Invariant of argument permutations
    ((1, 2, 3, 4), log_multinomial(4, 2, 1, 3)),
    ((1, 2, 3, 4), log_multinomial(2, 1, 3, 4)),
    ((1, 2, 3, 4), log_multinomial(3, 4, 1, 2)),
    # Boundary values
    ((5, 0), 0),
    ((1, 2, 0, 4), log_multinomial(1, 2, 4))
])
def test_log_multinomial(args, expected):
    res = log_multinomial(*args)
    assert_almost_equal(res, expected)
    return


@pytest.mark.parametrize("kwargs,expected", [
    (
        {
            "phenotypes": [
                ("Aa", "Bb")
            ]
        },
        {
            ("A", "B"): 0.25,
            ("A", "b"): 0.25,
            ("a", "B"): 0.25,
            ("a", "b"): 0.25
        }
    ),
    (
        {
            "phenotypes": [
                ("aa", "BB"),
                ("AA", "BB")
            ]
        },
        {
            ("a", "B"): 0.5,
            ("A", "B"): 0.5
        }
    ),
    (
        {
            "phenotypes": [
                ("Aa", "BB", "Cc"),
                ("AA", "BB", "cc"),
                ("AA", "BB", "cc"),
                ("Aa", "BB", "Cc")
            ]
        },
        {
            ("a", "B", "c"): 0.0,
            ("A", "B", "C"): 0.0,
            ("A", "B", "c"): 0.75,
            ("a", "B", "C"): 0.25
        }
    )
])
def test_expectation_maximization(kwargs, expected):
    # TODO: Add a trivial case where there is just one possible parent haplotype
    (res, _) = expectation_maximization(**kwargs)
    assert res == expected
    return


def test_model_init():
    proba_haplotypes = {("a", "b"): 0.5, ("A", "B"): 0.6}
    assert_raises(AssertionError, Model, proba_haplotypes)


@pytest.mark.parametrize("proba_haplotypes,n_obs,expected", [
    (
        {
            ("A", "B", "C"): 0.24,
            ("a", "b", "c"): 0.66,
            ("A", "b", "C"): 0.1
        },
        10,
        {
            ("A", "B", "C"): 0.25,
            ("a", "b", "c"): 0.75
        }
    ),
    (
        {
            ("A", "B", "C"): 0.24,
            ("a", "b", "c"): 0.66,
            ("A", "b", "C"): 0.1
        },
        1000,
        {
            ("A", "B", "C"): 0.2370,
            ("A", "b", "C"): 0.1045,
            ("a", "b", "c"): 0.6585
        }
    )
])
def test_model(proba_haplotypes, n_obs, expected):
    model = Model(proba_haplotypes)
    phenotypes = model.random(n_obs)
    model_fitted = model.fit(phenotypes)
    # Assert that essentially nonzero probabilities coincide with expected
    result = {
        k: v for (k, v) in model_fitted.proba_haplotypes.items() if v >= 1e-8
    }
    assert_dicts_almost_equal(result, expected)
    return


@pytest.mark.parametrize("proba_haplotypes,phenotypes,expected", [
    # All haplotypes present
    (
        {
            ("A", "B"): 0.1,
            ("a", "B"): 0.2,
            ("A", "b"): 0.3,
            ("a", "b"): 0.4
        },
        [
            ('Aa', 'Bb'),
            ('aa', 'bb'),
            ('Aa', 'bb'),
            ('Aa', 'BB'),
            ('aa', 'Bb'),
        ],
        [
            {(('A', 'B'), ('a', 'b')): 0.4, (('A', 'b'), ('a', 'B')): 0.6},
            {(('a', 'b'), ('a', 'b')): 1.0},
            {(('A', 'b'), ('a', 'b')): 1.0},
            {(('A', 'B'), ('a', 'B')): 1.0},
            {(('a', 'B'), ('a', 'b')): 1.0},
        ]
    ),
    # Some haplotypes missing
    (
        {
            ("A", "B"): 0.5,
            ("a", "b"): 0.5
        },
        [
            ('aa', 'bb'),
            ('Aa', 'Bb'),
            ('AA', 'BB'),
            ('aa', 'bb'),
            ('AA', 'BB'),
        ],
        [
            {(('a', 'b'), ('a', 'b')): 1.0},
            {(('A', 'B'), ('a', 'b')): np.NaN, (('A', 'b'), ('a', 'B')): np.NaN},
            {(('A', 'B'), ('A', 'B')): 1.0},
            {(('a', 'b'), ('a', 'b')): 1.0},
            {(('A', 'B'), ('A', 'B')): 1.0},
        ]
    ),
    # TODO: Diplotypes zero probability
])
def test_proba_diplotypes(proba_haplotypes, phenotypes, expected):
    model = Model(proba_haplotypes)
    proba_diplotypes = model.calculate_proba_diplotypes(phenotypes[0])
    for (phenotype, e) in zip(phenotypes, expected):
        proba_diplotypes = model.calculate_proba_diplotypes(phenotype)
        assert_dicts_almost_equal(proba_diplotypes, e)
    return
