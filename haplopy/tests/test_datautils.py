"""Unit tests for Datautils module

"""

from collections import Counter

import numpy as np
from numpy.testing import assert_equal
import pytest

from haplopy import datautils


@pytest.mark.parametrize("phenotypes,expected", [
    (
        [],
        []
    ),
    (
        [()],
        [()]
    ),
    (
        [("AA",)],
        [("A",)]
    ),
    (
        [("AA", "BB")],
        [("A", "B")]
    ),
    (
        [("Aa", "Bb")],
        [("a", "b"), ("a", "B"), ("A", "b"), ("A", "B")]
    ),
    (
        [("aA", "Bb", "Cc")],
        [
            ("a", "b", "C"), ("a", "b", "c"), ("a", "B", "C"), ("a", "B", "c"),
            ("A", "b", "C"), ("A", "b", "c"), ("A", "B", "c"), ("A", "B", "C")
        ]
    ),
    # Before parent haplotype ordering fixes (0.1.0), this example reproduces
    # an case with varying output ordering.
    (
        [
            ('aa', 'bb', 'cc'),
            ('Aa', 'Bb', 'Cc'),
            ('aa', 'bb', 'cc'),
            ('aa', 'bb', 'cc'),
            ('Aa', 'Bb', 'Cc'),
            ('aa', 'bb', 'cc'),
            ('Aa', 'Bb', 'Cc'),
            ('Aa', 'Bb', 'Cc'),
            ('Aa', 'Bb', 'Cc'),
            ('aa', 'bb', 'cc')
        ],
        [
            ('A', 'B', 'C'),
            ('A', 'B', 'c'),
            ('A', 'b', 'C'),
            ('A', 'b', 'c'),
            ('a', 'B', 'C'),
            ('a', 'B', 'c'),
            ('a', 'b', 'C'),
            ('a', 'b', 'c')
        ]
    )
])
def test_find_admissible_haplotypes(phenotypes, expected):
    res = datautils.find_admissible_haplotypes(phenotypes)
    assert set(res) == set(expected)  # Ordering doesn't matter
    return


@pytest.mark.parametrize("phenotype,expected", [
    (
        (),
        [((), ())]
    ),
    # One homozygous haplotype
    (
        ("AA",),
        [(("A",), ("A",))]
    ),
    (
        ("AA", "BB"),
        [(("A", "B"), ("A", "B"))]
    ),
    (
        ("Aa", "BB", "Cc"),
        [
            (("A", "B", "C"), ("a", "B", "c")),
            (("A", "B", "c"), ("a", "B", "C"))
        ]
    )
])
def test_factorize(phenotype, expected):
    res = datautils.factorize(phenotype)
    print(res)
    assert res == expected
    return


@pytest.mark.parametrize("diplotype_representation,haplotypes,expected", [
    (
        [],
        ["foobar", 1, 2],
        np.ndarray(shape=(3, 0), dtype=int)
    ),
    (
        [[(0, 0)]],
        [("A", "B")],
        np.array([[2]])
    ),
    (
        [[(0, 1), (3, 2)], [(2, 2)]],
        [("A", "b"), ("a", "B"), ("A", "B"), ("a", "b")],
        np.array([
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 2],
            [0, 1, 0]
        ])
    )
])
def test_build_diplotype_matrix(diplotype_representation, haplotypes, expected):
    res = datautils.build_diplotype_matrix(diplotype_representation, haplotypes)
    assert_equal(res.toarray(), expected)
    return


def test_match():
    raise NotImplementedError


def test_unphase():
    raise NotImplementedError


def test_fill():
    raise NotImplementedError


def test_build_diplotype_representation():
    raise NotImplementedError


def test_count_distinct():
    raise NotImplementedError
