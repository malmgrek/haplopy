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


@pytest.mark.parametrize("haplotype,haplotypes,expected", [
    # Nothing to replace, present in reference list
    (
        ("A", "B"),
        [("A", "B"), ("A", "b"), ("a", "B"), ("a", "b")],
        [("A", "B")]
    ),
    # Nothing to replace, not present in reference list
    (
        ("x", "y"),
        [("A", "B"), ("A", "b"), ("a", "B"), ("a", "b")],
        []
    ),
    # Nothing to replace, empty reference list
    (
        ("A", "B"),
        [],
        []
    ),
    (
        (".", "b"),
        [("A", "B"), ("A", "b"), ("a", "B"), ("a", "b")],
        [("A", "b"), ("a", "b")]
    ),
    # A complete wildcard finds all
    (

        (".", "."),
        [("A", "B"), ("A", "b"), ("a", "B"), ("a", "b")],
        [("A", "B"), ("A", "b"), ("a", "B"), ("a", "b")]
    )
])
def test_match(haplotype, haplotypes, expected):
    res = datautils.match(haplotype, haplotypes)
    assert res == expected
    return


@pytest.mark.parametrize("haplotype", [
    ("A", "*"),
    ("A", "ö"),
    ("9", ",")
])
def test_match_invalid(haplotype):
    with pytest.raises(ValueError):
        datautils.match(haplotype, [("a", "b")])
    return


@pytest.mark.parametrize("diplotype,expected", [
    ((("A",), ("B")), ("AB",)),
    ((("A", "B"), ("A", "B")), ("AA", "BB")),
    ((("A", "B", "C"), ("a", "b", "c")), ("Aa", "Bb", "Cc")),
    ((("a", "b", "c"), ("A", "B", "C")), ("Aa", "Bb", "Cc")),
])
def test_unphase(diplotype, expected):
    res = datautils.unphase(diplotype)
    assert res == expected
    return


@pytest.mark.parametrize("diplotype,haplotypes,expected", [
    (
        (("A", "."), ("a", "b")),
        [("A", "B"), ("A", "b"), ("a", "B"), ("a", "b")],
        [(("A", "B"), ("a", "b")), (("A", "b"), ("a", "b"))]
    ),
    # A complete wildcard finds all
    (
        ((".", "."), (".", ".")),
        [("A", "B"), ("A", "b"), ("a", "B"), ("a", "b")],
        [
            (('A', 'B'), ('A', 'B')),
            (('A', 'B'), ('A', 'b')),
            (('A', 'B'), ('a', 'B')),
            (('A', 'B'), ('a', 'b')),
            (('A', 'b'), ('A', 'b')),
            (('A', 'b'), ('a', 'B')),
            (('A', 'b'), ('a', 'b')),
            (('a', 'B'), ('a', 'B')),
            (('a', 'B'), ('a', 'b')),
            (('a', 'b'), ('a', 'b')),
        ]
    ),
    # Unmatched come as they are
    (
        (("x", "y"), (".", "z")),
        [("A", "B"), ("A", "b"), ("a", "B"), ("a", "b")],
        [(("x", "y"), (".", "z"))]
    ),
    (
        (("a", "b"), ("A", "B")),
        [("x", "y")],
        [(("a", "b"), ("A", "B"))]
    ),
    (
        (("a", "b"), ("A", "B")),
        [],
        [(("a", "b"), ("A", "B"))]
    )
])
def test_fill(diplotype, haplotypes, expected):
    res = datautils.fill(diplotype, haplotypes)
    assert res == expected
    return


@pytest.mark.parametrize("counter,haplotypes,expected", [
    (
        Counter({("Aa", "Bb"): 2, ("AA", "BB"): 1}),
        [("A", "B"), ("A", "b"), ("a", "B"), ("a", "b")],
        [[(0, 3), (1, 2)], [(0, 0)]]
    )
    # Empty haplotypes
])
def test_build_diplotype_representation(counter, haplotypes, expected):
    res = datautils.build_diplotype_representation(counter, haplotypes)
    assert res == expected
    return


@pytest.mark.parametrize("phenotypes,expected", [
    (
        [],
        Counter()
    ),
    # Genotype permutation invariance
    (
        [("Aa", "Bb"), ("aA", "bB")],
        Counter({("Aa", "Bb"): 2})
    ),
    (
        [("AA", "BB", "CC"), ("Aa", "Bb", "cC")],
        Counter({("AA", "BB", "CC"): 1, ("Aa", "Bb", "Cc"): 1})
    )
])
def test_count_distinct(phenotypes, expected):
    assert datautils.count_distinct(phenotypes) == expected
    return
