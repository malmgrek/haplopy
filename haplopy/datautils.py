"""Data utils for HaploPy

Terminology
-----------

haplotype : A sequence of nucleotides.
            Example: ("a", "b")

diplotype : A pair of haplotypes.
            Example: (("a", "b"), ("A", "B"))

phenotype : A sequence of nucleotide pairs with unspecified diplotype.
            Example: ("Aa", "Bb")

genotype  : An item in the phenotype.
            Example: "Aa"

"""

from collections import Counter
from functools import reduce
import itertools
from operator import itemgetter
import re
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.sparse import dok_matrix


#
# Generic
#


def compose2(f, g):
    return lambda *args, **kwargs: f(g(*args, **kwargs))


def dmax(x: dict):
    """Dictionary's (argmax, max) key-value pair

    Note that this is a pure function. From Python 3.7 onwards, a dictionary is
    ordered by construction. Thus, in case of multiple keys of same numeric
    value, the result only depends on inputs initial order.

    Examples
    --------
    >>> dmax({"a": 1, "b": np.NaN, "c": 4})
    ("c", 4)

    """
    return max(
        filter(
            compose2(np.isfinite, itemgetter(1)),
            x.items()
        ),
        key=itemgetter(1),
        default=(None, None)
    )


#
# Haplotypes
#


def match(haplotype, haplotypes) -> List[Tuple[str]]:
    """Find haplotypes that match a given haplotype

    Examples
    --------
    >>> match(("a", "."), [("a", "b"), ("a", "B"), ("A", "B")])
    [("a", "b"), ("a", "B")]

    """


    def validate(s: str):
        valid = bool(re.compile("[a-zA-Z0-9\.]{2}").match(s))
        if not valid:
            raise ValueError(
                (
                    "Illegal characters in {} for pattern matching."
                    "Only a-z, A-Z and . are allowed."
                ).format(tuple(s))
            )
        return s


    return list(map(
        # Transform joint 2-strings back to tuples
        tuple,
        # Filter out 2-strings that don't match with the pattern
        filter(
            # Transform test haplotype into a regex pattern
            re.compile(validate("".join(haplotype))).match,
            # Join reference haplotypes into a list of 2-strings
            map(lambda h: "".join(h), haplotypes)
        )
    ))


#
# Diplotypes
#


def unphase(diplotype) -> Tuple[str]:
    """The most basic operation diplotype to phenotype mapping

    Examples
    --------
    >>> unphase((("A", "T", "G"), ("A", "A", "C")))
    ("AA", "AT", "GC")

    """
    return tuple("".join(sorted(snp)) for snp in zip(*diplotype))


def fill(diplotype, haplotypes) -> List[Tuple[Tuple[str]]]:
    """Attempt filling in missing values inside a diplotype

    Uses regular expressions to fill in missing SNPs with admissible values
    that are present in `haplotypes`.

    Leaves unmatchable patterns as they are.

    Examples
    --------
    >>> fill(
    ...     (("A", "."), ("a", "b")),
    ...     [("A", "B"), ("A", "b"), ("a", "B"), ("a", "b")]
    ... )
    [(("A", "B"), ("a", "b")), (("A", "b"), ("a", "b"))]

    """

    def expand(h):
        e = match(h, haplotypes)
        return (e if e else [h])

    def add_new(ds, d):
        return ds + [d] if not (d in ds or d[::-1] in ds) else ds

    return reduce(
        # Select unique (up to order switch) pairs
        add_new,
        # Cartesian product of all matching diplotypes
        itertools.product(*map(expand, diplotype)),
        []
    )


#
# Phenotypes
#


def count_distinct(phenotypes):
    """Count distinct phenotypes in a list

    Importantly, standardizes a dataset regarding differently ordered genotype
    strings within a phenotype.

    Examples
    --------
    >>> count_distinct([("aA", "bB"), ("Aa", "Bb")])
    Counter({('Aa', 'Bb'): 2})

    """

    # At the moment we have decided not to implement a tailored data type
    # for phenotypes, we need to resort to hacky sorting and string joins
    # to avoid misclassifying equivalent phenotypes with permuted genotypes.
    def sort_genotypes(phenotype):
        return tuple("".join(sorted(g)) for g in phenotype)

    return Counter(map(sort_genotypes, phenotypes))


def factorize(phenotype) -> List[Tuple[str]]:
    """List admissible diplotypes

    Examples
    --------
    >>> factorize(("Aa", "BB", "Cc"))
    [(('A', 'B', 'C'), ('a', 'B', 'c')), (('A', 'B', 'c'), ('a', 'B', 'C'))]

    """
    factors = list(itertools.product(*[
        # NOTE: Without sorting the function wouldn't be pure: set of string
        # doesn't have fixed order.
        sorted(set(locus)) for locus in phenotype
    ]))
    half = len(factors) // 2
    return (
        [(factors[0], factors[0])] if half == 0 else
        list(zip(factors[:half], factors[half:][::-1]))
    )


def factorize_fill(phenotype, haplotypes) -> List[Tuple[Tuple[str]]]:
    """Factorize to diplotypes with missing value filling

    Examples
    --------
    >>> factorize_fill(
    ...     ("Aa", "B."),
    ...     [("a", "b"), ("A", "b"), ("a", "B"), ("A", "B")]
    ... )
    [
        (('A', 'B'), ('a', 'B')),
        (('A', 'B'), ('a', 'b')),
        (('A', 'b'), ('a', 'B'))
    ]

    """
    # NOTE: Without sorting the function wouldn't be pure: set of string
    # doesn't have fixed order.
    return sorted(reduce(
        lambda ds, d: ds.union(
            map(
                lambda x: tuple(sorted(x)),
                fill(d, haplotypes)
            )
        ),
        factorize(phenotype),
        set()
    ))


def find_admissible_haplotypes(counter) -> List[Tuple[str]]:
    """List parent haplotypes

    Examples
    --------
    >>> from collections import Counter
    >>> find_admissible_diplotypes(Counter({
    ...     ("AA", "BB"): 2,
    ...     ("aa", "bb"): 1
    ... }))
    [("A", "B"), ("a", "b")]

    """
    # Sort the reduced set to fix output ordering.
    # In this way, we get a pure function.
    return sorted(
        reduce(
            lambda acc, x: acc.union(
                # Set of admissible haplotypes for single phenotype
                set(itertools.product(*map(set, x)))
            ),
            counter,
            set()
        )
    )


def build_diplotype_representation(counter, haplotypes) -> List[List[Tuple[int]]]:
    """Phenotype multiplicity and parent diplotype expansion

    Parameters
    ----------
    counter : collections.Counter
        Counts of distinct phenotypes, that is, "intra-genotype" string ordering
        doesn't matter.

    Returns
    -------
    representation : List[List[Tuple[int]]]
        Each item corresponds to the element in `counter` with same index.
        The item is a list of index pairs. Each index points to an element
        in `parent_haplotypes`, and the pair stands for an admissible parent
        haplotype couple.

    Examples
    --------
    >>> from collections import Counter
    >>> counter = Counter({("Aa", "Bb"): 2, ("aa", "bb"): 1})
    >>> haplotypes = [('a', 'b'), ('A', 'B'), ('a', 'B'), ('A', 'b')]
    >>> build_diplotype_representation(counter, haplotypes)
    [[(1, 0), (3, 2)], [(0, 0)]]

    """

    def factorize_to_index(phenotype):
        return [
            (haplotypes.index(x), haplotypes.index(y))
            for (x, y) in factorize(phenotype)
        ]

    return list(map(factorize_to_index, counter))


def build_diplotype_matrix(
        diplotype_representation: List[List[Tuple[int]]],
        haplotypes: List[Tuple[str]]
):
    """Haplotype multiplicity in a 'N haplotypes' * 'M diplotypes' matrix

    Indicates out how many times haplotype n is present in diplotype m

    Examples
    --------
    >>> from collections import Counter
    >>> counter = Counter({("Aa", "Bb"): 2, ("aa", "bb"): 1})
    >>> haplotypes = [('a', 'b'), ('A', 'B'), ('a', 'B'), ('A', 'b')]
    >>> drepr = build_diplotype_representation(counter, haplotypes)
    >>> build_diplotype_matrix(drepr, haplotypes).toarray()
    array([[1, 0, 2],
           [1, 0, 0],
           [0, 1, 0],
           [0, 1, 0]])


    """

    diplotypes = reduce(lambda x, y: x + y, diplotype_representation, [])
    matrix = dok_matrix(
        (len(haplotypes), len(diplotypes)),
        dtype=int
    )

    # Populate matrix
    for (i, diplotype) in enumerate(diplotypes):
        matrix[diplotype[0], i] += 1
        matrix[diplotype[1], i] += 1

    return matrix
