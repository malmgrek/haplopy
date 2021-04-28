"""Data utils for HaploPy

Terminology
-----------

haplotype : A sequence of nucleotides.
            For example ("a", "b").

diplotype : A pair of haplotypes.
            For example (("a", "b"), ("A", "B")).

phenotype : A sequence of nucleotide pairs with unspecified diplotype.
            For example ("Aa", "Bb").

genotype  : An item in the phenotype.
            For example "Aa".

"""

from collections import Counter
from functools import reduce
import itertools
import re
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.sparse import dok_matrix


class multiset():

    def __init__(self, xs):
        self.items = tuple(sorted(xs))

    def __eq__(self, other):
        return self.items == other.items

    def __hash__(self):
        return hash(self.items)

    def __repr__(self):
        return "multiset({{{0}}})".format(", ".join(map(str, self.items)))

    def __iter__(self):
        yield from self.items


#
# Haplotypes
#


def match(haplotype: Tuple[str], haplotypes: List[Tuple[str]]) -> List[Tuple[str]]:
    """Find haplotypes that match a given haplotype

    Example
    -------

    >>> find_matching_haplotypes(
    ...     ("a", "."),
    ...     [("a", "b"), ("a", "B"), ("A", "B")]
    ... )
    [("a", "b"), ("a", "B")]

    TODO: Test

    """
    return list(
        map(
            tuple,
            re.findall(
                "".join(haplotype),
                " ".join(map(lambda h: "".join(h), haplotypes))
            )
        )
    )


#
# Phenotypes
#


def Phenotype(*args):
    return tuple(map(frozenset, args))


def find_admissible_haplotypes(phenotype: Phenotype):
    return frozenset(itertools.product(*phenotype))


def factorize_to_diplotypes(phenotype: Phenotype):
    factors = list(itertools.product(*phenotype))
    half = len(factors) // 2
    return frozenset(
        [Diplotype(factors[0], factors[0])] if half == 0 else
        [Diplotype(x, y) for (x, y) in zip(factors[:half], factors[half:][::-1])]
    )


# TODO: Reduced version of find_admissible_haplotypes


def find_parent_haplotypes(phenotypes):
    """List parent haplotypes

    """
    unique_phenotypes = set(phenotypes)
    return sorted(  # Sort to fix output ordering
        reduce(
            lambda parents, phenotype: parents.union(
                set(itertools.product(*[
                    set(locus) for locus in phenotype
                ]))
            ),
            unique_phenotypes,
            set()
        )
    )


def factorize(phenotype: Tuple[str]) -> List[Tuple[str]]:
    """List admissible diplotypes

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


#
# Diplotypes
#


def Diplotype(x, y):
    return multiset([x, y])


def unphase(diplotype):
    """The most basic operation diplotype to phenotype mapping

    Example
    -------

    >>> unphase((("A", "T", "G"), ("A", "A", "C")))
    ("AA", "AT", "GC")

    TODO: Test

    """
    return Phenotype(*zip(*diplotype))


def fill(diplotype: Diplotype, haplotypes: List[Tuple[str]]) -> List[Diplotype]:
    """Attempt filling in missing values inside a diplotype

    Uses regular expressions to fill in missing SNPs with admissible values
    that are present in `haplotypes`.

    Leaves unmatchable patterns as they are.

    Example
    -------

    >>> fill(
    ...     Diplotype(("A", "."), ("a", "b")),
    ...     [("A", "B"), ("A", "b"), ("a", "B"), ("a", "b")]
    ... )
    [Diplotype(("A", "B"), ("a", "b")), Diplotype(("A", "b"), ("a", "b"))]

    TODO: Test

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
# Model-fitting helpers
#


def build_diplotype_expansion(
        haplotypes: List[Tuple[str]],
        phenotypes: List[Phenotype]  # TODO: Replace with phenotype_counter
) -> Tuple[Counter, List[List[multiset]]]:
    """Phenotype multiplicity and parent diplotype expansion

    Helps building data structures in multinomial model EM fitting

    Returns
    -------

    counter : Counter
    diplotype_expansion : List[List[tuple]]
        Each item corresponds to the element in `counter` with same index.
        The item is a list of index pairs. Each index points to an element
        in `parent_haplotypes`, and the pair stands for an admissible parent
        haplotype couple.

    """

    counter = Counter(phenotypes)

    def factorize_to_index(phenotype):
        return [
            (haplotypes.index(x), haplotypes.index(y))
            for (x, y) in factorize(phenotype)
        ]

    return (
        counter,
        list(map(factorize_to_index, counter))
    )


def build_diplotype_matrix(
        haplotypes: List[Tuple[str]],
        diplotype_expansion: List[List[multiset]]
):
    """Haplotype multiplicity in a 'N haplotypes' * 'M diplotypes' matrix

    Points out how many times haplotype n is present in diplotype m

    """

    diplotypes = reduce(lambda x, y: x + y, diplotype_expansion, [])
    matrix = dok_matrix(
        (len(haplotypes), len(diplotypes)),
        dtype=int
    )

    # Populate matrix
    for (i, diplotype) in enumerate(diplotypes):
        matrix[diplotype[0], i] += 1
        matrix[diplotype[1], i] += 1

    return matrix




