"""Data utils for HaploPy

Terminology
-----------

TODO / FIXME: Update to new container formalism
TODO: Make the whole code work with 0's and 1's instead of strings

haplotype : A sequence of nucleotides.
            For example ("a", "b").

diplotype : A pair of haplotypes.
            For example (("a", "b"), ("A", "B")).

phenotype : A sequence of nucleotide pairs with unspecified diplotype.
            For example ("Aa", "Bb").

genotype  : An item in the phenotype.
            For example "Aa".

"""

from __future__ import annotations

from collections import Counter
from functools import reduce
import itertools
import re
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.sparse import dok_matrix


def match(haplotype: Tuple[str], haplotypes: List[Tuple[str]]) -> List[Tuple[str]]:
    """Find haplotypes that match a given haplotype

    Example
    -------

    >>> match( ("a", "."), [("a", "b"), ("a", "B"), ("A", "B")])
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


class Phenotype():

    def __init__(self, *genotypes):
        self.genotypes = tuple(map(lambda g: "".join(sorted(g)), genotypes))

    def __repr__(self):
        return f"Phenotype{self.genotypes}"

    def __eq__(self, other):
        return self.genotypes == other.genotypes

    def __hash__(self):
        return hash(self.genotypes)

    @property
    def admissible_haplotypes(self) -> List[Tuple[str]]:
        """List possible source haplotypes

        """
        return list(itertools.product(*map(set, self.genotypes)))

    def factorize_to_diplotypes(self):
        haplotypes = self.admissible_haplotypes
        half = len(haplotypes) // 2
        return (
            [Diplotype(haplotypes[0], haplotypes[0])] if half == 0 else
            [
                Diplotype(x, y) for (x, y) in zip(
                    haplotypes[:half], haplotypes[half:][::-1]
                )
            ]
        )
        return

    def factorize_to_index(self, haplotypes):
        return [
            (
                haplotypes.index(diplotype.haplotypes[0]),
                haplotypes.index(diplotype.haplotypes[1])
            )
            for diplotype in self.factorize_to_diplotypes()
        ]


class Diplotype():

    def __init__(self, x, y):
        self.haplotypes = tuple(sorted([x, y]))

    def __repr__(self):
        return f"Diplotype{self.haplotypes}"

    def __eq__(self, other):
        return self.haplotypes == other.haplotypes

    def __hash__(self):
        return hash(self.haplotypes)

    def unphase(self) -> Phenotype:
        return Phenotype(
            *("".join(snp) for snp in zip(*self.haplotypes))
        )

    def fill(self, haplotypes):
        return


def unphase(diplotype: Diplotype) -> Phenotype:
    """The most basic operation diplotype to phenotype mapping

    Example
    -------

    >>> unphase((("A", "T", "G"), ("A", "A", "C")))
    ("AA", "AT", "GC")

    TODO: Test

    """
    return tuple("".join(sorted(snp)) for snp in zip(*diplotype))


def find_parent_haplotypes(phenotypes: List[Tuple[str]]) -> List[Tuple[str]]:
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
# Helpers for the multinomial EM algorithm
#


def _build_diplotype_representation(haplotypes, phenotypes: List[Phenotype]):
    counter = Counter(phenotypes)
    return (
        counter,
        [phenotype.factorize_to_index(haplotypes) for phenotype in counter]
    )


def _build_diplotype_matrix(haplotypes, diplotype_representation):
    return


def build_diplotype_expansion(
        haplotypes: List[Tuple[str]],
        phenotypes: List[Tuple[str]]
) -> Tuple[Counter, List[List[Tuple[int]]]]:
    """Phenotype multiplicity and parent diplotype expansion

    Returns
    -------

    diplotype_expansion : List[List[tuple]]
        Each item corresponds to the element in `counter` with same index.
        The item is a list of index pairs. Each index points to an element
        in `parent_haplotypes`, and the pair stands for an admissible parent
        haplotype couple.

    FIXME: Counter minds phenotype locus ordering!

    TODO: Use this, but change Model.proba_haplotypes to Model.probas and
          Model.haplotypes

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
        diplotype_expansion: List[List[Tuple[int]]]
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


def find_matching_haplotypes(
        haplotype: Tuple[str],
        haplotypes: List[Tuple[str]]
) -> List[Tuple[str]]:
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


def fill_diplotype(
        diplotype: Tuple[Tuple[str]],
        haplotypes: List[Tuple[str]]
) -> List[Tuple[Tuple[str]]]:
    """Attempt filling in missing values inside a diplotype

    Uses regular expressions to fill in missing SNPs with admissible values
    that are present in `haplotypes`.

    Leaves unmatchable patterns as they are.

    Example
    -------

    >>> fill_diplotype(
    ...     (("A", "."), ("a", "b")),
    ...     [("A", "B"), ("A", "b"), ("a", "B"), ("a", "b")]
    ... )
    [(("A", "B"), ("a", "b")), (("A", "b"), ("a", "b"))]

    TODO: Test

    """

    def expand(h):
        e = find_matching_haplotypes(h, haplotypes)
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
