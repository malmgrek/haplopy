"""Data utils for HaploPy

Terminology
-----------

haplotype : A sequence of nucleotides

diplotype : A pair of haplotypes

phenotype : A sequence of nucleotide pairs with unspecified diplotype

"""

from collections import Counter
from functools import reduce
import itertools
import re
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.sparse import dok_matrix


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


def fill_diplotypes(
        diplotypes: List[Tuple[Tuple[str]]],
        haplotypes: List[Tuple[str]]
) -> List[Tuple[Tuple[str]]]:
    """Attempt filling in missing values inside diplotypes

    Uses regular expressions to fill in missing SNPs with admissible values
    that are present in `haplotypes`.

    Leaves unmatchable patterns as they are.

    """
    raise NotImplementedError
