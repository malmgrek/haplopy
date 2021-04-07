"""Data utils for HaploPy

Terminology
-----------

haplotype : A sequence of nucleotides

genotype : A pair of haplotypes

phenotype : A sequence of nucleotide pairs with unspecified genotype

"""

from collections import Counter
from functools import reduce
import itertools
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.sparse import dok_matrix


def find_parent_haplotypes(phenotypes: List[Tuple[str]]) -> List[Tuple[str]]:
    """List parent haplotypes

    """
    unique_phenotypes = set(phenotypes)
    return list(reduce(
        lambda parents, phenotype: parents.union(
            set(itertools.product(*[set(diplo) for diplo in phenotype]))
        ),
        unique_phenotypes,
        set()
    ))


def factorize(phenotype: Tuple[str]) -> List[Tuple[str]]:
    """List admissible genotypes

    """
    factors = list(itertools.product(*[
        set(diplo) for diplo in phenotype
    ]))
    half = len(factors) // 2
    return (
        [(factors[0], factors[0])] if half == 0 else
        list(zip(factors[:half], factors[half:][::-1]))
    )


def describe_phenotypes(phenotypes: List[Tuple[str]]) -> tuple:
    """Phenotype multiplicity and parent genotype expansion

    Returns
    -------

    parent_haplotypes : List[tuple]
        All haplotypes that are admissible parents for some phenotype in
        the dataset.
    counter : collections.Counter
        Occurrence count of each unique phenotype in the dataset.
    genotype_expansion : List[List[tuple]]
        Each item corresponds to the element in `counter` with same index.
        The item is a list of index pairs. Each index points to an element
        in `parent_haplotypes`, and the pair stands for an admissible parent
        haplotype couple.

    """

    counter = Counter(phenotypes)
    parent_haplotypes = find_parent_haplotypes(phenotypes)

    def factorize_to_index(phenotype):
        return [
            (parent_haplotypes.index(x), parent_haplotypes.index(y))
            for (x, y) in factorize(phenotype)
        ]

    genotype_expansion = list(map(factorize_to_index, counter))

    return (parent_haplotypes, counter, genotype_expansion)


def build_genotype_matrix(genotype_expansion, parent_haplotypes):
    """Haplotype multiplicity in a 'N haplotypes' * 'M genotypes' matrix

    Points out how many times haplotype n is present in genotype m

    """

    genotypes = reduce(lambda x, y: x + y, genotype_expansion, [])
    matrix = dok_matrix(
        (len(parent_haplotypes), len(genotypes)),
        dtype=int
    )

    # Populate matrix
    for (i, genotype) in enumerate(genotypes):
        matrix[genotype[0], i] += 1
        matrix[genotype[1], i] += 1

    return matrix
