"""Plotting methods for haplotype probabilities

"""

from collections import Counter
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_haplotypes(
        proba_haplotypes: Dict[Tuple[str], float],
        thres=1e-3,
        title=r"Haplotype relative frequencies (area $\sim$ p)",
        **kwargs
):
    """Visualize non-zero probabilities as a Hinton diagram

    """

    (haplotypes, ps) = zip(
        *sorted(filter(lambda xs: xs[1] > thres, proba_haplotypes.items()))
    )

    (fig, ax) = plt.subplots(**kwargs)
    ax = hinton(ps, ax=ax)
    ax.set_xticks(np.arange(len(ps)))
    ax.set_xticklabels(
        list(map(lambda x: "".join(x), haplotypes)),
        rotation=30,
        ha="right"
    )
    ax.set_title(title)

    return fig


def plot_phenotypes(phenotypes, title="Phenotype counts", **kwargs):
    """Visualize phenotype data as a historgram

    """

    (labels, value_counts) = zip(*Counter(phenotypes).items())

    (fig, ax) = plt.subplots(**kwargs)

    ax.bar(np.arange(len(value_counts)), value_counts, align="center")
    ax.set_xticks(np.arange(len(value_counts)))
    ax.set_xticklabels(
        list(map(lambda xs: "(" + ", ".join(xs) + ")", labels)),
        rotation=30,
        ha="right"
    )
    ax.set_title("Phenotype counts")
    ax.set_ylabel("Value counts")

    fig.tight_layout()

    return fig


def hinton(ps: List[float], ax):
    """Visualize an array as a Hinton diagram

    """

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (i, p) in enumerate(ps):
        size = np.sqrt(abs(p))
        ax.add_patch(
            plt.Rectangle(
                [i - size / 2, -size / 2],
                size,
                size,
                facecolor="white",
                edgecolor="white"
            )
        )

    ax.autoscale_view()

    return ax
