# HaploPy

Estimating haplotype frequencies in population 
using measurements of unphased genotype data.

## Maximum likelihood approach

Consider unphased genotype (phenotype) observations of form

    ['AA', 'CG', 'CC', TA']
    
where the constituent haplotypes are unknown. Based on a set of such observations, 
we want to estimate the phenotype frequencies in the population, i.e., 
``['A', 'C', 'C', 'A']`` etc.. Assuming random mating, the likelihood model can be 
derived from multinomial probability distribution. This project implements the 
Expectation Maximization algorithm for maximum likelihood estimation of the
haplotype frequencies. 

### Example usage

```python
import haplopy as hp


# haplotype frequencies in simulated population
proba_haplotypes = {'TGTC': .33, 
                    'ACTC': .2,      
                    'ACTC': .1, 
                    'TCTC': .12,
                    'TGAC': .5,
                    'ACTG': .2}
                    
# simulate phenotype data
phenotype_data = hp.PhenotypeData.simulate(proba_haplotypes, nobs=100)
phenotype_data.sample  # prints the `observations` 
                       # drawn from multinomial distribution

# run EM to estimate haplotype frequencies
res = hp.expectation_maximization(phenotype_data, max_iter=10)
res  # estimated frequencies
```

## Bayesian approach

TODO