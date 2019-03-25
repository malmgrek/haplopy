# HaploPy

Estimating haplotype frequencies in population 
using measurements of unphased genotype data.

## Examples

```python
import haplopy as hp


# haplotype frequencies in simulated population
proba_haplotypes = {'TGT': .33, 
                    'TTT': .2,      
                    'GCT': .1, 
                    'CCT': .12,
                    'CCG': .5,
                    'TCT': .2}
                    
# simulate phenotype data
phenotype_data = hp.PhenotypeData.simulate(proba_haplotypes, nobs=100)
phenotype_data.sample  # prints the `observations` 
                       # drawn from multinomial distribution

# run EM to estimate haplotype frequencies
res = hp.expectation_maximization(phenotype_data, max_iter=10)
res  # estimated frequencies
```