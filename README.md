# HaploPy

Estimating haplotype frequencies in population 
using measurements of unphased genotype data.

## Examples

```
import haplopy as hp


proba_haplotypes = {'TGT': 33, 
                    'TTT': 20,      
                    'GCT': 10, 
                    'CCT': 12,
                    'CCG': 5,
                    'TCT': 20}
phenotype_data = hp.PhenotypeData.simulate(proba_haplotypes, nobs=100)
```