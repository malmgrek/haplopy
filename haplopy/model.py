import itertools
import numpy as np
import pandas as pd


__version__ = '0.0.1.dev'
__author__ = 'Stratos Staboulis'


def update_configuration(defaults, custom_settings):
    """Updates configuration dict after asserting for unknown keys

    :param defaults: Default configuration
    :param custom_settings: New configuration
    :return: config: Updated configuration

    """
    # update config
    unknown_keys = set(custom_settings) - set(defaults)
    assert unknown_keys <= set(), \
        'Unknown argument(s): ' + \
        ', '.join(str(key) for key in unknown_keys)
    config = defaults.copy()
    config.update(custom_settings)
    return config


def parse_csv_data(path_to_file,
                   index='Sample ID',
                   columns='NCBI SNP Reference',
                   values='Call',
                   duplicate=False):
    """Parse an input CSV genotype data into Python list form

    :param path_to_file: Source file
    :param index: Sample index.
    :param columns:
    :param values:
    :param duplicate:
    :return: gt: Full list of genotypes
             gt_no_missing: Genotypes without any missing values
    """
    df = pd.read_csv(path_to_file)

    # remove duplicate gene names
    if duplicate:
        df = df[df[duplicate[0]] != duplicate[1]]
    df = df.dropna().pivot(index=index,
                           columns=columns,
                           values=values).replace({'/': '',
                                                   'NOAMP': '**',
                                                   'UND': '**',
                                                   '-': '*'},
                                                  regex=True)

    # list from DataFrame
    gt = [list(row) for row in df.values]

    # remove missing values
    gt_no_missing = [row for row in gt if '*' not in ''.join(row)]
    return gt, gt_no_missing


def remove_duplicates(l):
    """Remove duplicates from list in order preserving way

    :param l: input list.
    :return: output list with unique items.
    """
    checked = []
    for item in l:
        if (item not in checked) and ([item[1], item[0]] not in checked):
            checked.append(item)
    return checked


def simulate_genotype_data(num_data, haplotype_dict):
    """Simulate unphased genotype data

    :param num_data: Number of output genotype samples
    :param haplotype_dict: Dictionary containing population haplotypes and
    their relative frequencies.
    :return: List of drawn unphased genotypes. List length is num_data.

    """
    haplotype_list = list(haplotype_dict.keys())
    p = np.array(list(haplotype_dict.values()))
    p /= np.sum(p)  # normalize
    gt = []
    for n in range(0, num_data):
        samp = list(np.random.multinomial(1, p))
        h1 = [haplotype_list[i] for i, v in enumerate(samp) if v == 1][0]
        samp = list(np.random.multinomial(1, p))
        h2 = [haplotype_list[i] for i, v in enumerate(samp) if v == 1][0]
        gt.append([a1 + a2 for a1, a2 in zip(h1, h2)])
    return gt


def reconstruct_parent_haplotype(gt):
    """Find possible parent haplotypes from an unphased genotype measurement

    :param gt: List of unphased genotypes of arbitrary fixed length.
    :return: ht_gt: List of lists. Each sublist contains all possible haplotype
                    pairs that can yield the corresponding unphased genotype.
    :return: ht_all: List containing all possible parent haplotypes in the
                     unphased genotype data as a whole.

    """
    ht_all = []
    ht_gt = []
    for pair in gt:
        _ht = []
        expand_pair = list(itertools.product(*pair))
        for i in range(0, len(expand_pair) // 2):
            row_top = ''.join(expand_pair[i])
            row_bot = ''.join(expand_pair[-i - 1])
            ht_all.append(row_top)
            ht_all.append(row_bot)
            _ht.append([row_top, row_bot])
        _ht = remove_duplicates(_ht)
        ht_gt.append(_ht)
    ht_all = list(set(ht_all))
    return ht_gt, ht_all


def build_haplotype_indicator(gt, ht_gt, ht_all):
    """Constructs a Kroenecker style indicator matrix for the EM algorithm.

    :param gt: List of unphased genotype data of fixed length.
    :param ht_gt: List of lists. Each sublist contains all possible haplotype
                  pairs that can yield the corresponding unphased genotype.
    :param ht_all: List containing all possible parent haplotypes in the
                   unphased genotype data as a whole.
    :return: Matrix with N_genotype rows and N_max columns where N_max is the
             maximal number of possible parent haplotypes over all genotypes.

    """
    num_data = len(gt)
    num_loc = len(gt[0])
    indicator_dict = dict.fromkeys(ht_all)
    for h in ht_all:
        indicator = np.zeros((num_data, 2 ** (num_loc - 1)))
        for i, ht in enumerate(ht_gt):
            row = np.zeros(2 ** (num_loc - 1))
            for j, h_pair in enumerate(ht):
                row[j] = h_pair.count(h)
            indicator[i, :] = row
        indicator_dict[h] = indicator
    return indicator_dict


def expectation_haplotype_probability(p_dict, ht):
    """Genotype probabilities for the Expectation step

    :param p_dict: Dictionary of relative haplotype frequencies. Keys are
                   haplotype codes.
    :param ht: List of haplotype pairs.
    :return: Numpy array of genotype probabilities corresponding to each
             haplotype pair.

    """
    p_e = []
    for pair in ht:
        if pair[0] == pair[1]:
            p_e.append(p_dict[pair[0]] ** 2)
        else:
            p_e.append(2. * p_dict[pair[0]] * p_dict[pair[1]])
    return np.array(p_e)


def build_maximization_matrix(p_dict, gt, ht_gt):
    """Matrix of genotype probabilities for the maximization step

    :param p_dict: Dictionary of relative haplotype frequencies. Keys are
    haplotype codes.
    :param ht_gt: List of lists. Each sublist contains all possible haplotype
    pairs that can yield the corresponding unphased genotype.
    :param gt: Unphased genotype data
    :return: Matrix as a Numpy array.

    """
    num_loc = len(gt[0])
    num_data = len(gt)
    p_mat = np.zeros((num_data, 2 ** (num_loc - 1)))
    log_likelihood = 0.0
    for i, ht in enumerate(ht_gt):
        p_e = expectation_haplotype_probability(p_dict, ht)
        p_e_total = np.sum(p_e)
        log_likelihood -= np.log(p_e_total)
        p_m = p_e / p_e_total / num_data
        p_mat[i, :len(p_m)] = p_m
    return p_mat, log_likelihood


def evaluate_posterior(p_dict, ht):
    p_e = expectation_haplotype_probability(p_dict, ht)
    p_e_total = np.sum(p_e)
    p_m = p_e / p_e_total
    return p_m


def evaluate_log_likelihood(p_mat):
    val = np.log(p_mat.sum(axis=1)).sum()
    return val


class HaplotypeAnalysis(object):
    """Population haplotype frequency estimation from unphased genotype data

    """

    DEFAULTS = {
        'path_to_data': None,
        'genotype': None
    }

    def __init__(self, **kwargs):

        # update configuration
        config = update_configuration(self.DEFAULTS, kwargs)

        # reconstruct parent haplotype data from the genotype
        gt = config['genotype']
        ht_gt, ht_all = reconstruct_parent_haplotype(gt)

        # haplotype indicator
        ht_ind = build_haplotype_indicator(gt, ht_gt, ht_all)

        # update instance attributes
        self.config = config
        self.genotype = gt
        self.parent_haplotypes = ht_all
        self.parent_haplotypes_per_genotype = ht_gt
        self.haplotype_indicator = ht_ind
        self.frequency = dict.fromkeys(ht_all, 1 / len(ht_all))
        self.log_likelihood = np.inf

    def estimate_frequency(self, max_iter=20):
        """Estimate haplotype frequencies from unphased genotype data

        Uses the EM algorithm.

        :return: Frequency estimated stored in self.frequency

        """
        # TODO Keep track of the evolution of the probability iterates
        # TODO Implement log-likelihood calculator
        # TODO Randomized values for initial frequencies

        # expectation maximization algorithm
        logl_list = [np.inf]  # initial value "infinity"
        i_iter = 1
        convergence = 1e-8  # stop if log-likelihood goes under this
        while i_iter <= max_iter:
            p_mat, logl = build_maximization_matrix(
                self.frequency,
                self.genotype,
                self.parent_haplotypes_per_genotype)
            for haplotype in self.parent_haplotypes:
                frequency_new = \
                    0.5 * np.sum(self.haplotype_indicator[haplotype] * p_mat)
                # update frequencies
                if frequency_new <= 1.0e-06:
                    frequency_new = 0.
                self.frequency[haplotype] = frequency_new
            logl_list.append(logl)
            # print("Iteration %d, log-likelihood: %f." %(i_iter,ll))
            # check relative change
            change_logl = \
                (logl_list[i_iter-1] - logl_list[i_iter]) / logl_list[i_iter]
            if change_logl < convergence:
                print("Converged.")  # stop if converged
                self.log_likelihood = logl_list[i_iter]
                break
            i_iter += 1
            
    def run_using_random_initialization(self, num_run=10, max_iter=20):
        """Run the EM-algorithm using randomly initialized frequencies

        Runs for a given number of runs and return the frequencies
        corresponding to the lowest log-likelihood.

        """
        best_logl = np.inf
        for r in range(num_run):
            freqs = \
                np.random.dirichlet(np.ones(len(self.parent_haplotypes)), 1)[0]
            for f in range(len(freqs)):
                self.frequency[self.parent_haplotypes[f]] = freqs[f]
            self.log_likelihood = np.inf
            self.estimate_frequency(max_iter)
            if self.log_likelihood < best_logl:
                best_logl = self.log_likelihood
                best_freq = self.frequency
            print("Done %d out of %d runs."%(r + 1, num_run))
        self.log_likelihood = best_logl
        self.frequency = best_freq
        
    def impute_missing_data(self, gt_missing):
        """Imputes missing data points (denoted by asterisks) in input string

        It assumes that the haplotype frequencies have been
        estimated (or have been otherwise assinged, e.g., from a known dataset).

        Okay so first get the possible haplotypes for the genotype with missing
        data where the missing data is marked as *.
        So if gt_missing=['A/B','*/*','A/B'] you'll get
        ht_gt_missing_all=[['A*A', 'B*B'], ['A*B', 'B*A']]
        ht_gt_missing_all, ht_missing_all =
            reconstruct_parent_haplotype(gt_missing)
        
        Next we need to get the parent haplotypes that are possible for this
        specific phenotype disregarding the missing snips. So based on the
        previous example we should compare the ht_gt_missing_all to the existing
        parent haplotypes based on the values that are not missing and collect
        all the possible combinations. So those haplotypes which have
        A*B,B*B,A*B and B*A where the missing value can be anything.
        So the function call could be something like:
        
        ht_gt_parent =
            get_all_existing(ht_gt_missing_all,self.parent_haplotypes)
        
        Finally we need to compute the posterior probabilities for the pairs
        given the estimated frequencies
        I.e.,
        p(h1,h2|gt_missing,freqs) =
            p(gt_missing|h1,h2,freqs)/sum(h1,h2)p(gt|h1,h2,freqs),
        which is given by the build_maximization_matrix function in the p_mat
        variable. So we just need to call:
        
        p_mat,ll = build_maximization_matrix(
                self.frequency,
                gt_missing,
                ht_gt_parent)
        
        And then return p_mat
        return p_mat

        """


if __name__ == "__main__":

    true_frequency = {'TGT': 14. / 32.,
                      'TTT': 4. / 32.,
                      'GCT': 6. / 32.,
                      'CCT': 2. / 32.,
                      'CCG': 4. / 32.,
                      'TCT': 2. / 32.}

    data = simulate_genotype_data(100, true_frequency)
    ha = HaplotypeAnalysis(genotype=data)
    ha.estimate_frequency(max_iter=25)
    ha.run_using_random_initialization(100,30)

    print('\nHaplotype frequencies in population estimated using', len(data),
          '\nunphased genotype measurements:\n')
    for key in true_frequency:
        print(key + ':  ' +
              'True =', true_frequency[key],
              '  Estimated =', ha.frequency[key])
    gt_missing = []
    gt_missing.append(['AB','**','AB'])
    
    ht_gt_missing_all,ht_missing_all = ha.impute_missing_data(gt_missing)
    print(ht_gt_missing_all)
    print(ht_missing_all)
