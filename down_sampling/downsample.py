import numpy as np
import sys
sys.path.append('../')    # import from folder one directory out
import util
import matplotlib.pyplot as plt


# divergance and information measures
def get_KL_div(p, q):   # p and q distributions (ie non-negative)
    # add a small value to avoid dividings by zero
    e = 2**-100
    norm_p = (p+e)/(p+e).sum()
    norm_q = (q+e)/(q+e).sum()
    return (norm_p*np.log2(norm_p / norm_q)).sum()


def est_dist_hist(np_period, min_max):
    (df_min, df_max) = min_max
    [count, bin_lower]  = np.histogram(np_period, bins=5000, range=(np.floor(df_min), np.ceil(df_max)))
    return count/count.sum()    # normalise to distribution


def get_down_sample(original, reduction):
    original_samples = len(original)
    indicies = np.linspace(0, original_samples-1, int(original_samples/reduction)).astype(int)
    return original[indicies]


def run_down_sample(original, reductions, min_max):    # input numpy array
    original_hist = est_dist_hist(original, min_max)
    KL_div = []
    for reduction in reductions:
        down_hist = est_dist_hist(get_down_sample(original, reduction), min_max)
        KL_div.append(get_KL_div(original_hist, down_hist))
    return np.array(KL_div)


def main(df, upper_down):
    min_max = (df.min().min(), df.max().max())
    reductions = np.linspace(1, upper_down, 20)
    mean_KL = np.zeros(len(reductions))
    count = 0
    for index, row in df.iterrows():
        mean_KL += run_down_sample(row.values, reductions, min_max)
        count += 1
    mean_KL /= count
    plt.plot(reductions, mean_KL)
    plt.xlabel('Reduction Size')
    plt.ylabel('KL divergance')
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) == 1:   # upper limit of downsampling
        sys.argv.append('20')
    if len(sys.argv) == 2:   # dataset not given at command line
        sys.argv.append('../Datasets/star_light/train.csv')
    for dataset in sys.argv[2:]:
        df = util.csv_to_df(dataset)
        main(df, int(sys.argv[1]))
