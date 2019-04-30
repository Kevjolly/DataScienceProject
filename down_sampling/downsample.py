import numpy as np
import sys
sys.path.append('../')    # import from folder one directory out
import util
import matplotlib.pyplot as plt
import pandas as pd
import simple_classify


def frequency_dist(time_series, original_len):
    ft = np.fft.rfft(time_series)
    dist = np.zeros(int((original_len/2)+1))
    dist[:len(ft)] = abs(ft)  # pad with zeros of higher frequencies that can't be captured
    return dist/dist.sum()# make to distribution


def get_KL_div(p, q):   # p and q distributions (ie non-negative)
    # TODO: symetric KL
    # add a small value to avoid dividings by zero
    e = 2**-100
    norm_p = (p+e)/(p+e).sum()
    norm_q = (q+e)/(q+e).sum()
    return (norm_p*np.log2(norm_p / norm_q)).sum()


def est_dist_hist(np_period, min_max, bins):
    (df_min, df_max) = min_max
    [count, bin_lower]  = np.histogram(np_period, bins=bins, range=(np.floor(df_min), np.ceil(df_max)))
    return count/count.sum()    # normalise to distribution


def even_down_sample(original, reduction):
    original_samples = len(original)
    indicies = np.linspace(0, original_samples-1, int(original_samples/reduction)).astype(int)
    return original[indicies]


def run_down_sample_hist(original, reductions, min_max, bins):    # input numpy array
    original_len = len(original)
    original_hist = est_dist_hist(original, min_max, bins)
    KL_div_hist = []
    for reduction in reductions:
        even_down = even_down_sample(original, reduction)
        
        down_hist = est_dist_hist(even_down, min_max, bins)

        KL_div_hist.append(get_KL_div(original_hist, down_hist))
    return np.array(KL_div_hist)


def run_down_sample_fft(original, reductions, min_max, bins):    # input numpy array
    original_len = len(original)
    original_fft = frequency_dist(original, original_len)
    KL_div_fft = []
    for reduction in reductions:
        even_down = even_down_sample(original, reduction)
        
        down_fft = frequency_dist(even_down, original_len)

        KL_div_fft.append(get_KL_div(original_fft, down_fft))
    return np.array(KL_div_fft)


def plot_down_sampled(original, reductions):
    fig = plt.figure()
    plot_num = 1
    print(original.shape)
    for reduction in reductions:
        even_down = even_down_sample(original, reduction)
        plt.subplot(len(reductions), 1, plot_num)
        plt.plot(np.linspace(0, 1, len(even_down)), even_down)
        plot_num += 1
    plt.show()


def get_mean(df, reductions, run_function, bins):
    min_max = (df.min().min(), df.max().max())
    
    mean_KL = np.zeros(len(reductions))
    count = 0
    for index, row in df.iterrows():
        mean_KL += run_function(row.values, reductions, min_max, bins)
        count += 1
    return mean_KL / count


def downsampled_df(df, reduction):
    down_df = [0]*df.shape[0]
    count = 0
    for index, row in df.iterrows():
        down_df[count] = even_down_sample(row.values, reduction)
        count += 1
    return pd.DataFrame(np.array(down_df)).set_index(df.index)


def get_distances(df, reductions):
    means = []
    stds = []
    for reduction in reductions:
        down_df = downsampled_df(df, reduction)
        class_mean, class_std = util.mean_std_of_class(down_df)
        means.append(np.mean(class_mean))
        stds.append(np.mean(class_std))
    return (means, stds)


def test_distances(df, reductions):
    classes = util.unique_classes(df)
    fig = plt.figure()
    plot_num = 1
    for star_type in classes:
        all_class = df.loc[[star_type]]
        (means, stds) = get_distances(all_class, reductions)
        plt.subplot(len(classes), 1, plot_num)
        plt.plot(reductions, means, label = 'mean class: '+str(star_type))
        # plt.plot(reductions, stds, label = 'std class: '+str(star_type))
        plt.legend()
        plot_num += 1
    plt.show()


def test_classification(df, reductions):
    train_df, test_df = simple_classify.split_df(df)
    accuracies = []
    for reduction in reductions:
        train_down = downsampled_df(train_df, reduction)
        test_down = downsampled_df(test_df, reduction)
        accuracies.append(simple_classify.classify_df(train_down, test_down))
    plt.plot(reductions, accuracies)
    plt.xlabel('Reduction Size')
    plt.ylabel('Test Accuracy')
    plt.show()


def main(df, upper_down):
    reductions = np.linspace(1, upper_down, 500)
    print(reductions)
    test_classification(df, reductions)
    # test_distances(df, reductions)
    # plot_down_sampled(df.iloc[0], reductions)
    bins = [10, 100, 200, 500, 1000]
    for single_bin in bins:   
        # current_mean = get_mean(df, reductions, run_down_sample_hist, single_bin)
        # plt.plot(reductions, current_mean, label ='hist: bin size '+str(single_bin))
        print(single_bin)
    plt.plot(reductions, get_mean(df, reductions, run_down_sample_fft, 0), label='fourier transform')
    plt.legend()
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
