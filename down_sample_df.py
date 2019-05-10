import numpy as np
import matplotlib.pyplot as plt
import sys
import util
import pandas as pd

labels = {}
labels[1] = "Cephoids"
labels[2] = "Eclipsing Binary"
labels[3] = "RR Lyrae"
#sys.stdout = open('clean_test_means.txt', 'w')

def plot_period_std(ax, mean, std):
    ax.plot(np.linspace(0,1,len(mean)),mean+std, 'r-', alpha=0.3)
    ax.plot(np.linspace(0,1,len(mean)),mean-std, 'r-', alpha=0.3)
    ax.fill_between(np.linspace(0,1,len(mean)), mean-std, mean+std, alpha=0.1)
    ax.plot(np.linspace(0,1,len(mean)),mean, alpha=0.3)

def plot_outliers(ax, outlier, c, alpha):
    ax.plot(np.linspace(0,1,len(outlier)),outlier, c + '-', alpha=alpha)


def even_down_sample(original, reduction, start):
	# original is numpy array (or df row?) of single time series
	# reduction is the reduction size
	# start is how much phase shifting 
    original_samples = len(original)
    indicies = np.linspace(0, original_samples-1, int(original_samples/reduction)).astype(int) + start
    indicies = indicies[indicies<original_samples] # shifted start position needs last index removed as past end
    return original[indicies]

def find_anomalies_mean_L1(filename):
    df = util.csv_to_df(filename)
    df = df*-1
    class_labels = util.unique_classes(df)
    print("Class labels of dataset: '"+filename+"' are: "+str(class_labels)+'\n')
    df_min, df_max = util.min_max_of_df(df)
    fig = plt.figure()
    ax_count = 1
    for class_label in class_labels:
        if class_label==1:
            continue
        if class_label==3:
            continue

        dfC = df.loc[[class_label]]
        dfC = dfC.apply(lambda x: even_down_sample(x, 20, 0), axis=1)
        class_mean, class_std = util.mean_std_of_class(dfC)
        #print(class_mean.values.tolist())
        dfCL1 = dfC.subtract(class_mean, axis=1)
        dfCL1 = dfCL1.abs()
        dfC["L1"] = dfCL1.sum(axis=1)
        #dfCL1 = dfCL1 ** 2
        #dfC["L1"] = np.sqrt(dfCL1.sum(axis=1))
        q3 = dfC["L1"].quantile(0.75)
        q1 = dfC["L1"].quantile(0.25)
        print(q3, q1)
        iqr = q3-q1
        dfC["iter"] = [i for i in range(dfC.shape[0])]
        dfCOutliers = dfC[(dfC.L1>q3+1.5*iqr) | (dfC.L1<q1-1.5*iqr)]
        dfCNonOutliers = dfC[(dfC.L1<q3+1.5*iqr) & (dfC.L1>q1-1.5*iqr)]
        print("outliers: ", dfCOutliers["iter"])
        dfCOutliers.drop('L1', axis=1, inplace=True)
        dfCOutliers.drop('iter', axis=1, inplace=True)
        #dfCOutliers.drop('L1', axis=1, inplace=True)
        #dfCNonOutliers.drop('L1', axis=1, inplace=True)

        ax = fig.add_subplot(len(class_labels), 1, ax_count)   # count change count to using class_label: assumes classes are ordered 1-n though
        ax.set_ylim(df_min, df_max)
        plot_period_std(ax, class_mean, class_std)
        #plot_outliers(ax, dfCOutliers.iloc[0, :])
        dfCOutliers.apply(lambda x: plot_outliers(ax, x, 'g', 1), axis=1)
        #dfCNonOutliers.apply(lambda x: plot_outliers(ax, x, 'b', 0.1), axis=1)
        ax.set_title(labels[class_label])
        if ax_count < len(class_labels):
            plt.xticks([])   # overlaps with titles
        ax_count += 1
        print(labels[class_label], ": Rows:", dfC.shape[0] , ", Outliers:" , dfCOutliers.shape[0] ,'\n')
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) == 1:   # dataset not given at command line
        sys.argv.append('Datasets/star_light/train.csv')
        sys.argv.append('Datasets/star_light/test.csv')
    for dataset in sys.argv[1:]:
        find_anomalies_mean_L1(dataset)