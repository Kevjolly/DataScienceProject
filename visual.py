import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import util
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot

def plot_hist(filename):
    df = util.csv_to_df(filename)
    class_labels = util.unique_classes(df)
    fig = plt.figure()
    fig,axes = plt.subplots(len(class_labels),1,sharex='col',sharey='row')

    ax_count = 0
    for class_label in class_labels:
        observations = df.loc[class_label].values.flatten()
        sns.distplot(observations,ax=axes[ax_count],hist=True,kde=True,
        	label="1")
        ax_count += 1
        
    plt.show()


def plot_joint_hist(filename):
    df = util.csv_to_df(filename)
    class_labels = util.unique_classes(df)
    
    min_size = min(len(df.loc[1].values.flatten()),len(df.loc[2].values.flatten()))
    sns.jointplot(df.loc[1].values.flatten()[:min_size],df.loc[2].values.flatten()[:min_size],kind="kde")
    min_size = min(len(df.loc[1].values.flatten()),len(df.loc[3].values.flatten()))
    sns.jointplot(df.loc[1].values.flatten()[:min_size],df.loc[3].values.flatten()[:min_size],kind="kde")
    min_size = min(len(df.loc[2].values.flatten()),len(df.loc[3].values.flatten()))
    sns.jointplot(df.loc[2].values.flatten()[:min_size],df.loc[3].values.flatten()[:min_size],kind="kde")
    
    plt.show()

def plot_box(filename):
    df = util.csv_to_df(filename)
    class_labels = util.unique_classes(df)

    plt.boxplot([df.loc[1].values.flatten(),
    	df.loc[2].values.flatten(),
    	df.loc[3].values.flatten()])

    plt.show()

def plot_lag(filename):
    df = util.csv_to_df(filename)
    class_labels = util.unique_classes(df)
    fig = plt.figure()
    ax_count = 1
    for class_label in class_labels:
        ax = fig.add_subplot(len(class_labels), 1, ax_count)   # count change count to using class_label: assumes classes are ordered 1-n though
        class_mean, class_std = util.mean_std_of_class(df.loc[[class_label]])
        lag_plot(class_mean,lag=1,label=class_label)
        ax_count += 1
    plt.show()

def plot_ac(filename):
    df = util.csv_to_df(filename)
    class_labels = util.unique_classes(df)
    fig = plt.figure()
    class_mean, class_std = util.mean_std_of_class(df.loc[1])
    autocorrelation_plot(class_mean,label='1')
    class_mean, class_std = util.mean_std_of_class(df.loc[2])
    autocorrelation_plot(class_mean,label='2')
    class_mean, class_std = util.mean_std_of_class(df.loc[3])
    autocorrelation_plot(class_mean,label='3')
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) == 1:   # dataset not given at command line
        sys.argv.append('Datasets/star_light/train.csv')
    for dataset in sys.argv[1:]:
    	# plot_hist(dataset)
        # plot_joint_hist(dataset)
        # plot_box(dataset)
        # plot_lag(dataset)
        # plot_ac(dataset)


