import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
'''
functions for loading data from csv
and getting basic attributed of data
as well as visulisation 
'''


def csv_to_df(filename):
    return pd.read_csv(filename, header=None)


def unique_classes(df):
    try:
        return df[0].unique()
    except:
        return df['3'].unique()   # if column names are used


def mean_std_of_class(df, class_label):
    try:
        current_class = df.loc[df[0]==class_label].iloc[:, 1:]
    except:
        current_class = df.loc[df['3']==class_label].iloc[:, 1:]   # if column names are used
    return current_class.mean(0), current_class.std(0)


def min_max_of_df(df):
    return df.min().min(), df.max().max()


def plot_period_std(ax, mean, std):
    ax.plot(np.linspace(0,1,len(mean)),mean+std, 'r-')
    ax.plot(np.linspace(0,1,len(mean)),mean-std, 'r-')
    ax.fill_between(np.linspace(0,1,len(mean)), mean-std, mean+std, alpha=0.3)
    ax.plot(np.linspace(0,1,len(mean)),mean)


def load_plot_data(filename):
    df = csv_to_df(filename)
    class_labels = unique_classes(df)
    print("Class labels of dataset: '"+filename+"' are: "+str(class_labels)+'\n')
    df_min, df_max = min_max_of_df(df)
    fig = plt.figure()
    ax_count = 1
    for class_label in class_labels:
        ax = fig.add_subplot(len(class_labels), 1, ax_count)   # count change count to using class_label: assumes classes are ordered 1-n though
        ax.set_ylim(df_min, df_max)
        class_mean, class_std = mean_std_of_class(df, class_label)
        plot_period_std(ax, class_mean, class_std)
        ax.set_title('Class label: '+str(class_label))
        if ax_count < len(class_labels):
            plt.xticks([])   # overlaps with titles
        ax_count += 1
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) == 1:   # dataset not given at command line
        sys.argv.append('Datasets/star_light/train.csv')
        sys.argv.append('Datasets/star_light/test.csv')
    for dataset in sys.argv[1:]:
        load_plot_data(dataset)

