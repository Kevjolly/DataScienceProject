import numpy as np
import matplotlib.pyplot as plt
import sys
import util
'''
functions for finiding anomalies and normalizing dataframe

call this scipt with n number of filepaths to datasets
(default of OGLE train/test is set for no arg inputs)
'''

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

def plot_outliers(ax, outlier, c, alpha, start_class):
    #if start_class==3:
    #    outlier = np.roll(outlier, -225)
    #if start_class==1:
    #    outlier = np.roll(outlier, -350)
    ax.plot(np.linspace(0,1,len(outlier)),outlier, c + '-', alpha=alpha)


def find_class_summaries(filename):
    df = util.csv_to_df(filename)
    class_labels = util.unique_classes(df)
    print("Class labels of dataset: '"+filename+"' are: "+str(class_labels)+'\n')
    df_min, df_max = util.min_max_of_df(df)
    fig = plt.figure()
    ax_count = 1
    class_summaries = {}
    for class_label in class_labels:
        class_summary = {}

        dfC = df.loc[[class_label]]
        class_mean, class_std = util.mean_std_of_class(dfC)

        dfCL1 = dfC.subtract(class_mean, axis=1)
        dfCL1 = dfCL1.abs()
        #dfCL1 = dfCL1 ** 2

        dfC["L1"] = dfCL1.sum(axis=1)
        q3 = dfC["L1"].quantile(0.75)
        q1 = dfC["L1"].quantile(0.25)
        iqr = q3-q1
        outlierUpperBound = q3+1.5*iqr
        outlierLowerBound = q1-1.5*iqr
        dfCOutliers = dfC[(dfC.L1>q3+1.5*iqr) | (dfC.L1<q1-1.5*iqr)]
        dfCOutliers.drop('L1', axis=1, inplace=True)
        print("Rows:", dfC.shape[0] , ", Outliers:" , dfCOutliers.shape[0] ,'\n')
        class_summary["mean"] = class_mean
        class_summary["outliers"] = dfCOutliers
        class_summary["outlierLowerBound"] = outlierLowerBound
        class_summary["outlierUpperBound"] = outlierUpperBound
        class_summaries[class_label] = class_summary

    return class_summaries
        

def find_anomalies_mean_L1(filename):
    df = util.csv_to_df(filename)
    df = df*-1
    class_labels = util.unique_classes(df)
    print("Class labels of dataset: '"+filename+"' are: "+str(class_labels)+'\n')
    df_min, df_max = util.min_max_of_df(df)
    fig = plt.figure()
    ax_count = 1
    for class_label in class_labels:
        dfC = df.loc[[class_label]]
        class_mean, class_std = util.mean_std_of_class(dfC)
        #print(class_mean.values.tolist())
        dfCL1 = dfC.subtract(class_mean, axis=1)
        dfCL1 = dfCL1.abs()
        dfC["L1"] = dfCL1.sum(axis=1)
        #dfCL1 = dfCL1 ** 2
        #dfC["L1"] = np.sqrt(dfCL1.sum(axis=1))
        q3 = dfC["L1"].quantile(0.75)
        q1 = dfC["L1"].quantile(0.25)
        iqr = q3-q1
        dfCOutliers = dfC[(dfC.L1>q3+1.5*iqr) | (dfC.L1<q1-1.5*iqr)]
        dfCNonOutliers = dfC[(dfC.L1<q3+1.5*iqr) & (dfC.L1>q1-1.5*iqr)]
        dfCOutliers.drop('L1', axis=1, inplace=True)
        #dfCNonOutliers.drop('L1', axis=1, inplace=True)

        ax = fig.add_subplot(len(class_labels), 1, ax_count)   # count change count to using class_label: assumes classes are ordered 1-n though
        ax.set_ylim(df_min, df_max)
        plot_period_std(ax, class_mean, class_std)
        #plot_outliers(ax, dfCOutliers.iloc[0, :])
        dfCOutliers.apply(lambda x: plot_outliers(ax, x, 'g', 1, class_label), axis=1)
        #dfCNonOutliers.apply(lambda x: plot_outliers(ax, x, 'b', 0.1), axis=1)
        ax.set_title(labels[class_label])
        if ax_count < len(class_labels):
            plt.xticks([])   # overlaps with titles
        ax_count += 1
        print(labels[class_label], ": Rows:", dfC.shape[0] , ", Outliers:" , dfCOutliers.shape[0] ,'\n')
    plt.show()
        
def find_new_outliers(dfOutiers, mean, lB, uB):
    dfOutiers = dfOutiers.subtract(mean, axis=1)
    dfOutiers = dfOutiers.abs()
    dfOutiers["L1"] = dfOutiers.sum(axis=1)
    dfOutiers = dfOutiers[(dfOutiers.L1>uB) | (dfOutiers.L1<lB)]
    return dfOutiers.shape[0]

def separate_new_outliers(dfOutiers, mean, lB, uB, fig, n):
    
    dfInit = dfOutiers
    df_min, df_max = util.min_max_of_df(dfInit)
    dfOutiers = dfOutiers.subtract(mean, axis=1)
    dfOutiers = dfOutiers.abs()
    dfOutiers["L1"] = dfOutiers.sum(axis=1)
    dfInit["L1"] = dfOutiers["L1"]
    dfOutiersNew = dfInit[(dfInit.L1>uB) | (dfInit.L1<lB)]
    dfOutiersOld = dfInit[(dfInit.L1<uB) | (dfInit.L1>lB)]
    dfInit.drop('L1', axis=1, inplace=True)
    dfOutiersNew.drop('L1', axis=1, inplace=True)
    dfOutiersOld.drop('L1', axis=1, inplace=True)

    ax = fig.add_subplot(3, 1, n)   # count change count to using class_label: assumes classes are ordered 1-n though
    ax.set_ylim(df_min, df_max)
    dfOutiersOld.apply(lambda x: plot_outliers(ax, x, 'g', 1), axis=1)
    dfOutiersNew.apply(lambda x: plot_outliers(ax, x, 'b',0.4), axis=1)
    


if __name__ == '__main__':
    if len(sys.argv) == 1:   # dataset not given at command line
        sys.argv.append('Datasets/star_light/train.csv')
        #sys.argv.append('Datasets/star_light/test.csv')
    for dataset in sys.argv[1:]:
        find_anomalies_mean_L1(dataset)