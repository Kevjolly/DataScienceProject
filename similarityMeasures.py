import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

import sys
import util
'''
functions for finiding anomalies and normalizing dataframe

call this scipt with n number of filepaths to datasets
(default of OGLE train/test is set for no arg inputs)
'''

def find_class_summaries(filename, type):
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
        if type=="L1":
            dfCL1 = dfCL1.abs()
        elif type=="L2":
            dfCL1 = dfCL1 ** 2

        dfC["L1"] = dfCL1.sum(axis=1)
        dfC['num'] = np.arange(len(dfC))
        q3 = dfC["L1"].quantile(0.75)
        q1 = dfC["L1"].quantile(0.25)
        iqr = q3-q1
        outlierUpperBound = q3+1.5*iqr
        outlierLowerBound = q1-1.5*iqr
        dfCOutliers = dfC[(dfC.L1>q3+1.5*iqr) | (dfC.L1<q1-1.5*iqr)]
        dfCOutliers.drop('L1', axis=1, inplace=True)
        print("Rows:", dfC.shape[0] , ", Outliers:" , dfCOutliers.shape[0] ,'\n')
        class_summary["outliers"] = dfCOutliers["num"]
        class_summaries[class_label] = class_summary

    return class_summaries
        

if __name__ == '__main__':
    if len(sys.argv) == 1:   # dataset not given at command line
        sys.argv.append('Datasets/star_light/train.csv')
        #sys.argv.append('Datasets/star_light/test.csv')
    for dataset in sys.argv[1:]:
        class_summaries_L1 = find_class_summaries(dataset, "L1")
        class_summaries_L2 = find_class_summaries(dataset, "L2")
        for label, summary in class_summaries_L1.items():
            y1 = set(class_summaries_L1[label]["outliers"].values.flatten())
            y2 = set(class_summaries_L2[label]["outliers"].values.flatten())
            print(label, y1, y2, "\n")
            venn2([y1, y2], set_labels = ('L1', 'L2'))
            plt.show()
