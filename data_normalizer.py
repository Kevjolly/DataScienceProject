import numpy as np
import matplotlib.pyplot as plt
import sys
import util
'''
functions for finiding anomalies and normalizing dataframe

call this scipt with n number of filepaths to datasets
(default of OGLE train/test is set for no arg inputs)
'''

def find_anomalies(filename):
    df = util.csv_to_df(filename)
    class_labels = util.unique_classes(df)
    print("Class labels of dataset: '"+filename+"' are: "+str(class_labels)+'\n')
    for class_label in class_labels:
        dfC = df.loc[[class_label]]
        class_mean, class_std = util.mean_std_of_class(dfC)
        

if __name__ == '__main__':
    if len(sys.argv) == 1:   # dataset not given at command line
        sys.argv.append('Datasets/star_light/train.csv')
        #sys.argv.append('Datasets/star_light/test.csv')
    for dataset in sys.argv[1:]:
        find_anomalies(dataset)