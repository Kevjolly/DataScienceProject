import numpy as np
import matplotlib.pyplot as plt
import sys
import util
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

labels = {}
labels[1] = "Cephoids"
labels[2] = "Eclipsing Binary"
labels[3] = "RR Lyrae"
#sys.stdout = open('clean_test_means.txt', 'w')
clr = {}
clr[1] = "r-"
clr[2] = "g-"
clr[3] = "b-"

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
	fig = plt.figure(figsize=(9,4))
	ax = plt.subplot(111)
	plt.ylabel('No. of Outliers')
	plt.xlabel('eps')
	plt.title('Estimating eps parameter of DBSCAN method')
	for class_label in class_labels:

		dfC = df.loc[[class_label]]
		dfC1 = dfC.apply(lambda x: even_down_sample(x, 20, 0), axis=1)
		class_mean, class_std = util.mean_std_of_class(dfC1)
		print("std: ",  np.sum(np.square(class_std)))

		dfC2 = dfC1.drop(columns=[0])
		x = []
		y = []
		for i in range(100):
			X = StandardScaler().fit_transform(dfC2.values)
			db = DBSCAN(eps=4+(i)/10, min_samples=10).fit(X)
			labels_i = db.labels_
			s = pd.Series(labels_i)
			sz = s[s==-1].size
			x.append(4+(i)/10)
			y.append(sz)

		plt.plot(x, y, clr[class_label], label=labels[class_label])
		print("Rows:", dfC.shape[0], '\n')
		print(pd.Series(labels_i).value_counts())
		#return labels
	ax.legend(loc='upper right', shadow=True, ncol=1)
	plt.show()

if __name__ == '__main__':
    if len(sys.argv) == 1:   # dataset not given at command line
        sys.argv.append('Datasets/star_light/train.csv')
        #sys.argv.append('Datasets/star_light/test.csv')
    for dataset in sys.argv[1:]:
        dfC = find_anomalies_mean_L1(dataset)