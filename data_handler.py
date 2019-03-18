import numpy as np
import matplotlib.pyplot as plt
from pandas import *
#todo: generalise the class distribution from unique()

test = read_csv('Datasets/star_light/test.csv', header=None)
train = read_csv('Datasets/star_light/train.csv', header=None)

my_class = train

class1 = my_class.loc[my_class[0]==1].iloc[:,1:]
class2 = my_class.loc[my_class[0]==2].iloc[:,1:]
class3 = my_class.loc[my_class[0]==3].iloc[:,1:]

mean1 = class1.mean(0)
mean2 = class2.mean(0)
mean3 = class3.mean(0)

std1 = class1.std(0)
std2 = class2.std(0)
std3 = class3.std(0)

max_lim = my_class.max().max()
min_lim = my_class.min().min()

fig = plt.figure()
ax = fig.add_subplot(311)
ax.set_ylim(min_lim,max_lim)
plt.plot(np.linspace(0,1,len(mean1)),mean1+std1, 'r-')
plt.plot(np.linspace(0,1,len(mean1)),mean1-std1, 'r-')
ax.fill_between(np.linspace(0,1,len(mean1)), mean1-std1, mean1+std1, alpha=0.3)
plt.plot(np.linspace(0,1,len(mean1)),mean1)

ax = fig.add_subplot(312)
ax.set_ylim(min_lim,max_lim)
plt.plot(np.linspace(0,1,len(mean2)),mean2+std2, 'r-')
plt.plot(np.linspace(0,1,len(mean2)),mean2-std2, 'r-')
ax.fill_between(np.linspace(0,1,len(mean2)), mean2-std2, mean2+std2, alpha=0.3)
plt.plot(np.linspace(0,1,len(mean2)),mean2)

#plt.figure()
ax = fig.add_subplot(313)
ax.set_ylim(min_lim,max_lim)
plt.plot(np.linspace(0,1,len(mean3)),mean3+std3, 'r-')
plt.plot(np.linspace(0,1,len(mean3)),mean3-std3, 'r-')
ax.fill_between(np.linspace(0,1,len(mean3)), mean3-std3, mean3+std3, alpha=0.3)
plt.plot(np.linspace(0,1,len(mean3)),mean3)

plt.show()
