import numpy as np
from downsample import *
import sys
sys.path.append('../')    # import from folder one directory out
import util
import pandas as pd
import data_handler
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=12)


df = util.csv_to_df('../Datasets/star_light/train.csv')
reductions = np.linspace(1, 200, 100) 
shifting = [0, 10, 50, 100, 200]



KL = np.zeros([len(shifting),len(reductions)])  
shift_count = 0
for start in shifting:

    KL[shift_count,:] = get_mean(df, reductions, run_down_sample_fft, 0, start)
    shift_count += 1

acc_mean = np.mean(KL, axis=0)
acc_std = np.std(KL, axis=0)
acc_max = np.max(KL, axis=0)
acc_min = np.min(KL, axis=0)

plt.plot(reductions, acc_mean+acc_std, 'r-', linewidth=0.5)
plt.plot(reductions, acc_mean-acc_std, 'r-', linewidth=0.5)
# plt.plot(reductions, acc_max, 'k-')
# plt.plot(reductions, acc_min, 'k-')
plt.fill_between(reductions, acc_mean-acc_std, acc_mean+acc_std, alpha=0.3)
plt.plot(reductions, acc_mean, 'g-', linewidth=2)
# plt.plot(reductions, current_mean, label =str(single_bin)+' Bins')


# plt.plot(reductions, get_mean(df, reductions, run_down_sample_fft, 0, shifting), label='fourier transform')
# plt.legend(loc=2)
plt.xlabel('Reduction Size', fontsize=18)
plt.ylabel('KL divergance', fontsize=18)
plt.show()