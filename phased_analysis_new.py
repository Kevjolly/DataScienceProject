#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import LombScargle
from matplotlib import rc
import sys

rc('text', usetex=True)
rc('font', family='serif',size=12)

sys.stdout = open('phased2_means.txt', 'w')

stars = ['cepheid','eclipse_binary','RR_lyrae']
#stars = ['RR_lyrae']
formal = ['Cepheid', 'Eclipse Binary', 'RR Lyrae']
#formal = ['RR Lyrae']
star_names = {star:form for star, form in zip(stars,formal)}
file_counts = [4586, 3633, 2172]
#file_counts = [27]

for count, star in enumerate(stars):
    num_files = file_counts[count]
    mags_dict = {}
    mag_ers_dict = {}
    time_dict = {}
    for i in range(num_files-1):
        if i%100==0:print(star,i)
        with open('phased_{0}/ogle{1}.txt'.format(star,i+1)) as f:
            length = sum(1 for line in f)
            time = np.zeros(length)
            mags = np.zeros(length)
            mag_ers = np.zeros(length)
            f.seek(0)
            for j,line in enumerate(f):
                time[j], mags[j], mag_ers[j] = line.split(' ')[-3:]
        
        mags_dict[i] = mags
        mag_ers_dict[i] = mag_ers
        time_dict[i] = time

    fig,ax = plt.subplots()
    bins = np.linspace(0, 1, 101)
    all_time = [*time_dict.values()]
    all_time = np.array([item for sublist in all_time for item in sublist])
    all_mags = [*mags_dict.values()]
    all_mags = np.array([item for sublist in all_mags for item in sublist])
    all_mag_ers = [*mag_ers_dict.values()]
    all_mag_ers = np.array([item for sublist in all_mag_ers for item in sublist])
    bin_phase = np.digitize(all_time, bins)
    mag_means = [all_mags[bin_phase == i].mean() for i in range(1, len(bins))]
    print(mag_means)
    mag_er_means = [all_mag_ers[bin_phase == i].mean() for i in range(1, len(bins))]
    ax.errorbar(bins[1:], mag_means, mag_er_means,fmt='.k',ecolor='gray',capsize=0)
    ax.set_xlabel('Phase')
    ax.set_ylabel('Normalised Magnitude')
    ax.set_title('Star: {0}'.format(formal[count]))
    ax.invert_yaxis()
    plt.tight_layout()
#for i in range(star_count):
    #axs[0].errorbar(time_dict[i],mags_dict[i],mag_ers_dict[i],fmt='.k',ecolor='gray',capsize=0)
    #axs[1].errorbar(phase_dict[i],mags_dict[i],mag_ers_dict[i],fmt='.k',ecolor='gray',capsize=0)
#axs[0].set_title(star_names[star])
#axs[0].set_xlabel('Heliocentric Julian Day - 2450000')
#axs[1].set_xlabel('Phase')
#axs[0].invert_yaxis()
#axs[1].invert_yaxis()
#axs[2].plot(frequency,power)
#axs[2].set_xlabel('Frequency (Days$^{-1}$)')
plt.show()


