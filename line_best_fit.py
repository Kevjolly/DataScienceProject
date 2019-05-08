#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import LombScargle
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif',size=12)

stars = ['cepheid','eclipse_binary','RR_lyrae']
stars = ['cepheid']
formal = ['Cepheid', 'Eclipse Binary', 'RR Lyrae']
formal = ['RR Lyrae']
star_names = {star:form for star, form in zip(stars,formal)}
file_counts = [4586, 3633, 2172]
file_counts = [1000]
file_counts = [10]
#file_counts = [10, 10, 10]
phase_dict = {}
mags_dict = {}
mag_ers_dict = {}
phase_dict = {}

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

for count, star in enumerate(stars):
    num_files = file_counts[count]
    for i in range(num_files-1):
        with open('phased_{0}/ogle{1}.txt'.format(star,i+100)) as f:
            length = sum(1 for line in f)
            phase = np.zeros(length)
            mags = np.zeros(length)
            mag_ers = np.zeros(length)
            f.seek(0)
            for j,line in enumerate(f):
                phase[j], mags[j], mag_ers[j] = line.split(' ')[-3:]
        
        mag_av = np.mean(mags)
        mags-=mag_av
        mag_smooth = smooth(mags, 100)
        grad = smooth(np.diff(mag_smooth), 200)
        print(phase[np.argmin(grad)])
        #mid_loc = phase[np.argmin(grad)]
        #phase = (phase - mid_loc + 0.5)%1
        fig, ax = plt.subplots(2,1)
        ax[0].errorbar(phase,mags,mag_ers,fmt='.k',ecolor='gray',capsize=0)
        ax[0].set_xlabel('Phase')
        ax[0].set_ylabel('Normalised Magnitude')
        ax[0].set_title('Star: {0}'.format(formal[count]))
        ax[0].invert_yaxis()
        ax[0].plot(phase, mag_smooth, linewidth=5)
        ax[1].plot(phase[1:],grad)
        ax[1].invert_yaxis()
        plt.tight_layout()
#for i in range(star_count):
    #axs[0].errorbar(phase_dict[i],mags_dict[i],mag_ers_dict[i],fmt='.k',ecolor='gray',capsize=0)
    #axs[1].errorbar(phase_dict[i],mags_dict[i],mag_ers_dict[i],fmt='.k',ecolor='gray',capsize=0)
#axs[0].set_title(star_names[star])
#axs[0].set_xlabel('Heliocentric Julian Day - 2450000')
#axs[1].set_xlabel('Phase')
#axs[0].invert_yaxis()
#axs[1].invert_yaxis()
#axs[2].plot(frequency,power)
#axs[2].set_xlabel('Frequency (Days$^{-1}$)')
plt.show()


