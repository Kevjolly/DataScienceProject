#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import LombScargle
#from matplotlib import rc
#rc('text', usetex=True)
#rc('font', family='serif',size=12)

stars = ['cepheid','eclipse_binary','RR_lyrae']
formal = ['Cepheid', 'Eclipse Binary', 'RR Lyrae']
star_names = {star:form for star, form in zip(stars,formal)}
star_count = 100
phase_dict = {}
mags_dict = {}
mag_ers_dict = {}
time_dict = {}

for star in stars:
    i = 0
    skip_count = 0
    while i < star_count:
        with open('{0}/ogle{1}.txt'.format(star,i+1+skip_count)) as f:
            length = sum(1 for line in f)
            time = np.zeros(length)
            mags = np.zeros(length)
            mag_ers = np.zeros(length)
            f.seek(0)
            for j,line in enumerate(f):
                time[j], mags[j], mag_ers[j] = line.split(' ')[-3:]
        if np.mean(mag_ers)/np.ptp(mags) > 0.1: 
            skip_count+=1
            continue #remove unreliable data
        ls = LombScargle(time, mags, mag_ers)
        if star == 'eclipse_binary':
            frequency, power = ls.autopower(nyquist_factor=800,
                minimum_frequency=0.2, maximum_frequency=40)
            frequency/=2
            shift = 0.25
        else:
            frequency, power = ls.autopower(nyquist_factor=800,
                minimum_frequency=0.1, maximum_frequency=20)
            shift = 0.4
        if np.max(power)/np.mean(power) < 50:
            skip_count+=1
            continue
        #print(np.max(power)/np.mean(power))

        period_days = 1/frequency
        best_period = period_days[np.argmax(power)]
        fold = 1
        phase = (time / best_period) % fold
        phase = (phase + shift - phase[np.argmax(mags)]) % fold
        phase/=fold
        #regularise
        mag_av = np.mean(mags)
        mags-=mag_av
        phase_dict[i] = phase
        mags_dict[i] = mags
        mag_ers_dict[i] = mag_ers
        time_dict[i] = time
        i+=1

    fig,axs = plt.subplots(2,1,figsize=(6,7))
    bins = np.linspace(0, 1, 100)
    all_phase = [*phase_dict.values()]
    all_phase = np.array([item for sublist in all_phase for item in sublist])
    all_mags = [*mags_dict.values()]
    all_mags = np.array([item for sublist in all_mags for item in sublist])
    all_mag_ers = [*mag_ers_dict.values()]
    all_mag_ers = np.array([item for sublist in all_mag_ers for item in sublist])
    bin_phase = np.digitize(all_phase, bins)
    mag_means = [all_mags[bin_phase == i].mean() for i in range(1, len(bins))]
    mag_er_means = [all_mag_ers[bin_phase == i].mean() for i in range(1, len(bins))]
    [axs[0].errorbar(phase_dict[i],mags_dict[i],mag_ers_dict[i],fmt='.k',ecolor='gray',capsize=0)
        for i in range(star_count)]
    axs[1].errorbar(bins[1:], mag_means, mag_er_means,fmt='.k',ecolor='gray',capsize=0)
    axs[0].invert_yaxis()
    axs[1].invert_yaxis()
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


