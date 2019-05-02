#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import LombScargle
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif',size=12)

stars = ['cepheid','eclipse_binary','RR_lyrae']
formal = ['Cepheid', 'Eclipse Binary', 'RR Lyrae']
star_names = {star:form for star, form in zip(stars,formal)}
star_count = 1
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
                time[j], mags[j], mag_ers[j] = line.split(' ')
        if np.mean(mag_ers)/np.ptp(mags) > 0.05: 
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

        if np.max(power)/np.mean(power) < 40:
            skip_count+=1
            continue
        #print(np.max(power)/np.mean(power))
        period_days = 1/frequency
        best_period = period_days[np.argmax(power)]
        fold = 1
        #if star == 'eclipse_binary': fold = 2 #since full period is double than expected
        phase = (time / best_period) % fold
        phase = (phase + shift - phase[np.argmax(mags)]) % fold
        phase/=fold
        i+=1
        fig,axs = plt.subplots(3,1,figsize=(6,7))
        for k in range(star_count):
            axs[0].errorbar(time,mags,mag_ers,fmt='.k',ecolor='gray',capsize=0)
            axs[2].errorbar(phase,mags,mag_ers,fmt='.k',ecolor='gray',capsize=0)
        axs[0].set_title(star_names[star])
        axs[0].set_xlabel('Heliocentric Julian Day - 2450000')
        axs[0].set_ylabel('Brightness (Magnitude)')
        axs[0].invert_yaxis()
        axs[1].plot(period_days,power,'k')
        axs[1].scatter([period_days[np.argmax(power)]],[max(power)],color='r')
        axs[1].set_xlabel('Period (Days)')
        axs[1].set_ylabel('Likelihood')
        axs[2].set_xlabel('Phase')
        axs[2].set_ylabel('Brightness (Magnitude)')
        axs[2].invert_yaxis()
        plt.tight_layout()
        plt.savefig('eg_phase_{0}.jpg'.format(star),dpi=500)
#plt.show()


