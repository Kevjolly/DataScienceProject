#!/usr/bin/env python
import numpy as np
from astropy.stats import LombScargle

stars = ['cepheid','eclipse_binary','RR_lyrae']
file_count = [4631, 6138, 2475]

for star, num_files in zip(stars, file_count):
    i = 0
    skip_count = 0
    while i+skip_count+1 < num_files:
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
        if np.max(power)/np.mean(power) < 40:
            skip_count+=1
            continue
        
        period_days = 1/frequency
        best_period = period_days[np.argmax(power)]
        fold = 1
        phase = (time / best_period) % fold
        phase = (phase + shift - phase[np.argmax(mags)]) % fold
        phase/=fold
        order = phase.argsort()
        phase = phase[order]
        mags = mags[order]
        mag_ers = mag_ers[order]
        with open('phased_{0}/ogle{1}.txt'.format(star,i+1), 'w') as f:
            for k in range(len(phase)):
                line = str(phase[k])+' '+str(mags[k])+' '+str(mag_ers[k])
                f.write(line+'\n')
        i+=1
