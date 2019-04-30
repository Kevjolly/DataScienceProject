# Downsampling Data
Analyse the effect that downsampling the data has. Mainly looking at information content in the data.

Downsample by periodic re sampling.

## Density approximation of the data
First tried simple estimation by histogram (more info on this?). No obvious bin size to use and different bin size has a big effect on the estimated distribution.
Data is too high dimentionality for kernel density estimation
Used normalised Fourier transform coeffiecients as density estimation. Lends naturally to the time series data. Pad with zeros HOC or ignore?

## Measuring Information Loss
Used KL divergence between the orginal data and the downsampled data.  This measures the number of ‘missing’ bits of information, in other words, how similar they are to each other. 

## Other ways of measuring information loss
Also look at how the mean and standard deviations change with downsampling.

Measure the effect of performance (classification) that downsampling occurs
