--- AHEM ---
These are my personal notes from the data reduction. I am keeping them with the code so I don't lose them. That said, you are welcome to read. Just don't expect them to make sense. 

# bookmark
was going to implement a model-predicted error using the fluxfac and bkgnd values now stored in aggregate

# 2023-03-01
## red wing core / normalization issues
having a look at how much the normalized line shapes vary from their average


# 2023-02-28
## red wing core / normalization issues
looked to see if there are good g140m datasets of WDs or OAF stars and didn't find any (forgot to look for B stars)
checking to see if width of geocorna tracks with flux of red wing in gj 436 data
hmm in x2d images the image is not fully rectified. odd. I'll start from tag file and rectify. 
a downside to this is I won't have hot/dead pixel flags
I wanted to fit a gaussian with an offset, but then realized this doesn't make sense because the gaussian will be flat topped given that geocorona is an image of the slit. instead I will measure width from guassian kernel density estimate I guess.
Huh, in the tag file the lya trace is not tilted by the same angle as the geocorona. It looks flat. 
Actually after rotating the lya trace looks just as straight if not straighter, so I think it works.

Result: there is no clear correlation. In retrospect, this makes sense. The focus on the secondary onto the slit will not change the width of the slit image on the detector.

The only other way correlation I think of would be some sort of temperature measurement on the spacecraft. 
I tried correlating lya fluxes (normed by each epoch of data) with every temperature sensor on HST. Some are promising. Given that there are 55 sensors, a correlation has to be very strong for it to be significant. The one I thought was most promising was "Thermal controller zone 1B temp", the otcz1bt key in the _spt file header. A spearman r test gives a p value of 0.013, which degrades to 0.5 when considering 57 trials. However, I opened a support ticket to ask if there are known ways to correct for slit breathing and find out if this sensor or others are near the secondary optical assembly. 

# 2023-02-27
## crappy spectra
pulled the latest reference files and re-reduced the spectra. didn't change anything at all.
aha the problem is that even with the extraction location set and a search size of 10 pixels, x1dcorr still sometimes misses the spectral trace. I tried reducing maxsearch to 2 pixels, but then half of the extractions failed. So instead I just set it to zero and now it appears that it doesn't even try searching. This actually added in a new spectrum which must have previously failed. Also, it appears there were really four spectra that were bad previously
ocyh03020-1 ocyh03040-1 ocyh03040-3 ocyh01040-3 (indexing from 1)
the one that was added in is ocyh01040-4 (the very last exposure of them all)

this one changed somewhat significantly
ocyh02030-3 which begins at 2457485.4974919376 jd

Now I think the spectra are all good. 

## better error estimates
changed the chi2 function to use simulated errors. 
in spot checked spectra, simulated errors look good
in my example, this changed the chi2 prediction from 4404.931874612632 to 6446.559233658321. Seems reasonable. 
Ethan should also add an error hyperparameter since STIS pipeline way overinflates the errors
I verified errors of subexposures are still inflated even after re-reducing the data, so I should stick with the poisson errors. 

## red wing core / normalization issues

# 2023-02-23
## crappy spectra
paged through the spectra in the aggregated_data file and the bad ones are 71, 97, 101
when I generated a new file, it reordered the table. now the bad ones are at 8, 15, 86
exposure times are all similar, so that is not the cause
The bad spectra are from
'ocyh01040', 'ocyh03040', 'ocyh03020' 
These are all either the first or the last subexposure of a set. Seems like it must be an instrument/reduction issue. 
First step should probably be to rereduce the spectra with the latest reference files and pipeline version.
Mmm but stistools isn't working for me. I think I need to download command line tools and then reinstall stistools.
Huh, all of the bad data is from the ocyh program. I'll check for observing problems. These are all exposures where the pipeline missed the spectral trace, but there are other exposures that also missed the trace that don't seem to exhibit this problem. 
Perhaps I should make lightcurves of the full line flux for all exposures and look for oddities. PLus flares, actually.
I could also check if airglow is lower. 
Huh, looking at the x2d for 01040, all subexposures look roughly the same. Airglow is definitely not lower in the last one. The trace is at 489 in that sub. For the first three it is at 489 also. And the Lya flux in the last x2d seems to have the brightest Lya pixel of all. There must be some kind of pipeline issue. 

# suspiciously low errors near geocorona
spot checking the errors, they seem fine. looking at sqrt(f)/e shows a dip where there is a bunch of "extra" error and there are prominent dips where the airglow should be for the spectra I checked. 


## ideas on better slit breathing correction
in general paging through spectra has me wondering if normalizing to the red wing is really the best way to account for slit breathing. 
I wonder if we could track changes in the airglow to correlate with slit breathing? Flux changes wouldn't work since that changes naturally, but centroid and width might vary in step with the slit breathing. 
Another option is not to normalize every subexposure but instead normalize them by exposure. 
Yet another is to correlate red wing fluxes with the local solar time of the telescope, if that meta info exists. 
Or could maybe use airglow intensity by looking for deviations from smooth curve... except variations due to slit breathing are probably also smooth so never mind. 

## pixel shift
for some reason the data in the aggregated file is shifted by one pixel in wavelength
ohhh its because I applied the rv shift of GJ 436. all is well. 

# 2021-12-18
I guess I need to figure out again how to work with the STIS LSF. The problem is that it does not subdivide the pixels evenly. That makes going from wavelength to pixel/lsf dimensions hard. I suppose for each pixel I could figure out the wavelength points of the lsf, then interpolate the model at those points and sum? But that isn't really how it works. Rather it is that I need to add that flux to the pixels. You know... what I should do is simply interpolate the stis lsf onto an even grid. then that will make this straightforward.

# 2021-12-16
masking points where bkg/net ratio is > 2 looks good to me
TODO: I need to check all the subexposures against a coadd to make sure none look totally wacky

I got briefly worried about coadding spectra from different epochs to make a master coadd because of intrinsic variability, but then I realised that with the slit breathing it doesn't matter anyhow bc I still have to normalize everything. 

I considered trying to mask the central pixels that are airglow contaminated in the coadd, but decided not too because it isn't like some x1ds aren't filled there and others are. They are all basically the same. I think the coadd is as good as it is worth getting. 

# 2021-12-15
need to use inttag with increment keyword to subdivide exposures

I should make a "deconvolved" profile from the out of transit data to which we can apply the transit depths. still not too sure how to do that. might just be easiest to fit the Lya profile

the wavelength grids vary between the observations, so each convolution really will be its own thing and I should use a for loop
Because each convolution will be paired with a specific observation, ethan will need to know the time of each observation relative to mid transit
there is probably some legitimate wiggle room in terms of the transit timing and the wavelength solution of the data, but let's not worry about that. I should shift the data by the stellar rv though.
gotta remember to normalize by an unabsorbed region and also tell #evan that this means he should expect lower chi2 values otherwise 

need to make a coadd from the out of transit spectra

so process will be 
- interpolate the absorption profile onto the same grid as the oot profile
- multiply the oot profile by the predicted transit absorption
- pick a region where no absorption is present, normalize the flux in that region to the oot profile
- regrid to match the data grid, oversampled for the lsf if necessary
- convolve with the lsf
- bin to the data grid
- compare with the data, masking airglow region (region where bkg counts are above some threshold?)