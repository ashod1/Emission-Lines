# Emission-Lines
Predicting emission lines of bright galaxies from photometric data. I can't show any results or figures beceause DESI data is not public yet.

I am first using local linear regression to predict various emission lines (H_alpha, OII, etc.) from photometric fluxes like SDSS-G, SDSS-R, etc. To test predictions I'm using cross-validation with a 90-10 split.

I am also trying to use my own calculated fluxes from the spectral energy distribution of the galaxy to see if it's any better.

I also use some ML methods, like random forest, xgboost, and neural networks, to predict the emission lines. 
