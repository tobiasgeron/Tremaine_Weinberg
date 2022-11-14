# Tremaine_Weinberg

### Summary

This reposity contains everything to perform the Tremaine-Weinberg method on MaNGA galaxies in Python.
More info on the Tremaine-Weinberg method and some papers that use it (not with this package, though):  
Tremaine, Weinberg (1984): https://ui.adsabs.harvard.edu/abs/1984ApJ...282L...5T/abstract  
Aguerri et al. (2015): https://ui.adsabs.harvard.edu/abs/2015A%26A...576A.102A/abstract  
Cuomo et al. (2019): https://ui.adsabs.harvard.edu/abs/2019A%26A...632A..51C/abstract  
Garma-Oehmichen et al. (2020): https://ui.adsabs.harvard.edu/abs/2020MNRAS.491.3655G/abstract  
Géron et al. (2022): in prep.   

This package was developed for Géron et al. (2022, in prep) and is used there to determine the bar pattern speeds for a large sample of strongly and weakly barred galaxies, using data from MaNGA. The code can be found in Tremaine_Weinberg.py. An example of how to use the code can be found in Example.ipynb. Table 1 and Table 3 from Géron et al. (2022) are found in tables_geron2022/ directory. 

This code has been tested with python 3.7.9, numpy 1.19.2, matplotlib 3.3.2, [photutils](https://photutils.readthedocs.io/en/stable/  ) 1.2.0, [marvin](https://sdss-marvin.readthedocs.io/en/latest/) 2.7.1, scipy, 1.6.2, csv 1.0.   

### Description of code

Warning: This code is still under development.

The main function is: 

`Tremaine_Weinberg(PA, inc, barlen, PA_bar, maps, PA_err = 0.0, inc_err = 0.0, barlen_err = 0.0, PA_bar_err = 0.0, slit_width = 1, slit_separation = 0, slit_length_method = 'default', slit_length = np.inf, min_slit_length = 12, n_iter = 0, cosmo = [], redshift = np.nan, aperture_integration_method = 'center', forbidden_labels = ['DONOTUSE','UNRELIABLE','NOCOV'], deproject_bar = True, correct_velcurve = True, velcurve_aper_width = 5)`

INPUTS:  
`PA (float)`: Position angle of galaxy, in degrees. The PAs are defined as counterclockwise from 3 o'clock on the DECaLS pictures. 
`inc (float)`: Inclination of galaxy, in degrees.  
`barlen (float)`: Length of the entire bar of the galaxy, in arcsec (so not bar radius, but bar diameter!).  
`PA_bar (float)`: Position angle of the bar, in degrees.  
`maps (MaNGA Maps)`: A MaNGA maps object. If you have a MaNGA cube, can get the maps object by doing: Maps(plateifu = plateifu, bintype='VOR10'). See: https://sdss-marvin.readthedocs.io/en/latest/tools/maps.html. 

OPTIONAL INPUTS:  
`PA_err (float)`: Error on the galaxy PA, in degrees.  
`inc_err (float)`: Error on the inclination of the galaxy, in degrees.  
`barlen_err (float)`: Error on the length of the bar, in arcsec (error on the entire bar, not bar radius!).  
`PA_bar_err (float)`: Error on the PA of the bar, in degrees.  
`slit_width (float)`: Width of each slit, in arcsec.  
`slit_separation (float)`: Separation between slits, in arcsec.  
`slit_length_method (str)`: Either 'default' or 'user_defined'. Decides which method to use to determine the slit lengths.  
`slit_length (float)`: Maximum value of slit length, in arcsec. If slit_length_method == 'user_defined', this is the slit length that will be used.  
`min_slit_length (float)`: Minimum value of slit length, in arcsec. If slit is shorter than that, ignore slit.  
`n_iter (int)`: Amount of iterations used to determine the posterior distributions of Omega, Rcr and R. Default is 0. Recommended value for accurate posteriors and errors is 1000.  
`cosmo (astropy cosmology)`: An astropy cosmology (e.g.: FlatLambdaCDM(H0=70 km / (Mpc s), Om0=0.3, Tcmb0=2.725 K, Neff=3.04, m_nu=[0. 0. 0.] eV, Ob0=None)). Used together with 'redshift' to convert arcsec to kpc.  
`redshift (float)`: Redshift of the target. Used together with 'cosmo' to convert arcsec to kpc.   
`aperture_integration_method (bool)`: The integration method used with the apertures. Can be either 'center' or 'exact'.  
`forbidden_labels (list)`: List of possible labels in the MaNGA datacube. Will ignore spaxels that are associated with any of these labels. 
`deproject_bar (bool)`: Whether to deproject the bar using the PA, PA_bar and inclination of the galaxy. Strongly advised to always keep on True.  
`correct_velcurve (bool)`: Whether to correct the velocity and positions for the inclination and PA of the galaxy while determining the velocity curve. 
`velcurve_aper_width (int)`: How many pixels to use to determine the velocity curve.  

OUTPUTS
Returns TW class, defined in TremaineWeinberg.py. The TW class contains everything that is calculated. See Example.ipynb to see how to access it.



TODO: Write more documentation.
