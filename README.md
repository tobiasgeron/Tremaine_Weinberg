# Tremaine_Weinberg

[![DOI](https://zenodo.org/badge/493622015.svg)](https://zenodo.org/badge/latestdoi/493622015)


Please cite [Géron et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.521.1775G/abstract) when using this code.

Also, check out [this interactive tool](https://tobiasgeron.github.io/bar_kinematics.html) to see the pattern speed, corotation radius and curly R visualised on a barred galaxy and understand how these parameters interact with each other.  

Update 2025-09-03: This code now works for any IFU, not just MaNGA. See Example.ipynb.

### Summary

This repository contains everything to perform the Tremaine-Weinberg method on MaNGA galaxies in Python.
More info on the Tremaine-Weinberg method and some papers that use it (not with this package, though):  
Tremaine, Weinberg (1984): https://ui.adsabs.harvard.edu/abs/1984ApJ...282L...5T/abstract  
Aguerri et al. (2015): https://ui.adsabs.harvard.edu/abs/2015A%26A...576A.102A/abstract  
Cuomo et al. (2019): https://ui.adsabs.harvard.edu/abs/2019A%26A...632A..51C/abstract  
Garma-Oehmichen et al. (2020): https://ui.adsabs.harvard.edu/abs/2020MNRAS.491.3655G/abstract  
Géron et al. (2023): https://ui.adsabs.harvard.edu/abs/2023MNRAS.521.1775G/abstract   

This package was developed for Géron et al. (2023) and is used there to determine the bar pattern speeds for a large sample of strongly and weakly barred galaxies, using data from MaNGA. The code can be found in TremaineWeinberg.py. An example of how to use the code can be found in Example.ipynb. Table 1 and Table 3 from Géron et al. (2023) are found in tables_geron2023/ directory. 

This code has been tested with python 3.12.0, numpy 1.26.0, matplotlib 3.8.1, [photutils](https://photutils.readthedocs.io/en/stable/  ) 1.9.0, scipy, 1.15.3, csv 1.0.   

### Description of code

Warning: This code is still under development.

The simplest way to use the code is:
```
from TremaineWeinberg import Tremaine_Weinberg

tw = Tremaine_Weinberg(velocity, flux, position, PA, inc, barlen, PA_bar)

print(f'The bar pattern speed is {tw.Omega} km s-1 arcsec-1.')
```

All the possible arguments are found here:

`Tremaine_Weinberg(velocity, flux, position, PA, inc, barlen, PA_bar, velocity_err = None, flux_err = None, PA_err = 0.0, inc_err = 0.0, barlen_err = 0.0, PA_bar_err = 0.0, mask = None, slit_width = 1, slit_separation = 0, slit_length_method = 'default', slit_length = np.inf, min_slit_length = 5, n_iter = 0, cosmo = [], redshift = np.nan, aperture_integration_method = 'center', deproject_bar = True, correct_velcurve = True, velcurve_aper_width = 5, check_convergence = False, convergence_n = 2, convergence_threshold = 5, convergence_stepsize = 0, pixscale = None, name = '')`

INPUTS:  
`velocity` (np.array): Numpy array of the velocity field (gas or stars). Assumed to be in km/s. Assumed to be in NWSE (i.e. north up, west right) orientation.  
`flux` (np.array): Numpy array of the flux (gas or stars). Assumed to be in NWSE (i.e. north up, west right) orientation.  
`position` (np.array): Numpy array of the distance to the centre of the galaxy. Assumed to be in arcsec.  
`PA` (float): Position angle of galaxy, in degrees. The PAs are defined as East of North.  
`inc` (float): Inclination of galaxy, in degrees.  
`barlen` (float): Length of the entire bar of the galaxy, in arcsec (so not bar radius, but bar diameter!).  
`PA_bar` (float): Position angle of the bar, in degrees.  

OPTIONAL INPUTS:  
`velocity_err` (np.array): Error on the velocity field.  
`flux_err` (np.array): Error on the flux.   
`PA_err` (float): Error on the galaxy PA, in degrees.  
`inc_err` (float): Error on the inclination of the galaxy, in degrees.  
`barlen_err` (float): Error on the length of the bar, in arcsec (error on the entire bar, not bar radius!).  
`PA_bar_err` (float): Error on the PA of the bar, in degrees.  
`slit_width` (float): Width of each slit, in arcsec.  
`slit_separation` (float): Separation between slits, in arcsec.  
`slit_length_method` (str): Either 'default' or 'user_defined'. Decides which method to use to determine the slit lengths.  
`slit_length` (float): Maximum value of slit length, in arcsec. If slit_length_method == 'user_defined', this is the slit length that will be used.  
`min_slit_length` (float): Minimum value of slit length, in arcsec. If slit is shorter than that, ignore slit.  
`n_iter` (int): Amount of iterations used to determine the posterior distributions of Omega, Rcr and R. Default is 0. Recommended value for accurate posteriors and errors is 1000.  
`cosmo` (astropy cosmology): An astropy cosmology (e.g.: FlatLambdaCDM(H0=70 km / (Mpc s), Om0=0.3, Tcmb0=2.725 K, Neff=3.04, m_nu=[0. 0. 0.] eV, Ob0=None)). Used together with 'redshift' to convert arcsec to kpc.  
`redshift` (float): Redshift of the target. Used together with 'cosmo' to convert arcsec to kpc.   
`aperture_integration_method` (bool): The integration method used with the apertures. Can be either 'center' or 'exact'. Recommended to use 'exact' if PA is close to 0, 90, or 180 degrees.  
`deproject_bar` (bool): Whether to deproject the bar using the PA, PA_bar and inclination of the galaxy. Strongly advised to always keep on True.  
`correct_velcurve` (bool): Whether to correct the velocity and positions for the inclination and PA of the galaxy while determining the velocity curve.  
`velcurve_aper_width` (int): How many pixels to use to determine the velocity curve.  
`pixscale` (float): Pixel scale of the image. It is good to specify. If not, we try to calculate it.     
`name` (str): A name for the object. E.g. an ID, or plate-ifu number.   

OUTPUTS:  
Returns TW class, defined in TremaineWeinberg.py. The TW class contains everything that is calculated. See Example.ipynb to see how to access it.  

NOTES:  
Currently, if n_iter = 0, it will run once with best-guess inputs. If n_iter > 0, it will run the iterations, Omega, Rcr and R are the median of all the iterations. After the iterations, it will run one last time with the best-guess inputs for all the figures etc. All PAs are defined as East of North.  
    

TODO: Write more documentation.  



# Description tables in tables_geron2023/

This directory contains the contents of Tables 1 and 3 from Géron et al. (2023). A detailed description of every column is found below.

### Table 1

`iauname`: The NSA-iauname of this target.  
`dr8_id`: The GZ DESI identifier of this target.  
`MANGAID`: The MaNGA-id of this target.  
`PLATEIFU`: The MaNGA plate-ifu number of this target.  
`RA`: The right ascension of this target. Identical to the OBJRA column in the MaNGA DRPALL.  
`DEC`: The declination of this target. Identical to the OBJDEC column in the MaNGA DRPALL.  
`inc`: The inclination of the galaxy, measured in degrees.  
`inc_err`: The error on the inclination.  
`PA`: The (kinematic) position angle of the galaxy. Measured east of north, in degrees, between 0 and 180.  
`PA_err`: The error on the galaxy PA.  
`PA_bar`: The position angle of the bar. Measured east of north, in degrees, between 0 and 180.  
`PA_bar_err`: The error on the PA of the bar.  
`R_bar`: The bar radius, measured in arcsec.  
`R_bar_err`: The error on the bar radius.  
`R_bar_deproj_kpc`: The deprojected bar radius, measured in kpc.  
`R_bar_deproj_kpc_err`: The error on the deprojected bar radius.  
`redshift`: The redshift of the target. Taken from the NSA catalog.  
`nsa_elpetro_absmag_r` : The NSA absolute r-band magnitude from elliptical Petrosian fluxes, assuming Ωm=0.3, ΩΛ=0.7, h=0.7, taken from the MaNGA DRPALL.  
`nsa_sersic_absmag_r` : The NSA absolute r-band magnitude, assuming Ωm=0.3, ΩΛ=0.7, h=0.7, taken from the MaNGA DRPALL.    
`bar_type`: The bar type according to GZ DESI. Either 'Weak bar' or 'Strong bar'.  
`is_SF`: A flag whether the galaxy is star-forming (True) or quiescent (False), according to the relationship quoted in Belfiore+2018.

The details of how the bar type, inclination, position angles, bar length and their errors are measured can be found in Section 3 of Géron et al. (2023).



### Table 3
`iauname`: The NSA-iauname of this target.  
`dr8_id`: The GZ DESI identifier of this target.  
`MANGAID`: The MaNGA-id of this target.  
`PLATEIFU`: The MaNGA plate-ifu number of this target.  
`Omega`: The pattern speed of the target, measured in km s-1 arcsec-1.  
`Omega_ll`: The lower limit of the pattern speed.  
`Omega_ul`: The upper limit of the pattern speed.  
`Omega_phys`: The pattern speed of the target, measured in km s-1 kpc-1.  
`Omega_phys_ll`: The lower limit of the pattern speed.  
`Omega_phys_ul`: The upper limit of the pattern speed.  
`Rcr`: The corotation radius of the target, measured in arcsec.  
`Rcr_ll`: The lower limit of the corotation radius.  
`Rcr_ul`: The upper limit of the corotation radius.  
`Rcr_phys`: The corotation radius of the target, measured in kpc.  
`Rcr_phys_ll`: The lower limit of the corotation radius.  
`Rcr_phys_ul`: The upper limit of the corotation radius.  
`R`: The ratio R (=Rcr / Rbar) of the target. Dimensionless.  
`R_ll`: The lower limit of R.  
`R_ul`: The upper limit of R.  

To see how all these variables (and their errors) were measured, please refer to Géron et al. (2023).
