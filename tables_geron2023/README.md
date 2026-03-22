# Description tables in tables_geron2023/

This directory contains the contents of Tables 1 and 3 from [Géron et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.521.1775G/abstract). A detailed description of every column is found below.

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
