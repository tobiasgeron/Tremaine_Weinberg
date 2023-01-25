'''
Created December 2021 by Tobias Géron.
Contains everything to perform the Tremaine-Weinberg method on MaNGA galaxies.
More info on the method and papers that use it: 
Tremaine, Weinberg (1984): https://ui.adsabs.harvard.edu/abs/1984ApJ...282L...5T/abstract
Aguerri et al. (2015): https://ui.adsabs.harvard.edu/abs/2015A%26A...576A.102A/abstract
Cuomo et al. (2019): https://ui.adsabs.harvard.edu/abs/2019A%26A...632A..51C/abstract
Guo et al. (2019): https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.1733G
Garma-Oehmichen et al. (2020): https://ui.adsabs.harvard.edu/abs/2020MNRAS.491.3655G/abstract
Géron et al. (2022): in prep. 


TODO: 
Major:

Minor:
Apply Vsys correction after determining centre
Make function to visualise convergence of slits
'''



###############
### Imports ###
###############

import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import csv
import photutils
import marvin
import scipy


############
### Code ###
############

##------------##
## Main class ##
##------------##

class TW:
    '''
    Main class. This is what gets returned to the user after calling the Tremaine_Weinberg() function. 
    Will save all the results and has some methods to plot intermediate results.
    '''


    '''
    TODO: 
    '''
    def __init__(self,PA,inc,barlen,PA_bar,maps, forbidden_labels, PA_err,inc_err,barlen_err,PA_bar_err,slit_width, cosmo, redshift):
        '''
        PA: position angle of galaxy, in degrees.
        inc: inclination of galaxy, in degrees
        barlength: length of the entire bar of the galaxy, in arcsec
        PA_bar: position angle of the bar, in degrees
        maps: a MaNGA maps object. If you have a MaNGA cube, can get the maps object by doing: my_cube.getMaps(bintype='VOR10').

        The PAs are defined as East of North.
        '''
        self.mangaid = maps.mangaid
        self.plateifu = maps.plateifu
        self.slit_width = slit_width

        self.PA = PA #deg
        self.inc = inc #deg
        self.barlen = barlen #arcsec
        self.PA_bar = PA_bar #deg
        self.PA_err = PA_err
        self.inc_err = inc_err
        self.barlen_err = barlen_err
        self.PA_bar_err = PA_bar_err

        self.redshift = redshift
        self.cosmology = cosmo

        # get all relevant MaNGA maps
        self.forbidden_labels = forbidden_labels
        bintype = maps.bintype.name
        if bintype == 'SPX':
            self.stellar_flux = np.flip(maps['spx_mflux'],0)
        elif bintype in ['HYB10','VOR10','BIN']:
            self.stellar_flux = np.flip(maps['bin_mflux'],0)
        self.stellar_vel = np.flip(maps['stellar_vel'],0)
        self.on_sky_x = np.flip(maps['spx_ellcoo_on_sky_x'],0) #arcsec #should be spx_skycoo?
        self.on_sky_y = np.flip(maps['spx_ellcoo_on_sky_y'],0) #arcsec
        self.on_sky_xy = np.sqrt(self.on_sky_x.value**2 + self.on_sky_y.value**2)

        #Vsys correction
        self.stellar_vel, self.Vsys_corr = get_Vsys(self.stellar_vel, self.on_sky_xy, 5, forbidden_labels)

        #initialse lsts we want to track over all MC runs
        self.apers_lst = []
        self.NRMSE_V_curve_lst = []
        self.NRMSE_X_V_lst = []
        self.Omega_lst = []
        self.R_corot_lst = []
        self.R_lst = []
        self.barlen_deproj_lst = []

    def save_intermediate_MC_results(self,apers,Omega,R_corot,R,barlen_deproj,X_V,z,V_curve,V_curve_fit_params):
        self.apers_lst.append(apers)
        self.Omega_lst.append(Omega)
        self.R_corot_lst.append(R_corot)
        self.R_lst.append(R)
        self.barlen_deproj_lst.append(barlen_deproj)

        # Calculate normalised root mean squared error (NRMSE) for the X_V fit and the V_curve fit. 
        if len(apers) > 2: #i.e. there is more than one slit
            self.NRMSE_X_V_lst.append(np.sqrt(z[1][0]/len(X_V[0])) / ( np.max(X_V[1]) - np.min(X_V[1]) )) #https://en.wikipedia.org/wiki/Root-mean-square_deviation
        elif len(apers) == 2: #with two slits, no uncertainty possible
            self.NRMSE_X_V_lst.append(0)
        else: #with one slit, not even a line
            self.NRMSE_X_V_lst.append(np.nan)

        if ~np.isnan(V_curve_fit_params[-1]): # If MSE is np.nan, means the V curve fit failed.
            self.NRMSE_V_curve_lst.append(np.sqrt(V_curve_fit_params[2][0]/len(V_curve[0])) / ( np.max(V_curve[1]) - np.min(V_curve[1]) ))
        else: self.NRMSE_V_curve_lst.append(np.nan)

    def save_results(self,centre, pixscale, LON, slits, apers, aper_Omegas, X_map, X_Sigma, V_Sigma, X_V, z, Omega, Omega_err, R_corot, R_corot_err, R, R_err, V_curve, V_curve_apers, V_curve_fit_params, barlen_deproj):
        self.centre = centre
        self.pixscale = pixscale
        self.LON = LON
        self.slits = slits
        self.apertures = apers
        self.aper_Omegas = aper_Omegas
        self.X_map = X_map
        self.X_Sigma = X_Sigma
        self.V_Sigma = V_Sigma
        self.X_V = X_V
        self.fitted_line = z #direct output of np.polyfit(X,Y,full=True)
        self.Omega = Omega
        self.Omega_err = Omega_err
        self.R_corot = R_corot
        self.R_corot_err = R_corot_err
        self.R = R
        self.R_err = R_err
        self.V_curve = V_curve
        self.V_curve_apers = V_curve_apers
        self.V_curve_fit_params = V_curve_fit_params
        self.barlen_deproj = barlen_deproj


        # Calculate normalised root mean squared error (NRMSE) for the X_V fit and the V_curve fit. 
        if len(apers) > 2: #i.e. there is more than one slit
            self.NRMSE_X_V = np.sqrt(z[1][0]/len(X_V[0])) / ( np.max(X_V[1]) - np.min(X_V[1]) ) #https://en.wikipedia.org/wiki/Root-mean-square_deviation
        elif len(apers) == 2: #with two slits, no uncertainty possible
            self.NRMSE_X_V = 0
        else: #with one slit, not even a line
            self.NRMSE_X_V = np.nan

        if ~np.isnan(V_curve_fit_params[-1]):# If MSE is np.nan, means the V curve fit failed.
            self.NRMSE_V_curve = np.sqrt(V_curve_fit_params[2][0]/len(V_curve[0])) / ( np.max(V_curve[1]) - np.min(V_curve[1]) )
        else:
            self.NRMSE_V_curve = np.nan

        #Transform Omega and R_corot to physical units, if cosmology is provided
        if self.cosmology != [] and ~np.isnan(self.redshift):
            conversion = arcsec_to_kpc(1,self.redshift,self.cosmology)

            #For everything Omega
            self.Omega_phys = Omega / conversion 
            Omega_phys_ll = self.Omega_phys - (Omega - Omega_err[0]) / conversion
            Omega_phys_ul = (Omega + Omega_err[1]) / conversion - self.Omega_phys
            self.Omega_phys_err = (Omega_phys_ll,Omega_phys_ul)
            self.Omega_phys_lst = [i / conversion for i in self.Omega_lst]


            #For everything R_corot
            self.R_corot_phys = R_corot * conversion
            R_corot_phys_ll = self.R_corot_phys - (R_corot - R_corot_err[0]) * conversion
            R_corot_phys_ul = (R_corot + R_corot_err[1]) * conversion - self.R_corot_phys
            self.R_corot_phys_err = (R_corot_phys_ll,R_corot_phys_ul)
            self.R_corot_phys_lst = [i * conversion for i in self.R_corot_lst]
            
            #For everything barlength
            # Add the errors for barlen?
            self.barlen_phys = self.barlen * conversion
            self.barlen_deproj_phys = self.barlen_deproj * conversion

            
        else:
            self.Omega_phys = np.nan
            self.Omega_phys_err = (np.nan,np.nan)
            self.Omega_phys_lst = np.nan
            self.R_corot_phys = np.nan
            self.R_corot_phys_err = (np.nan,np.nan)
            self.R_corot_phys_lst = np.nan
            self.barlen_phys = np.nan
            self.barlen_deproj_phys = np.nan



    def plot_V_curve(self,standalone = True, plot_barlen = False):
        '''
        Plots the velocity curve.

        Note to self: the orange cross doesn't always seem to hit perfectly.
         This is because the velocity curve is made with the best-fit params, 
         while the Rcr is set using the median of the MCMC. 
         That median doesn't always equal the one obtained with best-fit params.
        '''

        if standalone:
            plt.figure(figsize = (5,5))

        #arcsec_max = np.max(self.V_curve[1]) / np.abs(self.Omega) * 1.2 #this is the maximum radius. Do times 1.2 for extra buffer
        if not np.isnan(self.R_corot):
            arcsec_max = np.max([self.R_corot, np.max(self.V_curve[0])])
        else:
            arcsec_max = np.max(self.V_curve[0])
        

        xs = np.linspace(0.2, arcsec_max, 1000)
        ys = np.abs(self.Omega) * np.array(xs)
        plt.plot(xs,ys,zorder =1, ls = '--',c='black',lw=1)

        plt.scatter(self.R_corot, self.R_corot * np.abs(self.Omega), marker = "x", color='C1', s=100, zorder = 2,linewidth = 4)
        plt.scatter(self.V_curve[0],self.V_curve[1] , color='C0', zorder=0, s = 10,alpha=0.4)


        #  add the best fit
        ys_fit = velFunc(xs,self.V_curve_fit_params[0][0],self.V_curve_fit_params[0][1])
        plt.plot(xs,ys_fit,c='black', zorder =1, lw = 2,alpha=0.8)

        if plot_barlen:
            plt.axvline(self.barlen_deproj/2, linestyle = ':', c='black', lw=2)

        plt.xlabel('Distance [arcsec]')
        plt.ylabel('V [km/s]')
        if standalone:
            plt.show()

    def plot_hist_MC(self, variables = ['Omega'], standalone = True, n_bins = 15, perc = 1):
        '''
        This plots the final posterior distribution from the MC for either Omega, R_corot or R.
        '''
        n_plots = len(variables)
        for i in variables:
            assert i in ['Omega','Omega_phys','R_corot','R_corot_phys','R'], "'Omega','Omega_phys','R_corot','R_corot_phys','R'"


        if standalone == False:
            assert len(variables) == 1, "standalone = False is only an option when only asking for one map to be drawn."
        if len(variables) != 1:
            standalone = True #Standalone = False is only an option when only asking for one map to be drawn.


        if standalone:
            plt.figure(figsize = (5*n_plots,4))

        for i,variable in enumerate(variables):



            #Set titles
            if 'Omega' in variable:
                title = r"$\Omega_{bar}$ [km s-1 arcsec-1]"
            elif 'R_corot' in variable:
                title = r'R$_{\rm CR}$ [arcsec]'
            elif 'R' in variable:
                title = r'$\mathcal{R}$ [-]'
            
            if '_phys' in variable:
                title = title.replace('arcsec','kpc')
            
            if variable == 'R_phys':
                variable = 'R'


            var = getattr(self,variable)
            var_err = getattr(self,variable+'_err')
            var_lst = getattr(self,variable+'_lst')
            UL = var + var_err[1]
            LL = var - var_err[0]

            xmin = np.nanpercentile(var_lst, perc)
            xmax = np.nanpercentile(var_lst, 100-perc)
            
            if standalone:
                plt.subplot(1,n_plots,i+1)
            
            plt.hist(var_lst,bins=n_bins, range = (xmin,xmax))
            plt.axvline(var, c='black')
            plt.axvline(LL,c='black', ls='--')
            plt.axvline(UL, c='black', ls='--')
            plt.xlabel(title)

            string = str(np.round(var,2))+ "$^{+" + str(np.round(var_err[1],2)) + "}_{-" + str(np.round(var_err[0],2)) + "}$"
            plt.text(0.95,0.8,string, fontsize=16, transform=plt.gca().transAxes, horizontalalignment='right')

        if standalone:
            plt.show()
 


    def plot_X_V(self, standalone = True):
        '''
        Plots the <X> vs <V> curve
        '''
        z = self.fitted_line
        xs = np.linspace(np.min(self.X_V[0]),np.max(self.X_V[0]))
        ys = xs*z[0][0]+z[0][1]

        #fig, ax = plt.subplots()
        if standalone:
            plt.figure(figsize = (5,5))

        ax = plt.gca()

        if z[0][0] < 0:
            y_pos = 0.92
        else:
            y_pos = 0.08

        if not np.isnan(z[0][0]):
            plt.text(0.45,y_pos,f'<V> = {round(z[0][0],3)} <X> + {round(z[0][1],3)}', fontsize=10, transform=ax.transAxes, horizontalalignment='left')

        # actual plot
        plt.plot(xs,ys)
        plt.scatter(self.X_V[0],self.X_V[1])
        plt.xlabel('<X> [arcsec]')
        plt.ylabel('<V> [km/s]')

        if standalone:
            plt.show()

    
    def plot_V_curve_contours(self, mapp = 'stellar_vel', standalone = True):

        if mapp == 'stellar_flux':
            mapp = self.stellar_flux
            preset = 'default'
            title = 'stellar flux'
        elif mapp == 'stellar_vel':
            mapp = self.stellar_vel
            preset = 'velocities'
            title = 'stellar velocity'
        elif mapp == 'X_Sigma':
            #mapp = self.X_Sigma.value
            mapp = self.X_Sigma
            title = r'X $\Sigma$'
        elif mapp == 'V_Sigma':
            #mapp = self.V_Sigma.value
            mapp = self.V_Sigma
            title = r'V $\Sigma$'

        if standalone:
            plt.figure(figsize = (5,4))
        
        fig = plt.gcf()
        ax = plt.gca()
        marvin.utils.plot.map.plot(dapmap = mapp,fig=fig,ax = ax)
        plt.title(title)
                
        plt.xlim(0, mapp.shape[0])
        plt.ylim(0, mapp.shape[1])

        for aper in self.V_curve_apers[1]: #are all ellipt
            plot_aper_contours(aper, aper_type = 'EllipticalAnnulus')
        plot_aper_contours(self.V_curve_apers[0]) #is a rect
        plt.title(title)

        if standalone:
            plt.show()




    def plot_maps(self, maps = ['stellar_flux','stellar_vel'], plot_LON = False, plot_slits = False, plot_apers = False, standalone = True, plot_colorbar=True, cbar_labels = ''):
        '''
        Plots flux and velocity maps.
        Standalone is only an option when asking for only one map to be drawn.
        cbar_labels: can add list of str which will be the labels on the colorbars

        no plot_barlen()?
        No longer needing plot_colorbar
        '''

        if standalone == False:
            assert len(maps) == 1, "standalone = False is only an option when only asking for one map to be drawn."
        if len(maps) != 1:
            standalone = True #Standalone = False is only an option when only asking for one map to be drawn.

        n_plots = len(maps)
        for i in maps:
            assert i in ['stellar_flux','stellar_vel','X_map','X_Sigma','V_Sigma'], "Can only plot: 'stellar_flux', 'stellar_vel', ' X_map', 'X_Sigma' or 'V_Sigma'"

        if plot_slits:
            plot_LON = False #otherwise LON will be plotted twice

        if standalone:
            plt.figure(figsize = (5*n_plots,4))

        for i in range(n_plots):
            #determine which map to plot
            if maps[i] == 'stellar_flux':
                mapp = self.stellar_flux
                unit = r'1 x 10$^{-17}$ erg s$^{-1}$ spaxel$^{-1}$ cm$^{-2}$'
                #preset = 'default'
                title = 'stellar flux'
            elif maps[i] == 'stellar_vel':
                mapp = self.stellar_vel
                unit = r'km s$^{-1}$'
                #preset = 'velocities'
                title = 'stellar velocity'
            elif maps[i] == 'X_map':
                mapp = self.X_map
                unit = 'arcsec'
                title = r'X'
            elif maps[i] == 'X_Sigma':
                #mapp = self.X_Sigma.value #old way, when doing with pcolormesh
                mapp = self.X_Sigma
                unit = r'1 x 10$^{-17}$ erg arcsec s$^{-1}$ spaxel$^{-1}$ cm$^{-2}$'
                title = r'X $\Sigma$'
            elif maps[i] == 'V_Sigma':
                #mapp = self.V_Sigma.value #old way, when doing with pcolormesh
                mapp = self.V_Sigma
                unit = r'1 x 10$^{-17}$ erg km s$^{-2}$ spaxel$^{-1}$ cm$^{-2}$'
                title = r'V $\Sigma$'

            if standalone:
                plt.subplot(1,n_plots,i+1)
            
            if cbar_labels == '': #if no predefined colorbar label, will put unit there
                cb_label = unit 
            else:
                cb_label = cbar_labels[i]

            if maps[i] in ['X_map']: #X_map is a special case.
                fig = plt.gcf()
                ax = plt.gca()
                marvin.utils.plot.map.plot(value = mapp,fig=fig,ax = ax, cblabel = cb_label)
                plt.title(title)
                plt.xlim(0, mapp.shape[0])
                plt.ylim(0, mapp.shape[1])

            else:
                fig = plt.gcf()
                ax = plt.gca()
                marvin.utils.plot.map.plot(dapmap = mapp,fig=fig,ax = ax, cblabel = cb_label)
                plt.title(title)
                plt.xlim(0, mapp.shape[0])
                plt.ylim(0, mapp.shape[1])

            #LON
            if plot_LON:
                xs_LON = np.linspace(0,mapp.shape[0],10)
                ys_LON = xs_LON * self.LON[0] + self.LON[1]
                plt.plot(xs_LON,ys_LON,color='red',zorder = 100)

            #slits
            if plot_slits:
                for i in range(len(self.slits)):
                    xs_slit = np.linspace(0,mapp.shape[0],10)
                    ys_slit = xs_slit * self.slits[i][0] + self.slits[i][1]
                    plt.plot(xs_slit,ys_slit,color='red',zorder = 100)

            #apers
            if plot_apers:
                for i in range(len(self.apertures)):
                    plot_aper_contours(self.apertures[i])

        if standalone:
            plt.tight_layout()
            plt.show()


    def plot_img(self, image_dir = '../output/gal_images_DECaLS/', pixscale = 0.15, n_pix = 424, standalone = True, plot_apers = False, plot_barlen = False, plot_slits=False, plot_hexagon = False):
        '''
        Will plot grz image. Has to option to overlay the apertures.
        Does need the custom functions we created to pull the images.

        pixscale is in arcsec/pix

        Need to have Marvin properly setup for this to work.

        plot_slits not working
        TODO: remove image_dir arg
        '''

        
        im = marvin.tools.image.Image(self.plateifu)
        im.get_new_cutout(n_pix*pixscale, n_pix*pixscale, scale=pixscale)
        ra,dec = im.ra, im.dec

        if standalone:
            plt.figure()

        plt.imshow(im.data, origin = 'lower') # NOTE: because of origin = 'lower', north will be down now... So, going 12o'clock, 3, 6, 9, on the image, the cardinal directions now are S, W, N, E.
        plt.xticks([])
        plt.yticks([])

        centre_manga = self.centre
        centre_img = (math.ceil(n_pix/2), math.ceil(n_pix/2))
        pixscale_manga = self.pixscale
        pixscale_img = pixscale 

        if plot_apers:
            for i in range(len(self.apertures)):
                aper = self.apertures[i]
                aper_temp = photutils.aperture.RectangularAperture(
                                        (
                                        (aper.positions[0] - centre_manga[0]) * pixscale_manga / pixscale_img + centre_img[0], 
                                        (aper.positions[1] - centre_manga[1]) * pixscale_manga / pixscale_img + centre_img[1]
                                        ), 
                                        w = aper.w * pixscale_manga / pixscale_img, 
                                        h = aper.h * pixscale_manga / pixscale_img, 
                                        theta = aper.theta)
                plot_aper_contours(aper_temp, color='red', ls = '-', alpha = 0.3)

        if plot_barlen:
            barlen_pix = self.barlen / pixscale_img #pix in img
            plt.plot([centre_img[1] - np.sin(self.PA_bar/180*np.pi)*barlen_pix/2, centre_img[1] + np.sin(self.PA_bar/180*np.pi)*barlen_pix/2],
                    [centre_img[0] - np.cos(self.PA_bar/180*np.pi)*barlen_pix/2, centre_img[0] + np.cos(self.PA_bar/180*np.pi)*barlen_pix/2], 
                    c='yellow')

        if plot_hexagon:
            ax = plt.gca()
            im.overlay_hexagon(ax, color='magenta', linewidth=1)

        #slits
        """
        if plot_slits:
            '''
            find slit centres. we know PA. Get eqs from that. 
            '''
            points = get_slit_centres(self.stellar_flux,self.slits,centre_manga)

            for i in range(len(self.slits)):
                xs_slit = np.linspace(0,img.shape[0],10)
                ys_slit = xs_slit * self.slits[i][0] + self.slits[i][1]
                plt.plot(xs_slit,ys_slit,color='red')
        
        """
        if standalone:
            plt.tight_layout()
            plt.show()


##----------##
## Main def ##
##----------##

def Tremaine_Weinberg(PA, inc, barlen, PA_bar, maps, PA_err = 0.0, inc_err = 0.0, barlen_err = 0.0, PA_bar_err = 0.0,
                    slit_width = 1, slit_separation = 0, slit_length_method = 'default', slit_length = np.inf,
                    min_slit_length = 5, n_iter = 0, cosmo = [], redshift = np.nan, aperture_integration_method = 'center', 
                    forbidden_labels = ['DONOTUSE','UNRELIABLE','NOCOV'], deproject_bar = True, correct_velcurve = True, 
                    velcurve_aper_width = 5, check_convergence = False, convergence_n = 2, convergence_threshold = 5, 
                    convergence_stepsize = 0):
    
    '''
    Main function that user will call. Will return the TW class. 

    Inputs:
    PA (float): Position angle of galaxy, in degrees. The PAs are defined as East of North.
    inc (float): Inclination of galaxy, in degrees.
    barlen (float): Length of the entire bar of the galaxy, in arcsec (so not bar radius, but bar diameter!).
    PA_bar (float): Position angle of the bar, in degrees.
    maps (MaNGA Maps): A MaNGA maps object. If you have a MaNGA plateifu, can get the maps object by doing: Maps(plateifu = plateifu, bintype='VOR10'). See: https://sdss-marvin.readthedocs.io/en/latest/tools/maps.html.


    Optional inputs:
    PA_err (float): Error on the galaxy PA, in degrees.
    inc_err (float): Error on the inclination of the galaxy, in degrees.
    barlen_err (float): Error on the length of the bar, in arcsec (error on the entire bar, not bar radius!).
    PA_bar_err (float): Error on the PA of the bar, in degrees.

    slit_width (float): Width of each slit, in arcsec.
    slit_separation (float): Separation between slits, in arcsec.
    slit_length_method (str): Either 'default' or 'user_defined'. Decides which method to use to determine the slit lengths.
    slit_length (float): Maximum value of slit length, in arcsec. If slit_length_method == 'user_defined', this is the slit length that will be used.
    min_slit_length (float): Minimum value of slit length, in arcsec. If slit is shorter than that, ignore slit.

    n_iter (int): Amount of iterations used to determine the posterior distributions of Omega, Rcr and R. Default is 0. Recommended value for accurate posteriors and errors is 1000.
    cosmo (astropy cosmology): An astropy cosmology (e.g.: FlatLambdaCDM(H0=70 km / (Mpc s), Om0=0.3, Tcmb0=2.725 K, Neff=3.04, m_nu=[0. 0. 0.] eV, Ob0=None)). Used together with `redshift' to convert arcsec to kpc.
    redshift (float): Redshift of the target. Used together with `cosmo' to convert arcsec to kpc. 
    aperture_integration_method (bool): The integration method used with the apertures. Can be either 'center' or 'exact'.
    forbidden_labels (list): List of possible labels in the MaNGA datacube. Will ignore spaxels that are associated with any of these labels.
    deproject_bar (bool): Whether to deproject the bar using the PA, PA_bar and inclination of the galaxy. Strongly advised to always keep on True.
    correct_velcurve (bool): Whether to correct the velocity and positions for the inclination and PA of the galaxy while determining the velocity curve.
    velcurve_aper_width (int): How many pixels to use to determine the velocity curve.

    Outputs:
    Returns TW class, defined in TremaineWeinberg.py. The TW class contains everything that is calculated. See Example.ipynb to see how to access it.

    Notes:
    Currently, if n_iter = 0, it will run once with best-guess inputs. If n_iter > 0, it will run the iterations, Omega, Rcr and R are the
    median of all the iterations. After the iterations, it will run one last time with the best-guess inputs for all the figures etc. 
    All PAs are defined as East of North.
    '''


    #Part 0: initialise class and other stuff
    tw = TW(PA, inc, barlen, PA_bar, maps, forbidden_labels, PA_err, inc_err, barlen_err, PA_bar_err, slit_width,cosmo, redshift) 
    pixscale = get_pixscale(tw)
    centre = get_centre(tw.stellar_flux)

    # Part 1 - 5 are in this loop
    if n_iter == 0: # If n_iter == 0, do everything once and don't sample
        PA_err = 0
        inc_err = 0
        PA_bar_err = 0
        barlen_err = 0
    
    for n in range(n_iter+1):
        if n == n_iter: #means we're over the MC. Doing one more round with the best-fit values for the figures.
            PA_temp = PA
            inc_temp = inc
            PA_bar_temp = PA_bar
            barlen_temp = barlen
        else:
            PA_temp = np.random.normal(PA, PA_err)
            inc_temp = np.random.normal(inc, inc_err)
            PA_bar_temp = np.random.normal(PA_bar, PA_bar_err)
            barlen_temp = np.random.normal(barlen, barlen_err)

        
        # Adjust inc_temp if needed
        if inc_temp < 0:
            inc_temp = 0
        elif inc_temp > 90:
            inc_temp = 90

        # Adjust PA_temp if needed
        if PA_temp == 0 or PA_temp == 180: #otherwise code crashes
            PA_temp = 10e-6
        if PA_temp == 90:
            PA_temp = 90 - 10e-6

        # Adjust barlen_temp if needed
        if barlen_temp < 0:
            barlen_temp = 0

        # Part 1: Get Line of Nodes (LON)
        m_LON, b_LON = get_LON(tw.stellar_flux, PA_temp, centre)
        xs_LON = np.linspace(0,tw.stellar_flux.shape[0],10)
        ys_LON = xs_LON * m_LON + b_LON

        # Part 2: Get other slits
        slits = get_pseudo_slits(tw.stellar_flux, (m_LON, b_LON), PA_temp, PA_bar_temp, barlen_temp, centre,
            pixscale = pixscale, sep = slit_separation, width_slit = slit_width)


        # Part 3: Create X * Sigma and Vlos * Sigma maps
        X_map = create_X_map(tw.stellar_flux, slits[0], centre, pixscale)
        X_Sigma = tw.stellar_flux * X_map
        V_Sigma = tw.stellar_flux * tw.stellar_vel



        # Part 4: Convert slits to actual apertures
        points = get_slit_centres(tw.stellar_flux,slits,centre)
        if slit_length_method == 'default':
            hex_map = create_hexagon_map(tw.stellar_vel,forbidden_labels)
        else:
            hex_map = []

        apers = []
        aper_Omegas = [[],[]] #position 0 is for converged apertures, position 1 is not not converged apertures
        for i, s in enumerate(slits):
            aper = get_aper(points[i], slit_width/pixscale, (PA_temp)/180*np.pi, tw.stellar_vel, slit_length_method = slit_length_method, hex_map = hex_map, slit_length = np.round(slit_length/pixscale))
            if aper.h > min_slit_length/pixscale: #ensure that it is not too short
                
                if check_convergence: 
                    converged, aper_Omega = aperture_convergence(aper,X_Sigma, V_Sigma, aperture_integration_method, convergence_n, convergence_threshold, convergence_stepsize)
                else: 
                    converged = True
                    aper_Omega = []

                if converged:
                    aper_Omegas[0].append(aper_Omega)
                    apers.append(aper)
                else:
                    aper_Omegas[1].append(aper_Omega)

        # Step 5: Do integration and determine Omega
        Xs, Vs, z, Omega = determine_pattern_speed(tw.stellar_flux, X_Sigma, V_Sigma, apers, inc_temp, aperture_integration_method, forbidden_labels = tw.forbidden_labels)
        

        # Step 6: Find V curve and corotation radius
        R_corot, vel, arcsec, V_curve_apers, V_curve_fit_params = determine_corotation_radius(Omega, tw.stellar_vel, tw.on_sky_xy, centre, PA_temp, inc_temp, maps, forbidden_labels = tw.forbidden_labels, correct_velcurve = correct_velcurve, velcurve_aper_width = velcurve_aper_width)
        delta_PA = np.abs(PA_temp - PA_bar_temp)
        if deproject_bar:
            bar_rad_deproj = barlen_temp/2 * np.sqrt(np.cos(delta_PA/180*np.pi)**2 + np.sin(delta_PA/180*np.pi)**2 / np.cos(inc_temp/180*np.pi)**2) #https://ui.adsabs.harvard.edu/abs/2007MNRAS.381..943G/abstract
            R = R_corot / bar_rad_deproj
        else:
            R = R_corot / (barlen_temp/2)

        if n < n_iter or n_iter == 0:
            tw.save_intermediate_MC_results(apers,Omega,R_corot,R,2*bar_rad_deproj,[Xs, Vs],z,[arcsec, vel],V_curve_fit_params)

    Omega = np.nanmedian(tw.Omega_lst)
    R_corot = np.nanmedian(tw.R_corot_lst)
    R = np.nanmedian(tw.R_lst)

    # Step 7: Error propogation: put in separate function later
    if n_iter == 0: 
        Omega_err_ll, Omega_err_ul = np.nan, np.nan
        R_corot_err_ll, R_corot_err_ul = np.nan, np.nan
        R_err_ll, R_err_ul = np.nan, np.nan
    else: 
        # Error on Omega
        Omega_err_ul = np.nanpercentile(tw.Omega_lst,84)-Omega
        Omega_err_ll = Omega - np.nanpercentile(tw.Omega_lst,16)

        #Error on R_corot
        R_corot_err_ul = np.nanpercentile(tw.R_corot_lst,84)-R_corot
        R_corot_err_ll = R_corot - np.nanpercentile(tw.R_corot_lst,16)

        #Error on curly R
        R_err_ul = np.nanpercentile(tw.R_lst,84)-R
        R_err_ll = R - np.nanpercentile(tw.R_lst,16)

    # Part -1: Save results and return
    tw.save_results(centre, pixscale, (m_LON, b_LON), slits, apers, aper_Omegas, X_map, X_Sigma, V_Sigma, [Xs, Vs], z, Omega, (Omega_err_ll, Omega_err_ul), R_corot, (R_corot_err_ll, R_corot_err_ul), R, (R_err_ll, R_err_ul), [arcsec, vel], V_curve_apers, V_curve_fit_params, bar_rad_deproj*2) 
    return tw


    





##-------------##
## Helper defs ##
##-------------##



# Part 0: Prepare MaNGA masks

def get_Vsys(stellar_vel, on_sky_xy, arcsec_range, forbidden_labels = ['DONOTUSE']):
    '''
    It looks like not all MaNGA maps have Vsys correctly removed. We can estimate Vsys by looking at the 
    central 5arcsecs. 
    '''

    stellar_vel_pixmask = stellar_vel.pixmask.get_mask(forbidden_labels)
    stellar_vel_value = stellar_vel.value

    # get Vsys
    central_vels = []
    for i in range(len(stellar_vel)):
        for j in range(len(stellar_vel[i])):
            if on_sky_xy[i][j] < arcsec_range:
                if not stellar_vel_pixmask[i][j] >=1:
                    central_vels.append(stellar_vel_value[i][j])
    
    Vsys = np.nanmean(central_vels)

    # apply Vsys
    for i in range(len(stellar_vel)):
        for j in range(len(stellar_vel[i])):
            if not stellar_vel_pixmask[i][j] >=1:
                stellar_vel.value[i][j] = stellar_vel_value[i][j] - Vsys


    return stellar_vel, Vsys



def arcsec_to_kpc(alpha,z,cosmo):
    '''
    alpha must be in arcsec. cosmo must be an astropy cosmology. E.g.:
    
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    '''
    L = cosmo.kpc_proper_per_arcmin(z) * alpha/60 #kpc/arcmin * arcmin = kpc
    return L.value



# Part 1: Get Line of Nodes
 
def get_centre(mapp, centre_method = 'brightest', sigma = 3):
    '''
    Two modes: get the centre of the map or find the brightest pixel.
    For even-pixelled maps, will round down when getting the centre
    TODO: Disable not usable regions
    
    '''


    assert centre_method in ['brightest','centre']
    
    if centre_method == 'centre':
        return (math.ceil(mapp.value.shape[0]/2),math.ceil(mapp.value.shape[1]/2)) 
    
    if centre_method == 'brightest':
        gaussian_field = scipy.ndimage.gaussian_filter(mapp.value, sigma=sigma)
        inds = np.where(gaussian_field == np.max(gaussian_field))
        if len(inds[0]) > 1:
            #if more than one pixel found, take one closest to actual centre
            dists = []
            centre = get_centre(mapp, centre_method = 'centre')
            for i in range(len(inds[0])):
                #loop over pairs, find dists to actual centre, take closest one.
                d = np.sqrt((centre[0]-inds[0][i])**2 + (centre[1]-inds[1][i]))
                dists.append(d)

            inds_dists = np.where(dists == np.min(dists))[0][0]
            return inds[1][inds_dists],inds[0][inds_dists]
        else:
            return inds[1][0], inds[0][0]



def get_LON(mapp, PA, centre):
    '''
    Give a MaNGA map and a PA, will return eqs of LON. 
    LON should go through centre and aligned with PA. So we have an angle and point
    '''
    PA_rad = PA * np.pi / 180.
    values = mapp.value
    p1 = centre #the centre
    p2 = (p1[0] + 10 * np.sin(PA_rad), p1[1] + 10 * np.cos(PA_rad)) #10 is arbitrary, could be anything.
    
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = -m * p2[0] + p2[1]
    
    return m,b


def get_pixscale(tw):
    '''
    Can also just do return 0.5, I believe.
    '''
    seps_x = []
    # for on_sky_x
    temp = tw.on_sky_x.value #to make it faster
    for i in range(len(tw.on_sky_x)):
        for j in range(len(tw.on_sky_x[i])-1):
            sep = np.abs(temp[i][j] - temp[i][j+1])
            seps_x.append(sep)

    seps_y = []
    temp = tw.on_sky_y.value #to make it faster
    for j in range(len(tw.on_sky_y[0])):
        for i in range(len(tw.on_sky_y)-1):
            sep = np.abs(temp[i][j] - temp[i+1][j])
            seps_y.append(sep)

    sep = np.average(np.array([seps_x,seps_y]).flatten())
    return sep
    
    

# Part 2: Get other slits

def get_slit_separation_correction(PA):
    '''
    See eqs 6 from Garma-Oehmichen (2020).

    TODO: implement
    '''
    
    return np.nan
    

def get_pseudo_slits(mapp,LON,PA, PA_bar, barlength, centre, sep = 1, width_slit = 0, pixscale = 0.5):
    '''
    This will return a list of all m,b of all pseudo_slits.
    pixscale should be in arcsec/pix. barlength, sep and width_slit should be in arcsec
    
    perpendicular line needs to go through centre
    '''
    
    sep = ( sep + width_slit ) / pixscale #pixels 
    barlength = barlength / pixscale #bar length in pixels
    sep = sep# * get_slit_separation_correction(PA) #add this as a keyword later?


    
    m_pLON = -1/LON[0] #m1 * m2 = -1 for perpendicular lines
    b_pLON = -m_pLON * centre[0] + centre[1]
    
    '''
    So eq of perpendicular line is: y = m * x + b, or y = m (x - x0) + y0. (x0,y0) is the centre
    We also know d**2 = (x1-x0)**2 + (y1-y0)**2. Substitute first into second to get:
    d**2 = (x1-x0)**2 + m**2 * (x1-x0)**2, solve for x1 and get:
    x1 = x0 + d / (sqrt(1+m**2))
    Put in first eqs again and get y1. Now you have the point.
    Now, get line through that point and with direction from original LON
    '''
    
    delta_PA = np.abs(PA - PA_bar)
    corr_factor = np.abs(np.sin(delta_PA/180*np.pi)) 
    
    pseudo_slits = [LON]
    
    for k in [-1,1]: #left and right side
        d = sep
        while d < barlength/2*corr_factor:

            x1 = centre[0] + k * d / np.sqrt(1+m_pLON**2)
            y1 = m_pLON * (x1 - centre[0]) + centre[1]
            #y1_2 = m_pLON * x1 + b_pLON

            m = LON[0]
            b = -m * x1 + y1
            pseudo_slits.append((m,b))
            
            d += sep
        
        
    return pseudo_slits


# Part 3: Create X * Sigma and Vlos * Sigma maps

def is_left_of(line,point):
    x = (point[1] - line[1])/line[0]
    
    if x >= point[0]:
        return -1
    else:
        return 1


def create_X_map(mapp, LON, centre, sep):
    '''
    x-axis should be aligned with LON.
    '''
    empty = np.full_like(mapp.value,np.nan)

    m_pLON = -1/LON[0] #m1 * m2 = -1 for perpendicular lines
    b_pLON = -m_pLON * centre[0] + centre[1]
    
    for i in range(len(empty)):
        for j in range(len(empty[i])):
            #find distance between i,j and line
            
            a, b, c = m_pLON, -1, b_pLON
            dist = np.abs(a*j+b*i+c)/np.sqrt(a**2+b**2) #https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
            
            empty[i][j] = is_left_of([m_pLON, b_pLON],[j,i])*dist*sep
    
    return empty



# Part 4: Get actual apertures
def get_slit_centres(mapp,slits,centre):
    '''
    Need slit centres to put actual apertures on them
    Returns slit centres. Based on LON - which should be the first in the slit. Creates perpendicular line
    and finds intersection with all slits.
    '''
    
    LON = slits[0]
    m_pLON = -1/LON[0] #m1 * m2 = -1 for perpendicular lines
    b_pLON = -m_pLON * centre[0] + centre[1]
    
    coor = []
    for s in slits:
        '''
        y1 = m1 * x1 + b1
        y2 = m2 * x2 + b2
        
        if x1 = x2 and y1 = y2, then
        
        x = (b2 - b1)/(m1-m2)
        y = put x in either of the eqs above.
        '''
    
        x = (s[1] - b_pLON)/(m_pLON - s[0])
        y = s[0] * x + s[1]
        
        coor.append((x,y))
    return coor


    
def plot_aper_contours(aper, color = 'white', aper_type = 'RectAper', ls = '-', alpha = 1):
    """
    Because of a weird quirk with how pcolormesh labels its axes, the indexing seems off by 0.5 pixel. 
    I.e., the data[0][0] position seems to be at position 0.5,0.5 in the plot.
    This is noticeable when plotting the aperature contours, which don't have this quirk. 
    So make function to deal with this.
    """
    
    assert len(aper.positions.flatten()) == 2, "Cannot deal with multiple positions, yet."
    
    if aper_type == 'RectAper':
        aper_plot = photutils.aperture.RectangularAperture((aper.positions[0]+0.5, aper.positions[1]+0.5), 
                                        w = aper.w, h = aper.h, theta = aper.theta)


    if aper_type == 'CircularAnnulus':
        
        aper_plot = CircularAnnulus((aper.positions[0]+0.5, aper.positions[1]+0.5),
                                     r_in = aper.r_in, r_out = aper.r_out)

    
    if aper_type == 'EllipticalAnnulus':
        
        aper_plot = photutils.aperture.EllipticalAnnulus((aper.positions[0]+0.5, aper.positions[1]+0.5),
                                     a_in = aper.a_in, a_out = aper.a_out,
                                     b_in = aper.b_in, b_out = aper.b_out,
                                     theta = aper.theta)

    aper_plot.plot(color=color, ls = ls, alpha = alpha, zorder = 100)

def create_hexagon_map(mapp,forbidden_labels):
    newmap = np.full_like(mapp.value,0)
    labels = mapp.pixmask.labels
    for i in range(len(mapp)):
        for j in range(len(mapp[i])):
            labels_in_pix = labels[i][j]
            for l in labels_in_pix:
                if l in forbidden_labels:#['NOCOV','LOWCOV','NOVALUE','DONOTUSE','UNRELIABLE']: #Or change to forbidden lablels?
                    newmap[i][j] = 1
    return newmap


def get_aper(slit_centre, slit_width, slit_theta, stellar_vel, slit_length_method = 'default', hex_map = [], slit_length = np.inf):
    '''
    Will convert slit into an actual aperture.

    slit_length_method can be in ['default', 'user_defined']. With user_defined, it'll just take whatever value is in slit_length.
    With default, it'll create a length for every slit while taking the edges of the hexagon into account.
    Will keep on increasing height of the aperture until it hits the end. That's Lmax for this slit.
    This is more accurate, but also slower.

    Units should be in pixels. 

    slit_theta here is in radians, from the positive x axis, increasing counter clockwise. Our x and y axis are reversed though in our arrays.

    slit_length (float): maximum value of slit length, in pixels
    '''

    slit_theta = -slit_theta #because of how angles are handled in photutils

    assert slit_length_method in ['default','user_defined'], "slit_length_method must be either 'default' or 'user_defined'."

    if slit_length_method == 'default':

        #algorithm will take big steps steps first, until it overshoots. Then reduce stepsize to finetune
        #this is done for speeding up the code. 
        h = 1

        h_stepsizes = [20,5,1]

        aper = photutils.aperture.RectangularAperture(slit_centre, w = slit_width, h = h, theta = slit_theta)
        reached_hex = float(photutils.aperture.aperture_photometry(hex_map, aper, method = 'center')['aperture_sum'])

        for h_stepsize in h_stepsizes:

            reached_hex = 0.0 
            while reached_hex == 0.0 and h <= slit_length:
                h+=h_stepsize
                aper = photutils.aperture.RectangularAperture(slit_centre, w = slit_width, h = h, theta = slit_theta)
                reached_hex = float(photutils.aperture.aperture_photometry(hex_map, aper, method = 'center')['aperture_sum'])

            h-=h_stepsize

        aper = photutils.aperture.RectangularAperture(slit_centre, w = slit_width, h = h, theta = slit_theta)    
    
    if slit_length_method == 'user_defined':
        assert ~np.isinf(slit_length), 'Please define slit_length to use this setting.'
        aper = photutils.aperture.RectangularAperture(slit_centre, w = slit_width, h = slit_length, theta = slit_theta)
    
    return aper

def aperture_convergence(aper, X_Sigma, V_Sigma, aperture_integration_method, convergence_n = 5, convergence_threshold = 5, convergence_stepsize = 0):
    '''
    Check if this aperture converges, as defined by Zou et al. (2019)

    Definition of convergence is arbitrary. In our case, we look at the change in the last convergence_n datapoints. If the median change is
    less than the convergence_threshold, we decide the slit has converged.

    If convergence_stepsize is left to default, we only check the last convergence_n slit lengths. This minimises time wasted on slit lengths
    that we're not checking for convergence anyway. 


    Below is outdated. TODO: Implement option to do this in the future.
    If convergence_stepsize is left to default, we implement a scheme so that more attention is placed near Lmax. This is done because
    testing slit convergence is time-consuming. So: every slit length between 1.5*conv_threshold is probed with a stepsize of 1.
    Them, between 1.5*conv_threshold and 3*conv_threshold, we probe with a stepsize of 2. Then, between 3*conv_threshold and 5*conv_threshold with a 
    stepsize of 5. Finally, we use a stepsize of 10 until we get to slit_length = 0. This way, we significantly speed up the testing
    convergence process, without sacrificing accuracy.


    '''

    Lmax = aper.h
    if convergence_stepsize == 0: #if no user-defined stepsize, implement our stepsize plan
        """
        threshold1, threshold2 = Lmax - math.ceil(1.2 * (convergence_n)), Lmax - math.ceil(5 * (convergence_n))
        slit_lengths = list(np.arange(Lmax, threshold1, -1)) + list(np.arange(threshold1, threshold2, -5))
        while slit_lengths[-1] > 0:# keep adding by 10 until we hit the bottom
            slit_lengths.append(slit_lengths[-1] - 20)
        slit_lengths = slit_lengths[::-1] # reverse
        slit_lengths = [i for i in slit_lengths if i > 0] #remove all negative ones
        """

        threshold = Lmax - math.ceil(convergence_n)+1
        slit_lengths = list(np.arange(threshold, Lmax+1, 1))
        slit_lengths = [i for i in slit_lengths if i > 0] #remove all negative ones

    else:
        slit_lengths = np.arange(1,Lmax+1,convergence_stepsize)

    aper_Omegas = []

    # Calculate Omega for all slit lengths
    for i, sl in enumerate(slit_lengths):
        aper_temp = photutils.aperture.RectangularAperture(aper.positions, w = aper.w, h = sl, theta = aper.theta)
        # Don't need to calculate flux integral here as it cancels out
        phot_table_X = photutils.aperture.aperture_photometry(X_Sigma.value, aper_temp, method = aperture_integration_method)
        phot_table_V = photutils.aperture.aperture_photometry(V_Sigma.value, aper_temp, method = aperture_integration_method)
        X = float(phot_table_X['aperture_sum'])
        V = float(phot_table_V['aperture_sum'])
        if X != 0:
            aper_Omegas.append(np.abs(V/X))
        else:
            aper_Omegas.append(np.nan)

    # Check convergence
    convergence_val = np.nanmedian(np.abs(np.diff(aper_Omegas))[-convergence_n+1:])
    #print(convergence_val)
    if convergence_val <= convergence_threshold:
        converged = True
    else:
        converged = False

    return converged, [aper_Omegas,slit_lengths]



# Step 5: Do integration and determine Omega

def determine_pattern_speed(stellar_flux, X_Sigma, V_Sigma, apers, inc, aperture_integration_method, forbidden_labels = ['DONOTUSE']):
    '''
    Takes the flux, X_Sigma, V_Sigma, apertures and inclination of the galaxy and returns the Xs and Vs integrals, the slope of the fitted line and the pattern speed.
    '''
    Xs = []
    Vs = []

    for aper in apers:#can I ignore forbidden_labels here? -> Yes
        phot_table_flux = photutils.aperture.aperture_photometry(stellar_flux.value, aper, method = aperture_integration_method)
        flux = float(phot_table_flux['aperture_sum'])

        if flux == 0:
            continue #just ignore
        else: #can I ignore forbidden_labels here? -> Yes
            phot_table_X = photutils.aperture.aperture_photometry(X_Sigma.value, aper, method = aperture_integration_method)
            Xs.append(float(phot_table_X['aperture_sum'])/flux)
            phot_table_V = photutils.aperture.aperture_photometry(V_Sigma.value, aper, method = aperture_integration_method)
            
            Vs.append(float(phot_table_V['aperture_sum'])/flux)

    # Get the fit
    if len(Xs) > 1:
        z = np.polyfit(Xs,Vs,1, full=True)
        Omega = z[0][0]/np.sin(inc/180*np.pi)
    else:
        z = (np.array([np.nan, np.nan]), np.array([]), np.nan, np.array([]), np.array([])) #so it has same shape as a normal z
        if len(Xs) == 1 and Xs[0] != 0:
            Omega = Vs[0]/Xs[0]/np.sin(inc/180*np.pi)
        else:
            Omega = np.nan

    return Xs, Vs, z, np.abs(Omega)


# Step 6: Find V curve and corotation radius


def create_phi_map(centre, PA, mapp):
    '''
    Creates a map with the difference in rads to the PA of the galaxy. Used to determine correction factor
    '''

    phi_map = np.full_like(mapp, 0)

    for i in range(len(phi_map)):
        for j in range(len(phi_map[i])):
            
            dX = j - centre[0]
            dY = i - centre[1]
            
            if dX != 0:
                angle = math.atan(dY/dX)/np.pi*180
            else:
                angle = 90
            angle = 90 - angle # These angles should be East of North too. Without doing 90-angle, they are counter-cockwise from positive x-axis
            phi = angle - PA

            while phi > 180:
                phi -= 180
            while phi < 0:
                phi += 180

            phi_map[i][j] = phi / 180 * np.pi

    return phi_map


def get_n_pix_in_aper(mapp,aper, method = 'center'):
    '''
    Needed to calculate average value in aperature. We need amount of pixels considered.
    mapp should only have the data

    Actually, maybe look into this instead: https://photutils.readthedocs.io/en/stable/api/photutils.aperture.PixelAperture.html#photutils.aperture.PixelAperture.area_overlap

    '''
    
    ones = np.full_like(mapp, 1) #to calculate n_pix
    
    phot_table = photutils.aperture.aperture_photometry(ones, aper, method = method)
    n_pix = phot_table['aperture_sum']
    
    #however, maybe some values are in annulus, but don't have values in the mapp
    illegal_vals = [0,np.nan]
    
    mask = aper.to_mask(method = method).to_image(shape = (mapp.shape))
    #masked_data = mask.to_image(shape = (mapp.shape)) * mapp
    
    n = 0
    for i in range(len(mapp)):
        for j in range(len(mapp[i])):
            if mask[i][j] != 0.0: #so if it is in the mask
                if mapp[i][j] in illegal_vals:
                    n+=1
    
    return float(n_pix - n)


def velFunc(xdata, Vflat,rt):
    '''
    Basic 2 parameter arctan function to model the rotation curve. From Courteau 1997.
    V(r) = Vsys + 2/pi * Vflat * arctan( (r - r0)/rt )
    '''
    xdata = np.array(xdata)
    r0 = 0 #assume this to be the case
    Vsys = 0 #assume this to be the case
    return Vsys + 2/np.pi * Vflat * np.arctan((xdata-r0)/rt)


def determine_corotation_radius(Omega, stellar_vel, on_sky_xy, centre, PA, inc, maps, forbidden_labels = ['DONOTUSE'], correct_velcurve = True, velcurve_aper_width = 10):
    '''
    Calculates the corotation radius, based on all the other parameters.
    '''
    # Find the rectangular aperature
    aper_rect = photutils.aperture.RectangularAperture(centre, w = velcurve_aper_width, h = stellar_vel.shape[0]*1.5, theta = (-PA)/180*np.pi) #-PA due to how photutils deal with their theta


    # apply the rect aperature first
    stellar_vel_rect = aper_rect.to_mask(method = 'center').to_image(shape = (stellar_vel.shape)) * stellar_vel.value * (stellar_vel.pixmask.get_mask(forbidden_labels)<1).astype(int) #last bit is to incorporate stellar_vel mask
    on_sky_xy_rect = aper_rect.to_mask(method = 'center').to_image(shape = (on_sky_xy.shape)) * on_sky_xy
    
    # apply correction (from inclination and phi)
    phi_map = create_phi_map(centre, PA, on_sky_xy)

    stellar_vel_rect_corr = np.full_like(stellar_vel_rect,0)
    on_sky_xy_rect_corr = np.full_like(on_sky_xy_rect,0)
    

    if correct_velcurve: 
        for i in range(len(stellar_vel_rect_corr)):
            for j in range(len(stellar_vel_rect_corr[i])):
                phi = phi_map[i][j]
                corr_factor_vel = 1/np.abs(np.sin(inc/180*np.pi) * np.cos(phi)) #https://www.aanda.org/articles/aa/pdf/2005/04/aah4175.pdf and https://articles.adsabs.harvard.edu//full/1989A%26A...223...47B/0000057.000.html
                corr_factor_xy = np.sqrt(np.cos(phi)**2 + np.sin(phi)**2 / np.cos(inc/180*np.pi)**2) #https://arxiv.org/pdf/1406.7463.pdf and https://academic.oup.com/mnras/article/381/3/943/1062417 and https://ui.adsabs.harvard.edu/abs/2007MNRAS.381..943G/abstract
                if corr_factor_vel < 3 and corr_factor_xy < 3: #if correction factor is too high, it'll blow up the results (probably inaccurately)
                    vel = stellar_vel_rect[i][j] * corr_factor_vel
                    stellar_vel_rect_corr[i][j] = vel

                    xy = on_sky_xy_rect[i][j] * corr_factor_xy
                    on_sky_xy_rect_corr[i][j] = xy    
    else:
        stellar_vel_rect_corr = stellar_vel_rect
        on_sky_xy_rect_corr = on_sky_xy_rect


    # Get the final curve
    arcsec = []
    vel = []

    #stellar_vel_mask = aper_rect.to_mask(method = aperture_integration_method).to_image(shape = (stellar_vel.shape))
    #on_sky_xy_mask = aper_rect.to_mask(method = aperture_integration_method).to_image(shape = (on_sky_xy.shape))
    apers_circ = []
    for i in range(len(stellar_vel_rect_corr)):
        for j in range(len(stellar_vel_rect_corr[i])):
            if stellar_vel_rect_corr[i][j] != 0.0 and on_sky_xy_rect_corr[i][j] != 0.0: #0.0 can be either because of outside rect apers or because no data.
                vel.append(np.abs(stellar_vel_rect_corr[i][j]))
                arcsec.append(on_sky_xy_rect_corr[i][j])


    # Fit profile with basic 2 param arctan function
    if len(arcsec) > 0 and len(vel) > 0:
        popt, pcov = scipy.optimize.curve_fit(velFunc, arcsec, vel, bounds = (0, np.inf))
        MSE = np.sum( (vel - velFunc(arcsec,popt[0],popt[1]))**2)


        # Get Omega intersection, which is intersection of velocity curve with Omega * r
        idx = 0
        buffer = 1.2
        while idx == 0:

            '''
            need to add case if the Vs_corot curve is just higher than the velocity curve. Found couple galaxies
            where this is the case, e.g.
            '''

            arcsec_max = np.max(vel) / np.abs(Omega) * buffer #this is the maximum radius. Do times buffer to increase range
            arcsec_fit = np.linspace(0, arcsec_max, 1000)

            Vs_corot = np.abs(Omega) * np.array(arcsec_fit)
            vels_fit = velFunc(arcsec_fit,popt[0],popt[1])
            

            idxs = np.argwhere(np.diff(np.sign(vels_fit - Vs_corot))).flatten()

            if len(idxs) > 1: #if there is more than one solution, we have found it. The first solution is always at r=0
                idx = idxs[1] #take the second one
                R_corot = arcsec_fit[idx]
            else:
                buffer = buffer * 1.2

            if arcsec_max > 3600: #if larger than an deg, it is not meaningful. Probably something else went wrong.
                R_corot = np.nan
                break


    else:
        R_corot, popt, pcov, MSE = np.nan, np.array([np.nan, np.nan]), np.array([[np.nan, np.nan],[np.nan, np.nan]]), np.nan 


    return R_corot, vel, arcsec, [aper_rect, apers_circ], [popt,pcov,np.array([MSE])]






####################################
### Saving and loading the class ###
####################################

def save_TWs(TWs,fileloc = ''):
    '''
    TWs must be a list of all TWs you want to save
    fileloc is where you want to save the data
    
    Uses the pickle package to save all TWs. Will run pickle.dumps() on every tw, save the returned 
    bytes string, and store it as a string in a csv file, where it can be read later on using the function 
    load_TWs()
    
    To save, I convert the object to bytes using pickle. Then, I convert the bytes to a hex, and save the hex
    as a string in a csv file. Feels a bit convoluted, but I couldn't directly save the bytes in a csv... 
    '''
    if fileloc == '':
        fileloc = 'temp.csv'
        
    with open(fileloc, 'w') as f:
            writer = csv.writer(f)
            for tw in TWs:
                hex_str = pickle.dumps(tw).hex()
                writer.writerow([hex_str])
        
def load_TWs(fileloc):
    '''
    Will return a list of TWs stored at fileloc, which was saved using save_TWs().
    '''
    
    TWs = []
    with open(fileloc,'r') as f:
        for row in f:
            TWs.append(pickle.loads(bytes().fromhex(row)))               
    return TWs