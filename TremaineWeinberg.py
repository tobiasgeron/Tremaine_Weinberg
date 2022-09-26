'''
Created December 2021 by Tobias Géron.
Contains everything to perform the Tremaine-Weinberg method on MaNGA galaxies.
More info on the method: 
Tremaine, Weinberg (1984): https://ui.adsabs.harvard.edu/abs/1984ApJ...282L...5T/abstract
Aguerri et al. (2015): https://ui.adsabs.harvard.edu/abs/2015A%26A...576A.102A/abstract
Cuomo et al. (2019): https://ui.adsabs.harvard.edu/abs/2019A%26A...632A..51C/abstract
Garma-Oehmichen et al. (2020): https://ui.adsabs.harvard.edu/abs/2020MNRAS.491.3655G/abstract
Géron et al. (2022): in prep. 


TODO: 
Major:
-Change PA to be East or North!!!

Minor:
- Lmax also limited by low SNR and other spaxels that aren't symmetric though? -> No cause often middle of 
stellar_vel map has low SNR so slits would have very low/nonexistent length?
-fix slit width and slit separation? Now they are confusing.
- if plot_maps(plot_apers = True and plot_slits = True), the slits don't go through middle of apers
- Should lower limits have - in their values or no? Check in both normal units and physical units! Think you should just be able to do Omega + ll and that it goes to the right lower limit. So yeah, make the ll negative.
- Take absolute value of Omega and check MC does it correctly.
- In step V, I'm ignoring low SNR and forbidden labels now. It goes roughly double as fast now. But is it okay to do so?
I'm quite sure the SNR makes sense, but not about the forbidden labels...
- Add the other params. Maybe add extra item in V_params to indicate the 
- fix error propogation. Some old stuff in there. 
-We are making nMC velcurves right? Do I need to save all of them too? Together with all the Vcs? Which one is being saved now?
- I think that currently, the last iteration is the one that gets saved in X_V plot and velcurve plot etc, not the one with the 
most likely estimates. FIX THIS!
-IMPORTANT: currently, if we nMC = 0, some Rcorot will be np.nan. Okay. But, if we do nMC = 1000, suddenly it gets a value.
This is because maybe 2/1000 Omegas is actually successful. How do we deal with this? Put a limit on the fraction on MC rounds that
shoud be successful? Or, if best-fit params fail, should it always fail? Difficult....
-Should I save all apers and slits as well of the MC? Think so. 
-To calculate Rcr, do method of Garma-Oehmichen instead of ours.
That method, by definition, only intersects once. 
- Problem with MC: how to determine final MC values? Depends on amount of slits etc.
We want to have system where we can just run MC, and determine limits later.
So make separate function? First, run the MC, save it, then run the limits to determine final 
values. 
-Do something similar for the _phys values. Only need to run it if we actually want it.
This will save some time!
'''



###############
### Imports ###
###############

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from photutils.aperture import EllipticalAnnulus
from photutils.aperture import RectangularAperture
from photutils.aperture import aperture_photometry
from photutils.isophote import Ellipse
from photutils.isophote import build_ellipse_model
from marvin.tools.image import Image
from scipy.ndimage import gaussian_filter
import math
from scipy.optimize import curve_fit
import marvin.utils.plot.map as mapplot
import time
import pickle
import csv

import sys
sys.path.append('/Users/geron/Documents/Projects/functions')
from mangaplots import mangacolorplot
#from get_decals_images import make_png_from_fits, dr2_style_rgb, nonlinear_map, save_carefully_resized_png, plot_decals_image, create_image_directory #no longer needed, for when plotting from DECaLS



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
    def __init__(self,PA,inc,barlen,PA_bar,maps, forbidden_labels, snr_min, PA_err,inc_err,barlen_err,PA_bar_err,func_adjust_flux,slit_width, cosmo, redshift):
        '''
        PA: position angle of galaxy, in degrees.
        inc: inclination of galaxy, in degrees
        barlength: length of the entire bar of the galaxy, in arcsec
        PA_bar: position angle of the bar, in degrees
        maps: a MaNGA maps object. If you have a MaNGA cube, can get the maps object by doing: my_cube.getMaps(bintype='VOR10').

        The PAs are defined as counterclockwise from 3 o'clock. 
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
        self.snr_min = snr_min
        bintype = maps.bintype.name
        if bintype == 'SPX':
            self.stellar_flux = func_adjust_flux(np.flip(maps['spx_mflux'],0))
        elif bintype in ['HYB10','VOR10','BIN']:
            self.stellar_flux = func_adjust_flux(np.flip(maps['bin_mflux'],0))
        self.stellar_vel = np.flip(maps['stellar_vel'],0)
        #self.radius = np.flip(maps['spx_ellcoo_r_h_kpc'],0) #not needed anymore
        self.on_sky_x = np.flip(maps['spx_ellcoo_on_sky_x'],0) #arcsec
        self.on_sky_y = np.flip(maps['spx_ellcoo_on_sky_y'],0) #arcsec
        #self.on_sky_ellcoo = np.flip(maps['spx_ellcoo_elliptical_radius'],0) #not needed anymore
        self.on_sky_xy = np.sqrt(self.on_sky_x.value**2 + self.on_sky_y.value**2)

        #Vsys correction
        self.stellar_vel, self.Vsys_corr = get_Vsys(self.stellar_vel, self.on_sky_xy, 5, forbidden_labels)
        #self.success = True #set true, if any checks fail, set this to false




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

    def save_results(self,centre, pixscale, LON, slits, apers, X_map, X_Sigma, V_Sigma, X_V, z, Omega, Omega_err, R_corot, R_corot_err, R, R_err, V_curve, V_curve_apers, V_curve_fit_params, barlen_deproj):
        self.centre = centre
        self.pixscale = pixscale
        self.LON = LON
        self.slits = slits
        self.apertures = apers
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

    def plot_hist_MC(self, variable, standalone = True, n_bins = 15):
        '''
        This plots the final posterior distribution from the MC for either Omega, R_corot or R.
        '''
        assert variable in ['Omega','Omega_phys','R_corot','R_corot_phys','R','R_phys'],"variable can either be Omega, Omega_phys, R_corot, R_corot_phys or R."
        
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
        
        if standalone:
            plt.figure(figsize = (5,5))
        
        plt.hist(var_lst,bins=n_bins)
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

        #add text
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
            mapp = self.X_Sigma.value
            title = r'X $\Sigma$'
        elif mapp == 'V_Sigma':
            mapp = self.V_Sigma.value
            title = r'V $\Sigma$'

        if standalone:
            plt.figure(figsize = (5,5))
        mangacolorplot(mapp,preset='velocities',colorbar=True, mask_keywords=self.forbidden_labels, snr_min = self.snr_min)
        
        plt.xlim(0, mapp.shape[0])
        plt.ylim(0, mapp.shape[1])

        for aper in self.V_curve_apers[1]: #are all ellipt
            plot_aper_contours(aper, aper_type = 'EllipticalAnnulus')
        plot_aper_contours(self.V_curve_apers[0]) #is a rect
        plt.title(title)

        if standalone:
            plt.show()




    def plot_maps(self, maps = ['stellar_flux','stellar_vel'], plot_LON = False, plot_slits = False, plot_apers = False, standalone = True, plot_colorbar=True):
        '''
        Plots flux and velocity maps.
        Standalone is only an option when asking for only one map to be drawn.

        TODO: If only asking for one, give option to make standalone 
        no plot_barlen()?
        '''

        if standalone == False:
            assert len(maps) == 1, "standalone = False is only an option when only asking for one map to be drawn."
        if len(maps) != 1:
            standalone = True #Standalone = False is only an option when only asking for one map to be drawn.

        n_plots = len(maps)
        for i in maps:
            assert i in ['stellar_flux','stellar_vel','X_map','X_Sigma','V_Sigma'], "Can only plot: 'stellar_flux','stellar_vel','X_Sigma' or 'V_Sigma'"

        if plot_slits:
            plot_LON = False #otherwise LON will be plotted twice

        if standalone:
            plt.figure(figsize = (5*n_plots,5))

        for i in range(n_plots):
            #determine which map to plot
            if maps[i] == 'stellar_flux':
                mapp = self.stellar_flux
                preset = 'default'
                title = 'stellar flux'
            elif maps[i] == 'stellar_vel':
                mapp = self.stellar_vel
                preset = 'velocities'
                title = 'stellar velocity'
            elif maps[i] == 'X_map':
                mapp = self.X_map
                title = r'X'
            elif maps[i] == 'X_Sigma':
                mapp = self.X_Sigma.value
                title = r'X $\Sigma$'
            elif maps[i] == 'V_Sigma':
                mapp = self.V_Sigma.value
                title = r'V $\Sigma$'

            if standalone:
                plt.subplot(1,n_plots,i+1)
            if maps[i] in ['stellar_vel','stellar_flux']:
                mangacolorplot(mapp,preset=preset,colorbar=plot_colorbar, mask_keywords = self.forbidden_labels, snr_min = self.snr_min)
            else:
                plt.pcolormesh(mapp)
                plt.colorbar()
            plt.title(title)
            plt.xlim(0, mapp.shape[0])
            plt.ylim(0, mapp.shape[1])

            #LON
            if plot_LON:
                xs_LON = np.linspace(0,mapp.shape[0],10)
                ys_LON = xs_LON * self.LON[0] + self.LON[1]
                plt.plot(xs_LON,ys_LON,color='red')

            #slits
            if plot_slits:
                for i in range(len(self.slits)):
                    xs_slit = np.linspace(0,mapp.shape[0],10)
                    ys_slit = xs_slit * self.slits[i][0] + self.slits[i][1]
                    plt.plot(xs_slit,ys_slit,color='red')

            #apers
            if plot_apers:
                for i in range(len(self.apertures)):
                    plot_aper_contours(self.apertures[i])

        if standalone:
            plt.tight_layout()
            plt.show()


    def plot_img(self, image_dir = '../output/gal_images_DECaLS/', pixscale = 0.15, n_pix = 424, plot_apers = True, plot_barlen = True, standalone = True, plot_slits=True):
        '''
        Will plot grz image. Has to option to overlay the apertures.
        Does need the custom functions we created to pull the images.

        pixscale is in arcsec/pix

        plot_slits not working
        TODO: remove image_dir arg
        '''

        
        im = Image(self.plateifu)
        im.get_new_cutout(n_pix*pixscale, n_pix*pixscale, scale=pixscale)
        ra,dec = im.ra, im.dec

        #create_image_directory(image_dir)

        if standalone:
            plt.figure()
        #img = plot_decals_image(ra,dec,image_dir, origin = 'lower', pixscale = pixscale) #old, for when picture came from DECaLS
        
        
        #img = im.plot() #cannot call im.plot(), as then it always returns new figure and the standalone arg doesn't work.
        plt.imshow(im.data, origin = 'lower')
        plt.xticks([])
        plt.yticks([])

        centre_manga = self.centre
        #centre_img = (math.ceil(img.shape[0]/2),math.ceil(img.shape[1]/2)) #old, for when picture came from DECaLS
        centre_img = (math.ceil(n_pix/2), math.ceil(n_pix/2))

        pixscale_manga = self.pixscale
        pixscale_img = pixscale 

        if plot_apers:
            for i in range(len(self.apertures)):
                aper = self.apertures[i]
                aper_temp = RectangularAperture(
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
            plt.plot([centre_img[1] + np.cos(self.PA_bar/180*np.pi)*barlen_pix/2, centre_img[1] - np.cos(self.PA_bar/180*np.pi)*barlen_pix/2 ],
                    [centre_img[0] + np.sin(self.PA_bar/180*np.pi)*barlen_pix/2, centre_img[0] - np.sin(self.PA_bar/180*np.pi)*barlen_pix/2], 
                    c='yellow')

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

def Tremaine_Weinberg(PA, inc, barlen, PA_bar, maps, 
                        slit_separation = 0, slit_width = 1, method = 'center', 
                        forbidden_labels = ['DONOTUSE','UNRELIABLE','NOCOV'], snr_min = 0,
                        PA_err = 0, inc_err = 0, barlen_err = 0, PA_bar_err = 0, 
                        n_MC = 0, corot_method = 'geron',
                        Vc = 0, correct_velcurve = True, deproject_bar = True,
                        h_method = 'individual', garma_oehmichen_correction = False,
                        cosmo = [], redshift = np.nan, velcurve_method = 'all',
                        aper_rect_width = 5, correct_xy = True, func_adjust_flux = lambda x:x,
                        print_times = False):
    '''
    Main function that user will call. Will return the TW class. 

    Inputs:
    PA: position angle of galaxy, in degrees.
    inc: inclination of galaxy, in degrees
    barlen: length of the entire bar of the galaxy, in arcsec (so not bar radius, but bar diameter!)
    PA_bar: position angle of the bar, in degrees
    maps: a MaNGA maps object. If you have a MaNGA cube, can get the maps object by doing: my_cube.getMaps(bintype='VOR10').


    Optional inputs:
    slit_separation in arcsec
    slit_width in arcsec
    method is the integration method used with the apertures. Can be either 'center' or 'exact'
    PA_err is the error (stdev) in the galaxy PA. Used to determine the error on Omega_bar. 
    n_MC is the amount of Monte Carlo loops used to determine the error on Omega_bar. Suggested to be 1000.
    if corot_method == 'Vc_userinput', then you also need to provide Vc value.
    velcurve_method can be in ['all','apers']

    The PAs are defined as counterclockwise from 3 o'clock. 

    func_adjust_flux is for testing purposes, will be applied to stellar flux map. Remove this in final version. Can only include
    operations like: +, -, *, /, or **.

    print_times is for debugging/speeding up the code

    Currently, if MC = 0, it will run once with best-guess inputs. If MC > 0, it will run the MC, Omega, Rcr and R are the
    median of the MC. After MC, it will run one last time with the best-guess inputs for all the figures etc. 
    '''


    #Part 0: initialise class and other stuff
    tw = TW(PA, inc, barlen, PA_bar, maps, forbidden_labels, snr_min, PA_err, inc_err, barlen_err, PA_bar_err, func_adjust_flux, slit_width,cosmo, redshift) 
    pixscale = get_pixscale(tw)
    centre = get_centre(tw.stellar_flux)

    # TODO in case n_MC == 0
    # Part 1 - 5 are in this MC loop
    if n_MC == 0:
        '''
        if parameters for MC are not specified, run through the loop once with the provided PA value.
        '''
        PA_err = 0
        inc_err = 0
        PA_bar_err = 0
        barlen_err = 0

    #n_MC += 1 #always add one, last one will be with normal PA, and will be used for the figures, but not saved in the MC loops
    
    for n in range(n_MC+1):
        if n == n_MC: #means we're over the MC. Doing one more round with the best-fit values for the figures.
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

        prev = time.time()

        # Part 1: Get Line of Nodes (LON)
        m_LON, b_LON = get_LON(tw.stellar_flux, PA_temp, centre)
        xs_LON = np.linspace(0,tw.stellar_flux.shape[0],10)
        ys_LON = xs_LON * m_LON + b_LON

        if print_times:
            print(f'Part I: {np.round(time.time() - prev,4)} seconds elapsed.')
        prev = time.time()

        # Part 2: Get other slits
        slits = get_pseudo_slits(tw.stellar_flux, (m_LON, b_LON), PA_temp, PA_bar_temp, barlen_temp, centre,
            pixscale = pixscale, sep = slit_separation, width_slit = slit_width, garma_oehmichen_correction = garma_oehmichen_correction)

        if print_times:
            print(f'Part II: {np.round(time.time() - prev,4)} seconds elapsed.')
        prev = time.time()


        # Part 3: Convert slits to actual apertures
        points = get_slit_centres(tw.stellar_flux,slits,centre)
        if h_method == 'individual':
            hex_map = create_hexagon_map(tw.stellar_vel,forbidden_labels)
        else:
            hex_map = []
        apers = []


        for i, s in enumerate(slits):
            aper = get_aper(points[i], slit_width/pixscale, (PA_temp-90)/180*np.pi, tw.stellar_vel, h_method = h_method, hex_map = hex_map)
            

            if aper.h > tw.stellar_vel.shape[0]/3: #ensure that it is not too short
                apers.append(aper)

        if print_times:
            print(f'Part III: {np.round(time.time() - prev,4)} seconds elapsed.')
        prev = time.time()

        # Part 4: Create X * Sigma and Vlos * Sigma maps
        X_map = create_X_map(tw.stellar_flux, slits[0], centre, pixscale)
        X_Sigma = tw.stellar_flux * X_map
        V_Sigma = tw.stellar_flux * tw.stellar_vel

        if print_times:
            print(f'Part IV: {np.round(time.time() - prev,4)} seconds elapsed.')
        prev = time.time()

        # Step 5: Do integration and determine Omega
        Xs, Vs, z, Omega = determine_pattern_speed(tw.stellar_flux, X_Sigma, V_Sigma, apers, inc_temp, method, forbidden_labels = tw.forbidden_labels, snr_min = tw.snr_min)
        Omega = np.abs(Omega)
        
        if print_times:
            print(f'Part V: {np.round(time.time() - prev,4)} seconds elapsed.')
        prev = time.time()

        # Step 6: Find V curve and corotation radius
        R_corot, vel, arcsec, V_curve_apers, V_curve_fit_params = determine_corotation_radius(Omega, tw.stellar_vel, tw.on_sky_xy, centre, PA_temp, inc_temp, maps, method, forbidden_labels = tw.forbidden_labels, corot_method = corot_method, Vc_input = Vc, correct_velcurve = correct_velcurve, velcurve_method = velcurve_method, aper_rect_width = aper_rect_width, correct_xy = correct_xy)
        delta_PA = np.abs(PA_temp - PA_bar_temp)
        if deproject_bar:
            bar_rad_deproj = barlen_temp/2 * np.sqrt(np.cos(delta_PA/180*np.pi)**2 + np.sin(delta_PA/180*np.pi)**2 / np.cos(inc_temp/180*np.pi)**2) #https://ui.adsabs.harvard.edu/abs/2007MNRAS.381..943G/abstract
            R = R_corot / bar_rad_deproj
        else:
            R = R_corot / (barlen_temp/2)

        if print_times:
            print(f'Part VI: {np.round(time.time() - prev,4)} seconds elapsed.')
        prev = time.time()

        if n < n_MC or n_MC == 0:
            tw.save_intermediate_MC_results(apers,Omega,R_corot,R,2*bar_rad_deproj,[Xs, Vs],z,[arcsec, vel],V_curve_fit_params)

    Omega = np.nanmedian(tw.Omega_lst)
    R_corot = np.nanmedian(tw.R_corot_lst)
    R = np.nanmedian(tw.R_lst)

    # Step 7: Error propogation: put in separate function later
    #https://en.wikipedia.org/wiki/Propagation_of_uncertainty
    if n_MC == 0: 
        Omega_err_ll, Omega_err_ul = np.nan, np.nan
        R_corot_err_ll, R_corot_err_ul = np.nan, np.nan
        R_err_ll, R_err_ul = np.nan, np.nan
    else: 
        # Error on Omega
        Omega_err_ul = np.nanpercentile(tw.Omega_lst,84)-Omega
        Omega_err_ll = Omega - np.nanpercentile(tw.Omega_lst,16)

        #Error on R_corot
        if corot_method != 'geron': #Then error on R_corot depends on determination of Vc and the error there
            Vc_err = np.sqrt(np.diag(V_curve_fit_params[1]))[1]
            R_corot_err_ll = np.abs(R_corot) * np.sqrt((Omega_err_ll/Omega)**2 + (Vc_err/V_curve_fit_params[0][1])**2)
            R_corot_err_ul = np.abs(R_corot) * np.sqrt((Omega_err_ul/Omega)**2 + (Vc_err/V_curve_fit_params[0][1])**2)

        if corot_method == 'geron': #Get R_corot error from MC
            R_corot_err_ul = np.nanpercentile(tw.R_corot_lst,84)-R_corot
            R_corot_err_ll = R_corot - np.nanpercentile(tw.R_corot_lst,16)

        #Error on curly R
        #barlen_err = 0 #TODO later
        #R_err_ul = np.abs(R) * np.sqrt( (barlen_err/barlen)**2 + (R_corot_err_ul/R_corot)**2)
        #R_err_ll = np.abs(R) * np.sqrt( (barlen_err/barlen)**2 + (R_corot_err_ll/R_corot)**2)
        R_err_ul = np.nanpercentile(tw.R_lst,84)-R
        R_err_ll = R - np.nanpercentile(tw.R_lst,16)

    if print_times:
        print(f'Step VII: {np.round(time.time() - prev,2)} seconds elapsed.')
    prev = time.time()

    # Part -1: Save results and return
    tw.save_results(centre, pixscale, (m_LON, b_LON), slits, apers, X_map, X_Sigma, V_Sigma, [Xs, Vs], z, Omega, (Omega_err_ll, Omega_err_ul), R_corot, (R_corot_err_ll, R_corot_err_ul), R, (R_err_ll, R_err_ul), [arcsec, vel], V_curve_apers, V_curve_fit_params, bar_rad_deproj*2) 
    return tw


    





##-------------##
## Helper defs ##
##-------------##


# Part 0: Prepare MaNGA masks

def apply_masks(mapp, forbidden_labels = ['DONOTUSE']):
    '''
    NOT IN USE ANYMORE - USING MANGA MASKS NOW. REMOVE IN NEXT VERSION
    MaNGA maps come with various labels that tell us whether to use a certain pixel. We will filter out those 
    pixels here and set those values to np.nan, instead of 0.
    '''

    for i in range(len(mapp)):
        for j in range(len(mapp[i])):
            for l in mapp[i][j].pixmask.labels:
                if l in forbidden_labels:
                    mapp.value[i][j] = np.nan

    return mapp


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
        gaussian_field = gaussian_filter(mapp.value, sigma=sigma)
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
    p2 = (p1[0] + 10 * np.cos(PA_rad), p1[1] + 10 * np.sin(PA_rad)) #10 is arbitrary, could be anything.
    
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = -m * p2[0] + p2[1]
    
    return m,b


def get_pixscale(tw):
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
    '''
    while PA > 90:
        PA -= 180
    while PA < -90:
        PA += 90
        
        
    if PA == 90 or PA == -90:
        return 1
    
    if PA < 90 and PA > 45:
        return 1/np.cos( (PA - 90)  * np.pi / 180. )
    
    if PA <= 45 and PA >= -45:
        return 1/np.cos(PA * np.pi / 180. )
    
    if PA < -45 and PA > -90:
        return 1/np.cos( (PA + 90)  * np.pi / 180. )
    
    print('Error in finding slit separation!')
    return np.nan
    

def get_pseudo_slits(mapp,LON,PA, PA_bar, barlength, centre, sep = 1, width_slit = 0, pixscale = 0.5, garma_oehmichen_correction = True):
    '''
    This will return a list of all m,b of all pseudo_slits.
    pixscale should be in arcsec/pix. barlength, sep and width_slit should be in arcsec
    
    perpendicular line needs to go through centre
    '''
    
    sep = ( sep + width_slit ) / pixscale #pixels 
    barlength = barlength / pixscale #bar length in pixels
    if garma_oehmichen_correction:
        sep = sep * get_slit_separation_correction(PA)


    
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




# Part 3: Get actual apertures
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
        aper_plot = RectangularAperture((aper.positions[0]+0.5, aper.positions[1]+0.5), 
                                        w = aper.w, h = aper.h, theta = aper.theta)


    if aper_type == 'CircularAnnulus':
        
        aper_plot = CircularAnnulus((aper.positions[0]+0.5, aper.positions[1]+0.5),
                                     r_in = aper.r_in, r_out = aper.r_out)

    
    if aper_type == 'EllipticalAnnulus':
        
        aper_plot = EllipticalAnnulus((aper.positions[0]+0.5, aper.positions[1]+0.5),
                                     a_in = aper.a_in, a_out = aper.a_out,
                                     b_in = aper.b_in, b_out = aper.b_out,
                                     theta = aper.theta)

    aper_plot.plot(color=color, ls = ls, alpha = alpha)

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


def get_aper(slit_centre, slit_width, slit_theta, stellar_vel, h_method = 'individual', hex_map = []):
    '''
    Will convert slit into an actual aperture.

    h_method can be in ['individual', 'max']. With max, it'll just take h = sqrt(2)*map.shape[0].
    With individual, it'll create a length for every slit while taking the edges of the hexagon into account.
    Will keep on increasing height of the aperture until it hits the end. That's Lmax for this slit.
    This is more accurate, but also slower.

    Units should be in pixels. 
    '''

    if h_method == 'individual':
        '''
        Before, the algorithm started with an h of 0, and added one pixel to it at the time. However, this took a while.
        '''

        """
        h = 2
        h_stepsize = 1 #can edit this
        temp = 0.0

        n_loops = 0
    
        while temp == 0.0:
            aper = RectangularAperture(slit_centre, w = slit_width, h = h, theta = slit_theta) #think I need to put h+=h_stepsize first
            temp = float(aperture_photometry(hex_map, aper, method = 'center')['aperture_sum'])
            h+=h_stepsize
            n_loops+=1
        
        h-=h_stepsize
        aper = RectangularAperture(slit_centre, w = slit_width, h = h, theta = slit_theta)
        #add a minimum aper length value?
        """

        #algorithm will now take big steps steps first, until it overshoots. Then reduce stepsize to finetune
        #this is done for speeding up the code. 
        h = 1

        h_stepsizes = [20,5,1]

        aper = RectangularAperture(slit_centre, w = slit_width, h = h, theta = slit_theta)
        reached_hex = float(aperture_photometry(hex_map, aper, method = 'center')['aperture_sum'])

        for h_stepsize in h_stepsizes:

            reached_hex = 0.0 
            while reached_hex == 0.0:
                h+=h_stepsize
                aper = RectangularAperture(slit_centre, w = slit_width, h = h, theta = slit_theta)
                reached_hex = float(aperture_photometry(hex_map, aper, method = 'center')['aperture_sum'])

            h-=h_stepsize

        aper = RectangularAperture(slit_centre, w = slit_width, h = h, theta = slit_theta)    
    
    if h_method == 'max':
        aper = RectangularAperture(slit_centre, w = slit_width, h = stellar_vel.shape[0]*np.sqrt(2), theta = slit_theta)
    
    return aper




# Part 4: Create X * Sigma and Vlos * Sigma maps

def is_left_of(line,point):
    x = (point[1] - line[1])/line[0]
    
    if x >= point[0]:
        return -1
    else:
        return 1

def create_X_map(mapp, LON, centre, sep):
    empty = np.full_like(mapp.value,np.nan)

    m_pLON = -1/LON[0] #m1 * m2 = -1 for perpendicular lines
    b_pLON = -m_pLON * centre[0] + centre[1]
    
    for i in range(len(empty)):
        for j in range(len(empty[i])):
            #find distance between i,j and line
            # See https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
            
            a, b, c = m_pLON, -1, b_pLON
            dist = np.abs(a*j+b*i+c)/np.sqrt(a**2+b**2)
            
            empty[i][j] = is_left_of([m_pLON, b_pLON],[j,i])*dist*sep
    
    return empty




# Step 5: Do integration and determine Omega

def determine_pattern_speed(stellar_flux, X_Sigma, V_Sigma, apers, inc, method, forbidden_labels = ['DONOTUSE'], snr_min = 0):
    Xs = []
    Vs = []

    for aper in apers:#can I ignore the snr and forbidden_labels here?
        #phot_table_flux = aperture_photometry(stellar_flux.value, aper, method = method, mask = ( mapplot.mask_low_snr(stellar_flux.value, stellar_flux.ivar, snr_min = snr_min)))
        phot_table_flux = aperture_photometry(stellar_flux.value, aper, method = method)
        flux = float(phot_table_flux['aperture_sum'])

        if flux == 0:
            continue #just ignore
        else: #can I ignore the snr and forbidden_labels here?
            #phot_table_X = aperture_photometry(X_Sigma.value, aper, method = method, mask = ( (X_Sigma.pixmask.get_mask(forbidden_labels)>=1) | mapplot.mask_low_snr(X_Sigma.value, X_Sigma.ivar, snr_min = snr_min)))
            #phot_table_X = aperture_photometry(X_Sigma.value, aper, method = method, mask = ( (X_Sigma.pixmask.get_mask(forbidden_labels)>=1)))
            phot_table_X = aperture_photometry(X_Sigma.value, aper, method = method)
            

            Xs.append(float(phot_table_X['aperture_sum'])/flux)
            
            #phot_table_V = aperture_photometry(V_Sigma.value, aper, method = method, mask = ( (V_Sigma.pixmask.get_mask(forbidden_labels)>=1) | mapplot.mask_low_snr(V_Sigma.value, V_Sigma.ivar, snr_min = snr_min)))
            phot_table_V = aperture_photometry(V_Sigma.value, aper, method = method)
            
            Vs.append(float(phot_table_V['aperture_sum'])/flux)


    # Get the fit
    if len(Xs) > 1:
        z = np.polyfit(Xs,Vs,1, full=True)
        Omega = z[0][0]/np.sin(inc/180*np.pi)
    else:
        z = (np.array([np.nan, np.nan]), np.array([]), np.nan, np.array([]), np.array([])) #so it has same shape as a normal z
        if len(Xs) == 1:
            Omega = Vs[0]/Xs[0]/np.sin(inc/180*np.pi)
        else:
            Omega = np.nan

    return Xs, Vs, z, Omega




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
    
    phot_table = aperture_photometry(ones, aper, method = method)
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


def determine_corotation_radius(Omega, stellar_vel, on_sky_xy, centre, PA, inc, maps, method, forbidden_labels = ['DONOTUSE'], corot_method = 'geron', Vc_input = 0, correct_velcurve = True, velcurve_method = 'apers', aper_rect_width = 10, correct_xy = True):
    '''
    TODO: change the method names etc

    Corot_method determiens how the corotation radius is calculated. Either:
    -fit two parameter arctan function, take Vflat, use that. 
    -create velocity profile from stellar velocity map, see where it intersects. 
    '''
        
    # Find the rectangular aperature
    aper_rect = RectangularAperture(centre, w = aper_rect_width, h = stellar_vel.shape[0]*1.5, theta = (PA-90)/180*np.pi)


    # apply the rect aperature first
    stellar_vel_rect = aper_rect.to_mask(method = method).to_image(shape = (stellar_vel.shape)) * stellar_vel.value * (stellar_vel.pixmask.get_mask(forbidden_labels)<1).astype(int) #last bit is to incorporate stellar_vel mask
    on_sky_xy_rect = aper_rect.to_mask(method = method).to_image(shape = (on_sky_xy.shape)) * on_sky_xy
    
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
                if not correct_xy:
                    corr_factor_xy = 1

                #print(f'Corr factors: {corr_factor_xy}, {corr_factor_vel}')
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

    if velcurve_method == 'apers':
        #Find the elliptical annuli
        n_pix = centre[0] #amount of pixels in one radius length
        n_bins = int(stellar_vel.shape[0]/4) #kind of arbitrary. For 76x76 image, it'll give 20 bins
        binsize_pix = n_pix / n_bins
        apers_circ = []
        a_in = 0.1 #cannot be 0
        eps = 1 - np.cos(inc/180*np.pi) # eps = 1- b/a = 1 - cos(i)
        #a is semimajor axis, b is semiminor axis
        for i in range(n_bins):
            a_out = a_in + binsize_pix
            b_in = a_in * (1 - eps)
            b_out = a_out * (1 - eps)
            aper = EllipticalAnnulus(centre, a_in = a_in, a_out = a_out, b_out = b_out, 
                                    b_in = b_in, theta = (PA)/180*np.pi)
            apers_circ.append(aper) 
            a_in = a_out

        for aper in apers_circ:
            n_pix_vel = get_n_pix_in_aper(stellar_vel_rect_corr, aper, method = method)
            n_pix_arcsec = get_n_pix_in_aper(on_sky_xy_rect_corr, aper, method = method)
            if n_pix_vel >= 10 and n_pix_arcsec >= 10 :
                phot_table = aperture_photometry(np.abs(stellar_vel_rect_corr), aper, method = method)
                vel.append(float(phot_table['aperture_sum']/n_pix_vel))

                phot_table = aperture_photometry(on_sky_xy_rect_corr, aper, method = method)
                arcsec.append(float(phot_table['aperture_sum']/n_pix_arcsec))

    if velcurve_method == 'all':
        #stellar_vel_mask = aper_rect.to_mask(method = method).to_image(shape = (stellar_vel.shape))
        #on_sky_xy_mask = aper_rect.to_mask(method = method).to_image(shape = (on_sky_xy.shape))
        apers_circ = []
        #print(np.array(stellar_vel_rect_corr).shape)
        for i in range(len(stellar_vel_rect_corr)):
            for j in range(len(stellar_vel_rect_corr[i])):
                if stellar_vel_rect_corr[i][j] != 0.0 and on_sky_xy_rect_corr[i][j] != 0.0: #0.0 can be either because of outside rect apers or because no data.
                    vel.append(np.abs(stellar_vel_rect_corr[i][j]))
                    arcsec.append(on_sky_xy_rect_corr[i][j])


    # Fit profile with basic 2 param arctan function
    if len(arcsec) > 0 and len(vel) > 0:
        popt, pcov = curve_fit(velFunc, arcsec, vel, bounds = (0, np.inf))
        MSE = np.sum( (vel - velFunc(arcsec,popt[0],popt[1]))**2)

        if corot_method == 'geron':

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


        if corot_method == 'Vc':
            R_corot =  np.abs(popt[0] / Omega)

        if corot_method == 'Vc_userinput':
            popt[0] = Vc_input
            R_corot =  np.abs(popt[0] / Omega)

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
        fileloc = str(int(time.time()))+'.csv'
        
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