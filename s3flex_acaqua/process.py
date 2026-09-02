
import os, shutil
import glob

import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd

import datetime as dt

import radcalnet_oc as radoc
import coxmunk

class Process():
    def __init__(self):
        pass

    def execute(self,
                 l1_pt,
                wl_min=350,
                wl_max = 1000,
                expon = 2,
                spectral_res = 0.1,

                # set aerosol optical thickness
                aot_refs = None, # 0.8*aot_ref.values,aot_ref.values,1.2*aot_ref.values]
                # ['ARCT_rh70', 'COAV_rh70', 'DESE_rh70', 'MACL_rh70', 'URBA_rh70', 'WASO_rh0']
                aerosol_combination = [1., 0., 0., 0., 0., 0.],


                # sunglint parameters
                wind_azi=0,
                stats = 'cm_iso',  # 'bh2006' #
                shadow = False,
                slope = False

                 ):

        ##################################################
        # Input parameters
        ##################################################

        wl_ref = 550
        ang_exp = l1_pt.A865
        aot_ref = l1_pt.T865 * (wl_ref / 865) ** -ang_exp
        if aot_refs is None:
            aot_refs=[aot_ref]
        wind = np.sqrt(
            l1_pt.horizontal_wind.isel(wind_vectors=0) ** 2 + l1_pt.horizontal_wind.isel(wind_vectors=1) ** 2).values
        pressure = float(l1_pt.sea_level_pressure.values)
        tco3 = float(l1_pt.total_ozone.values)
        tcwv = float(l1_pt.total_columnar_water_vapour.values)
        fwhm = l1_pt.fwhm.values
        sza = float(l1_pt.sza.values)  # np.arange(10,90,10)
        vza = float(l1_pt.vza.values)  # [0,20,40]
        # warning LUT implemented until VZA=48.92
        # take 48.91 if vza >48.92
        vza = np.min([vza, 48.91])
        azi = float(l1_pt.raa.values)
        azi_rt = (180 - azi) % 360
        szas = [sza]  # np.arange(10,90,10)
        vzas = [vza]  # [0,20,40]
        azis = [azi]
        tcno2 = 3e-6 / 100
        tcch4 = 1e-2

        ##################################################
        self.params = r'$sza= %.1f$' % (sza) + \
                 '\n$vza= %.1f$' % (vza) + \
                 '\n$razi= %.1f$' % (azi) + \
                 '\n$aot550 = %.3f$' % (aot_ref) + \
                 r'\n$ang.\ exp. = %.3f$' % (ang_exp) + \
                 '\n$pressure = %.1f$' % (pressure) + \
                 '\n$wind = %.1f$' % (wind) + \
                 '\n$tcwv = %.1f$' % (tcwv) + \
                 '\n$tco3 = %.1e$' % (tco3)

        # -------------------------------------
        # LUT preparation
        # -------------------------------------
        central_wl = np.arange(wl_min, wl_max, spectral_res)
        spectral = radoc.Spectral(central_wl, 0.3)
        lut = radoc.LUT(wl=central_wl)
        lut.load_auxiliary_data()
        lut.lut_preparation(aot_refs=aot_refs, sza=szas, vza=vzas, azi=[azi_rt],
                            aerosol_combination=aerosol_combination)
        gas_trans = radoc.GaseousTransmittance(lut.gas_lut)

        #-------------------------------------
        # simple process
        # -------------------------------------

        szas = lut.Rdiff_lut.sza
        mu0 = np.cos(np.radians(szas))

        # set total transmittance values
        gas_trans = radoc.GaseousTransmittance(lut.gas_lut)
        gas_trans.pressure = pressure
        gas_trans.gas_tc['h2o'] = tcwv
        gas_trans.gas_tc['o3'] = tco3
        gas_trans.gas_tc['ch4'] = tcch4

        # for diffuse radiance
        gas_trans.coef_abs_scat['h2o'] = 0.5
        gas_trans.coef_abs_scat['ch4'] = 1
        gas_trans.coef_abs_scat['o2'] = 0.25

        gas_trans.air_mass = 1. / mu0
        Tg_d = gas_trans.get_gaseous_transmittance()

        gas_trans.air_mass = 1. / np.cos(np.radians(vzas))
        Tg_u = gas_trans.get_gaseous_transmittance()

        # for total path
        gas_trans.coef_abs_scat['h2o'] = 1
        gas_trans.coef_abs_scat['ch4'] = 1
        gas_trans.coef_abs_scat['o2'] = 1

        gas_trans.air_mass = 1. / mu0
        Tg_tot_d = gas_trans.get_gaseous_transmittance()

        gas_trans.air_mass = 1. / np.cos(np.radians(vzas))
        Tg_tot_u = gas_trans.get_gaseous_transmittance()

        # get solar irradiance
        solar_irr = radoc.SolarIrradiance()
        signal = solar_irr.tsis.sel(wl=slice(350, 2500))
        E0 = signal * mu0
        wl_common = E0.wl.values

        F0 = spectral.convolve2(solar_irr.tsis, expon=expon)
        E0 = mu0 * F0
        Tg_d = spectral.convolve2(Tg_d, expon=expon).to_dataarray().squeeze(drop=True)
        Tg_u = spectral.convolve2(Tg_u, expon=expon)
        Tg_tot_d = spectral.convolve2(Tg_tot_d, expon=expon).to_dataarray().squeeze(drop=True)
        Tg_tot_u = spectral.convolve2(Tg_tot_u, expon=expon)

        # get transmittance Rayleigh + aerosols
        Tra_d = lut.trans_Ed.interp(wl=central_wl, method='quadratic')
        Tra_u = lut.trans_Eu.interp(wl=central_wl, method='quadratic')
        tra_u = lut.trans_Lu.interp(wl=central_wl, method='quadratic')

        # get transmittance Rayleigh + aerosols for direct light (e.g., sunglint)
        aot = lut.aot_lut.interp(wl=central_wl, method='quadratic')
        rot = lut.rot.interp(wl=central_wl, method='quadratic') * pressure / 1013.25
        Tra_dir_d = np.exp(-(rot + aot) / mu0)
        Tra_dir_u = np.exp(-(rot + aot) / np.cos(np.radians(vzas)))

        # get top-of-atmosphere reflectance or radiance
        self.Ratm = Tg_d * Tg_u * lut.Rdiff_lut.squeeze().interp(wl=central_wl, method='quadratic')
        Ed = Tra_d * Tg_d * E0
        Ed_ref = Ed.sel(wl=slice(400, 700)).mean()

        # get sunglint component
        Iglint, Qglint, Uglint, Vglint = radoc.Sunglint(sza, vza, azi, m=1.334).sunglint(
            wind, wind_azi, stats=stats, shadow=shadow, slope=slope)

        self.Rtoa_sunglint = Iglint * Tg_tot_d * Tg_tot_u * Tra_dir_d * Tra_dir_u

        # -------------------------------------
        # Spectral convolution and final step
        # -------------------------------------
        fwhm = l1_pt.fwhm.values

        central_wl = l1_pt.wl  # .values #np.arange(wl_min, wl_max,spectral_res)
        spectral = radoc.Spectral(central_wl, fwhm)
        self.Ratm_sat = spectral.convolve(self.Ratm)
        #Tg_u_sat = spectral.convolve(Tg_tot_u)
        #Tg_d_sat = spectral.convolve(Tg_d)
        self.Rtoa_sunglint_sat = spectral.convolve(self.Rtoa_sunglint.squeeze())
        self.Rrs = l1_pt.Rtoa - (self.Rtoa_sunglint_sat + self.Ratm_sat)

        return

