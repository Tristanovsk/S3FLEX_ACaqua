'''
Atmospheric correction of a FLEX (or Sentinel-3 FLEX-mode) pixel above water from the atmospheric parameters
of the Sentinel-3 L2A products, with the radiative transfer simulator radcalnet_oc (version >= 1.1).

The top-of-atmosphere reflectance is decomposed as (see the ATBD and https://radcalnet-oc.readthedocs.io)

    R_toa = T_g^atm R_atm + T_dir R_sg + pi T_down t_up T_g^w R_rs

with R_atm the diffuse reflectance of the atmosphere and of the sky reflected by the rough surface (OSOAA look-up
tables), R_sg the Cox-Munk reflectance of the direct sunlight (sunglint), T_dir the direct transmittance of the
Sun-surface-sensor path (gases, molecules, aerosols), T_down and t_up the diffuse transmittances, and T_g^atm,
T_g^w the gaseous transmittances of the path reflectance and of the water-leaving term, coupled with the
scattering (photon path-length model of radcalnet_oc 1.1, fitted on SOS-ABS simulations).
'''

import numpy as np
import xarray as xr

import radcalnet_oc as radoc
from radcalnet_oc.kernel import GasCoupling, GAS_COUPLING_FILE, GAS_COUPLING_PPLD_FILE

#: largest viewing zenith angle of the look-up tables (deg)
VZA_MAX = 48.91
#: OPAC aerosol models of the look-up tables
AEROSOL_MODELS = ['ARCT_rh70', 'COAV_rh70', 'DESE_rh70', 'MACL_rh70', 'URBA_rh70', 'WASO_rh0']
COUPLING_TABLES = {'path_length': GAS_COUPLING_PPLD_FILE, 'effective_air_mass': GAS_COUPLING_FILE}


class Process():
    '''
    Atmospheric correction of one pixel.

    :param gas_coupling: gaseous transmittance of the scattered light, as in radcalnet_oc.Process:
        'path_length' (default), 'effective_air_mass' or 'decoupled' (direct-beam transmittance for all the terms)
    :param o2_model: absorption of O2, 'lbl' (line by line in the O2-B and O2-A bands, default) or 'reptran'

    After :py:meth:`execute`, the attributes are (dimension ``aot_ref`` for the aerosol optical thicknesses
    simulated, ``wl`` the wavelengths of the hyperspectral simulation or of the sensor bands for ``*_sat``):

    - ``Ratm``, ``Rtoa_sunglint``: atmospheric reflectance T_g^atm R_atm and sunglint T_dir R_sg at the TOA
      (hyperspectral), and ``Ratm_sat``, ``Rtoa_sunglint_sat`` in the sensor bands;
    - ``trans_w``, ``trans_w_sat``: pi T_down t_up T_g^w, from R_rs to the water term at the TOA;
    - ``Rw_toa``: water-leaving term at the TOA, R_toa - R_atm - R_sunglint (sensor bands);
    - ``Rrs``: remote-sensing reflectance (sr-1, sensor bands);
    - ``Ed``: downwelling irradiance at the surface (mW m-2 nm-1, hyperspectral) and ``PAR``
      (umol photons m-2 s-1);
    - ``Rsg``: sunglint reflectance at the surface; ``params``: text of the input parameters for the figures.
    '''

    def __init__(self, gas_coupling='path_length', o2_model='lbl'):
        if gas_coupling not in ('path_length', 'effective_air_mass', 'decoupled'):
            raise ValueError(f'unknown gas_coupling {gas_coupling!r}')
        self.gas_coupling = gas_coupling
        self.o2_model = o2_model

    def execute(self,
                l1_pt,
                wl_min=350,
                wl_max=1000,
                expon=2,
                spectral_res=0.1,
                fwhm_hyp=0.3,
                # aerosol optical thicknesses at 550 nm to simulate (default: that of the L2A product)
                aot_refs=None,
                # proportions of the aerosol models, see AEROSOL_MODELS
                aerosol_combination=[1., 0., 0., 0., 0., 0.],
                # sunglint parameters
                wind_azi=0,
                stats='cm_iso',  # 'bh2006', 'cm_dir'
                shadow=False,
                slope=False):
        '''
        :param l1_pt: pixel of the co-registered L1 (``Rtoa``, ``wl``, ``fwhm``, geometry ``sza``, ``vza``,
            ``raa``) and Sentinel-3 L2A (``T865``, ``A865``, ``horizontal_wind``, ``sea_level_pressure``,
            ``total_ozone``, ``total_columnar_water_vapour``) products
        :param wl_min, wl_max: spectral range of the hyperspectral simulation (nm)
        :param expon: exponent of the super-Gaussian responses of the hyperspectral simulation (2: Gaussian)
        :param spectral_res: sampling of the hyperspectral simulation (nm)
        :param fwhm_hyp: full width at half maximum of the hyperspectral simulation (nm)
        '''
        # ------------------------------------------------------------------------------------------------
        # input parameters
        # ------------------------------------------------------------------------------------------------
        wl_ref = 550
        ang_exp = float(l1_pt.A865)
        aot_ref = float(l1_pt.T865) * (wl_ref / 865) ** -ang_exp
        if aot_refs is None:
            aot_refs = [aot_ref]
        aot_refs = np.atleast_1d(np.asarray(aot_refs, dtype=float))
        wind = float(np.sqrt(l1_pt.horizontal_wind.isel(wind_vectors=0) ** 2 +
                             l1_pt.horizontal_wind.isel(wind_vectors=1) ** 2))
        pressure = float(l1_pt.sea_level_pressure.values)
        tco3 = float(l1_pt.total_ozone.values)
        tcwv = float(l1_pt.total_columnar_water_vapour.values)
        sza = float(l1_pt.sza.values)
        # the look-up tables stop at VZA = 48.92 deg
        vza = min(float(l1_pt.vza.values), VZA_MAX)
        azi = float(l1_pt.raa.values)
        # azimuth of the radiative transfer convention of the look-up tables (0: towards the Sun)
        azi_rt = (180 - azi) % 360
        day_of_year = int(l1_pt.time.dt.dayofyear) if 'time' in l1_pt.coords else 172
        tcno2 = 3e-6 / 100
        tcch4 = 1e-2
        self.inputs = dict(sza=sza, vza=vza, azi=azi, aot550=aot_ref, ang_exp=ang_exp, pressure=pressure,
                           wind=wind, tcwv=tcwv, tco3=tco3, day_of_year=day_of_year)
        self.params = (r'$sza= %.1f$' % sza + '\n$vza= %.1f$' % vza + '\n$razi= %.1f$' % azi +
                       '\n$aot550 = %.3f$' % aot_ref + '\n$ang.\\ exp. = %.3f$' % ang_exp +
                       '\n$pressure = %.1f$' % pressure + '\n$wind = %.1f$' % wind +
                       '\n$tcwv = %.1f$' % tcwv + '\n$tco3 = %.1e$' % tco3)

        # ------------------------------------------------------------------------------------------------
        # look-up tables: diffuse reflectance and transmittances (wind speed of the L2A product)
        # ------------------------------------------------------------------------------------------------
        central_wl = np.arange(wl_min, wl_max, spectral_res)
        spectral = radoc.Spectral(central_wl, fwhm_hyp)
        lut = radoc.LUT(wl=central_wl)
        lut.load_auxiliary_data()
        lut.lut_preparation(wind=wind, aot_refs=aot_refs, sza=[sza], vza=[vza], azi=[azi_rt],
                            aerosol_combination=aerosol_combination)
        self.lut = lut
        mu0, muv = np.cos(np.radians(sza)), np.cos(np.radians(vza))
        m0, m = 1. / mu0, 1. / mu0 + 1. / muv
        aot = xr.DataArray(aot_refs, coords={'aot_ref': aot_refs}, dims='aot_ref')

        margin = 5 * fwhm_hyp + 1.

        def hyp(signal):
            '''Signal at the resolution of the hyperspectral simulation (cropped first, to limit the memory).'''
            signal = signal.sel(wl=slice(wl_min - margin, wl_max + margin))
            out = spectral.convolve2(signal.rename('signal'), expon=expon)
            out = out[list(out.data_vars)[0]] if isinstance(out, xr.Dataset) else out
            return out.squeeze(drop=True)

        # ------------------------------------------------------------------------------------------------
        # gaseous transmittances
        # ------------------------------------------------------------------------------------------------
        gas = radoc.GaseousTransmittance(lut.gas_lut, o2_model=self.o2_model)
        gas.pressure = pressure
        gas.gas_tc['h2o'] = tcwv
        gas.gas_tc['o3'] = tco3
        gas.gas_tc['ch4'] = tcch4
        gas.gas_tc['no2'] = tcno2
        # direct beam of the Sun-surface-sensor path (sunglint)
        gas.air_mass = m
        Tg_dir = gas.get_gaseous_transmittance()
        if self.gas_coupling == 'decoupled':
            Tg_atm = Tg_w = Tg_dir
            gas.air_mass = m0
            Tg_Ed = gas.get_gaseous_transmittance()
        else:
            coupling = GasCoupling(COUPLING_TABLES[self.gas_coupling])
            Tg_atm = gas.get_coupled_transmittance(m, 'atm', aot, coupling)
            Tg_w = gas.get_coupled_transmittance(m, 'w', aot, coupling)
            Tg_Ed = gas.get_coupled_transmittance(m0, 'Ed', aot, coupling)
        Tg_dir, Tg_atm, Tg_w, Tg_Ed = (hyp(T) for T in (Tg_dir, Tg_atm, Tg_w, Tg_Ed))
        self.Tg = xr.Dataset({'Tg_dir': Tg_dir, 'Tg_atm': Tg_atm, 'Tg_w': Tg_w, 'Tg_Ed': Tg_Ed})

        # ------------------------------------------------------------------------------------------------
        # solar irradiance and transmittances of the Rayleigh-aerosol atmosphere
        # ------------------------------------------------------------------------------------------------
        solar_irr = radoc.SolarIrradiance()
        # on the grid of the gaseous absorption table: one weight matrix for all the convolutions
        F0 = hyp(solar_irr.tsis.interp(wl=gas.gas_lut.wl.sel(wl=slice(wl_min - margin, wl_max + margin))))
        E0 = mu0 * radoc.Misc.earth_sun_correction(day_of_year) * F0
        Tra_d = lut.trans_Ed.interp(wl=central_wl, method='quadratic').squeeze(drop=True)
        tra_u = lut.trans_Lu.interp(wl=central_wl, method='quadratic').squeeze(drop=True)
        tau_a = lut.aot_lut.interp(wl=central_wl, method='quadratic').squeeze(drop=True)
        tau_r = lut.rot.interp(wl=central_wl, method='quadratic') * pressure / 1013.25
        Tdir = Tg_dir * np.exp(-(tau_r + tau_a) * m)

        # ------------------------------------------------------------------------------------------------
        # terms of the TOA reflectance and downwelling irradiance (hyperspectral)
        # ------------------------------------------------------------------------------------------------
        Rdiff = lut.Rdiff_lut.squeeze(drop=True).interp(wl=central_wl, method='quadratic')
        self.Ratm = Tg_atm * Rdiff
        Rsg = radoc.Sunglint(sza, vza, azi, m=1.334).sunglint(wind, wind_azi, stats=stats, shadow=shadow,
                                                             slope=slope)[0]
        self.Rsg = max(float(Rsg), 0.)
        self.Rtoa_sunglint = Tdir * self.Rsg
        self.trans_w = np.pi * Tra_d * tra_u * Tg_w
        self.E0 = E0
        self.Ed = Tra_d * Tg_Ed * E0
        PAR = np.atleast_1d(radoc.Radiometry().PAR(self.Ed))
        self.PAR = (xr.DataArray(PAR, coords={'aot_ref': aot_refs}, dims='aot_ref') if PAR.size == aot_refs.size
                    else xr.DataArray(PAR.squeeze())).rename('PAR')

        # ------------------------------------------------------------------------------------------------
        # sensor bands and water-leaving signal
        # ------------------------------------------------------------------------------------------------
        sat = radoc.Spectral(l1_pt.wl, l1_pt.fwhm.values)

        def bands(signal):
            out = sat.convolve(signal)
            if isinstance(out, xr.Dataset):
                out = out[list(out.data_vars)[0]]
            return out.squeeze('variable', drop=True) if 'variable' in out.dims else out

        self.Ratm_sat = bands(self.Ratm)
        self.Rtoa_sunglint_sat = bands(self.Rtoa_sunglint)
        self.trans_w_sat = bands(self.trans_w)
        self.Rw_toa = l1_pt.Rtoa - (self.Ratm_sat + self.Rtoa_sunglint_sat)
        self.Rrs = self.Rw_toa / self.trans_w_sat
        return self
