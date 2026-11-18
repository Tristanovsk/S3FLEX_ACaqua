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

Performance: everything that does not depend on the pixel (look-up tables, coupling tables, solar irradiance,
spectral convolution operators) is built once and reused by the following calls of :py:meth:`Process.execute`
(see :py:class:`SpectralOperator`), so that only the first call pays the set-up.
'''

import numpy as np
import scipy.sparse as sp
import xarray as xr

import radcalnet_oc as radoc
from radcalnet_oc.kernel import GasCoupling, GAS_COUPLING_FILE, GAS_COUPLING_PPLD_FILE

#: largest viewing zenith angle of the look-up tables (deg)
VZA_MAX = 48.91
#: OPAC aerosol models of the look-up tables
AEROSOL_MODELS = ['ARCT_rh70', 'COAV_rh70', 'DESE_rh70', 'MACL_rh70', 'URBA_rh70', 'WASO_rh0']
COUPLING_TABLES = {'path_length': GAS_COUPLING_PPLD_FILE, 'effective_air_mass': GAS_COUPLING_FILE}


class SpectralOperator():
    '''
    Spectral convolution of signals on a fixed wavelength grid, as a sparse matrix product.

    The weights are those of :py:meth:`radcalnet_oc.Spectral.weights` (same responses, threshold, trapezoidal
    rule and normalization), computed once by blocks of bands so that the dense matrix of all the bands is never
    built (15563 x 6500 = 0.8 GB for the hyperspectral simulation, 99.8 % zeros), and stored in CSR format.

    A band value is NaN when the signal is NaN at a wavelength where the response of the band is not null, or
    when the response does not overlap the signal, as in :py:meth:`radcalnet_oc.Spectral.convolve_nd`. The
    computation is in float64 (radcalnet_oc returns float32).

    :param central_wl: central wavelengths of the bands (nm)
    :param fwhm: full width at half maximum of the bands (nm), scalar or array
    :param wl_signal: wavelengths of the signals to convolve (nm)
    :param expon: exponent of the super-Gaussian responses (None: Gaussian)
    :param threshold: the responses below this fraction of their maximum are ignored (None: whole responses)
    :param block: number of bands per block of the computation of the weights
    '''

    def __init__(self, central_wl, fwhm, wl_signal, expon=None, threshold=None, block=500):
        central_wl = np.atleast_1d(np.asarray(central_wl, dtype=np.float64))
        fwhm = np.broadcast_to(np.asarray(fwhm, dtype=np.float64), central_wl.shape)
        self.wl_signal = np.asarray(wl_signal, dtype=np.float64)
        self.band_wl = central_wl
        columns, invalid = [], []
        for start in range(0, central_wl.size, block):
            stop = start + block
            weights = radoc.Spectral(central_wl[start:stop], np.array(fwhm[start:stop])).weights(
                self.wl_signal, expon=expon, threshold=threshold)
            # bands without overlap with the signal: NaN columns in radcalnet_oc
            invalid.append(np.isnan(weights).any(axis=0))
            columns.append(sp.csc_matrix(np.nan_to_num(weights, nan=0.)))
        self.invalid = np.concatenate(invalid)
        matrix = sp.hstack(columns, format='csr')
        matrix.eliminate_zeros()
        #: weights (wavelength of the signal, band), CSR
        self.matrix = matrix
        #: transposed weights (band, wavelength of the signal), CSR, used for the products
        self.matrix_t = matrix.T.tocsr()

    def apply(self, values):
        '''Convolution of an array with the wavelengths of the signal on the last axis.'''
        values = np.asarray(values, dtype=np.float64)
        shape = values.shape
        flat = values.reshape(-1, shape[-1])
        # sparse product: a NaN only reaches the bands whose response is not null at its wavelength
        out = (self.matrix_t @ flat.T).T
        out[:, self.invalid] = np.nan
        return out.reshape(shape[:-1] + (self.band_wl.size,))

    def __call__(self, signal):
        '''
        :param signal: :py:class:`xarray.DataArray` with a ``wl`` dimension on the wavelengths of the operator
        :return: :py:class:`xarray.DataArray` with the central wavelengths of the bands as ``wl`` coordinate
        '''
        if not np.array_equal(signal.wl.values, self.wl_signal):
            raise ValueError('signal not on the wavelengths of the operator')
        signal = signal.reset_coords(drop=True)
        out = xr.apply_ufunc(self.apply, signal, input_core_dims=[['wl']], output_core_dims=[['wl_band']],
                             dask='parallelized', output_dtypes=[np.float64],
                             dask_gufunc_kwargs={'output_sizes': {'wl_band': self.band_wl.size}})
        return out.rename(wl_band='wl').assign_coords(wl=self.band_wl)


# pixel-independent resources, shared by all the instances of Process (keys: grids and options)
_CACHE = {}


def _cached(key, build):
    if key not in _CACHE:
        _CACHE[key] = build()
    return _CACHE[key]


def clear_cache():
    '''Free the look-up tables and the operators kept between the calls of :py:meth:`Process.execute`.'''
    _CACHE.clear()


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
    - ``Rsg``: sunglint reflectance at the surface and ``Tdir_sat`` the direct transmittance in the sensor bands
      (``Rtoa_sunglint_sat = Tdir_sat * Rsg``); ``params``: text of the input parameters for the figures.

    The look-up tables and the spectral operators are built at the first call of :py:meth:`execute` and kept
    for the following calls (also by the other instances, see :py:func:`clear_cache`). The instances are not
    thread-safe (one instance per thread or process).
    '''

    def __init__(self, gas_coupling='path_length', o2_model='lbl'):
        if gas_coupling not in ('path_length', 'effective_air_mass', 'decoupled'):
            raise ValueError(f'unknown gas_coupling {gas_coupling!r}')
        self.gas_coupling = gas_coupling
        self.o2_model = o2_model
        self._luts = {}

    def _lut(self, central_wl):
        '''Look-up tables on the hyperspectral grid, loaded once per instance and grid.'''
        key = central_wl.tobytes()
        if key not in self._luts:
            self._luts[key] = radoc.LUT(wl=central_wl)
        return self._luts[key]

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
        lut = self._lut(central_wl)
        lut.lut_preparation(wind=wind, aot_refs=aot_refs, sza=[sza], vza=[vza], azi=[azi_rt],
                            aerosol_combination=aerosol_combination)
        self.lut = lut
        mu0, muv = np.cos(np.radians(sza)), np.cos(np.radians(vza))
        m0, m = 1. / mu0, 1. / mu0 + 1. / muv
        aot = xr.DataArray(aot_refs, coords={'aot_ref': aot_refs}, dims='aot_ref')

        # gaseous absorption grid (cropped first, to limit the memory) and convolution to the hyperspectral grid
        margin = 5 * fwhm_hyp + 1.
        wl_gas = lut.gas_lut.wl.sel(wl=slice(wl_min - margin, wl_max + margin)).values
        hyp_key = ('hyp', central_wl.tobytes(), float(fwhm_hyp), wl_gas.tobytes(), expon)
        hyp_op = _cached(hyp_key, lambda: SpectralOperator(central_wl, fwhm_hyp, wl_gas, expon=expon,
                                                           threshold=1e-4))

        def hyp(signal):
            '''Signal at the resolution of the hyperspectral simulation.'''
            return hyp_op(signal.sel(wl=slice(wl_min - margin, wl_max + margin))).squeeze(drop=True)

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
            coupling = _cached(('coupling', self.gas_coupling),
                               lambda: GasCoupling(COUPLING_TABLES[self.gas_coupling]))
            Tg_atm = gas.get_coupled_transmittance(m, 'atm', aot, coupling)
            Tg_w = gas.get_coupled_transmittance(m, 'w', aot, coupling)
            Tg_Ed = gas.get_coupled_transmittance(m0, 'Ed', aot, coupling)
        Tg_dir, Tg_atm, Tg_w, Tg_Ed = (hyp(T) for T in (Tg_dir, Tg_atm, Tg_w, Tg_Ed))
        self.Tg = xr.Dataset({'Tg_dir': Tg_dir, 'Tg_atm': Tg_atm, 'Tg_w': Tg_w, 'Tg_Ed': Tg_Ed})

        # ------------------------------------------------------------------------------------------------
        # solar irradiance and transmittances of the Rayleigh-aerosol atmosphere
        # ------------------------------------------------------------------------------------------------
        # extraterrestrial irradiance at the hyperspectral resolution: does not depend on the pixel
        F0 = _cached(('F0',) + hyp_key,
                     lambda: hyp(radoc.SolarIrradiance().tsis.interp(wl=wl_gas)))
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
        band_wl = np.asarray(l1_pt.wl.values, dtype=np.float64)
        band_fwhm = np.asarray(l1_pt.fwhm.values, dtype=np.float64)
        bands = _cached(('sat', band_wl.tobytes(), band_fwhm.tobytes(), central_wl.tobytes()),
                        lambda: SpectralOperator(band_wl, band_fwhm, central_wl))

        self.Ratm_sat = bands(self.Ratm)
        self.Rtoa_sunglint_sat = bands(self.Rtoa_sunglint)
        self.trans_w_sat = bands(self.trans_w)
        # direct transmittance in the bands: Rtoa_sunglint_sat = Tdir_sat * Rsg (Rsg does not depend on wl)
        self.Tdir_sat = bands(Tdir)
        self.Rw_toa = l1_pt.Rtoa - (self.Ratm_sat + self.Rtoa_sunglint_sat)
        self.Rrs = self.Rw_toa / self.trans_w_sat
        return self
