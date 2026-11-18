'''The sparse convolution operator gives the band values of radcalnet_oc.Spectral (no data file needed).'''

import numpy as np
import pytest
import xarray as xr

import radcalnet_oc as radoc
from s3flex_acaqua.process import SpectralOperator

WL_SIGNAL = np.arange(395., 905., 0.05)


@pytest.fixture
def signal():
    rng = np.random.default_rng(0)
    values = 1 + 0.5 * np.sin(WL_SIGNAL / 7.)[None, :] * rng.uniform(0.5, 1.5, (3, 1))
    return xr.DataArray(values, coords={'aot_ref': [0.05, 0.1, 0.2], 'wl': WL_SIGNAL}, dims=('aot_ref', 'wl'))


@pytest.mark.parametrize('expon, threshold, fwhm', [(2, 1e-4, 0.3), (3, 1e-4, 0.5), (None, None, 3.7)])
def test_same_as_radcalnet_oc(signal, expon, threshold, fwhm):
    # bands across the whole range, including one truncated at the edge and one outside the signal
    central_wl = np.concatenate([[395.1], np.arange(400., 900., 1.3), [1100.]])
    op = SpectralOperator(central_wl, fwhm, WL_SIGNAL, expon=expon, threshold=threshold, block=97)
    ref = radoc.Spectral(central_wl, fwhm)
    weights = ref.weights(WL_SIGNAL, expon=expon, threshold=threshold)
    expected = ref.convolve_nd(signal, weights).transpose('aot_ref', 'wl').values
    out = op(signal).transpose('aot_ref', 'wl')
    np.testing.assert_array_equal(out.wl.values, central_wl)
    assert np.isnan(out.values[:, -1]).all()
    # radcalnet_oc returns float32
    np.testing.assert_allclose(out.values, expected, rtol=2e-7, equal_nan=True)


def test_nan_only_reaches_overlapping_bands(signal):
    central_wl = np.array([500., 600., 700.])
    op = SpectralOperator(central_wl, 0.3, WL_SIGNAL, expon=2, threshold=1e-4)
    signal = signal.copy()
    signal[:, np.argmin(np.abs(WL_SIGNAL - 600.))] = np.nan
    out = op(signal).values
    assert np.isnan(out[:, 1]).all()
    assert np.isfinite(out[:, [0, 2]]).all()


def test_wrong_grid(signal):
    op = SpectralOperator([500.], 0.3, WL_SIGNAL[::2])
    with pytest.raises(ValueError):
        op(signal)
