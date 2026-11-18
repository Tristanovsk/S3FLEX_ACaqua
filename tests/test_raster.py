'''Interpolation of the terms of the nodes to the raster (synthetic data, no radiative transfer).'''

import numpy as np
import pytest
import xarray as xr

import radcalnet_oc as radoc
from s3flex_acaqua import raster

NY, NX, STEP = 37, 50, 16
WL = np.array([500., 665., 865.], dtype=np.float32)
AOT_GRID = np.array([0., 0.1, 0.3, 0.8])
COEFS = {'Ratm_sat': (0.01, 1e-4, -2e-5, 0.05, 0.01), 'Tdir_sat': (0.8, -1e-4, 1e-4, -0.3, 0.1),
         'trans_w_sat': (2.5, -1e-3, 1e-3, -0.4, 0.2)}


def linear_term(y, x, aot, wl, c):
    '''Term linear in y, x and aot (exactly interpolated), any function of the wavelength.'''
    return c[0] + c[1] * y + c[2] * x + c[3] * aot + c[4] * wl / 1000


@pytest.fixture
def case():
    rng = np.random.default_rng(0)
    T865 = rng.uniform(0., 0.3, (NY, NX)).astype(np.float32)
    A865 = np.zeros((NY, NX), dtype=np.float32)  # aot550 = T865
    T865[3, 4] = np.nan
    T865[5, 6] = 1.5  # above the grid: clipped
    yy, xx = np.meshgrid(np.arange(NY), np.arange(NX), indexing='ij')
    sza = 30 + 0.1 * yy
    vza = 5 + 1.2 * xx  # up to 64 deg: clipped at VZA_MAX for the sunglint
    raa = np.full((NY, NX), 170.)
    wind = np.stack([np.full((NY, NX), 3.), np.full((NY, NX), 4.)], axis=-1).astype(np.float32)
    prod = xr.Dataset({'Rtoa': (('wl', 'y', 'x'), rng.uniform(0.02, 0.1, (WL.size, NY, NX))),
                       'T865': (('y', 'x'), T865), 'A865': (('y', 'x'), A865),
                       'sza': (('y', 'x'), sza), 'vza': (('y', 'x'), vza), 'raa': (('y', 'x'), raa),
                       'horizontal_wind': (('y', 'x', 'wind_vectors'), wind)},
                      coords={'wl': WL, 'y': np.arange(NY)[::-1], 'x': np.arange(NX)}).chunk({'y': 10, 'x': 13})
    y_node, x_node = raster.node_positions(NY, STEP), raster.node_positions(NX, STEP)
    yn, xn, an, wn = np.meshgrid(y_node, x_node, AOT_GRID, WL, indexing='ij')
    nodes = xr.Dataset({k: (('y_node', 'x_node', 'aot_ref', 'wl'), linear_term(yn, xn, an, wn, c))
                        for k, c in COEFS.items()},
                       coords={'y_node': y_node, 'x_node': x_node, 'aot_ref': AOT_GRID, 'wl': WL})
    nodes['PAR'] = (('y_node', 'x_node', 'aot_ref'), 1500 - 300 * yn[..., 0] / NY - 200 * an[..., 0])
    nodes.attrs.update(step=STEP, ny=NY, nx=NX)
    return prod, nodes


def test_exact_for_linear_terms(case):
    prod, nodes = case
    out = raster.apply_nodes(prod, nodes, chunks=11).compute()
    aot = np.clip(prod.T865.values.astype(np.float64), AOT_GRID[0], AOT_GRID[-1])
    y, x = np.meshgrid(np.arange(NY), np.arange(NX), indexing='ij')
    wl = WL[:, None, None].astype(np.float64)
    rsg = raster.sunglint_iso(prod.sza.values, np.minimum(prod.vza.values, raster.VZA_MAX), prod.raa.values, 5.)
    assert rsg.max() > 1e-3
    path = linear_term(y, x, aot, wl, COEFS['Ratm_sat']) + rsg * linear_term(y, x, aot, wl, COEFS['Tdir_sat'])
    trans = linear_term(y, x, aot, wl, COEFS['trans_w_sat'])
    rw = prod.Rtoa.values - path
    np.testing.assert_allclose(out.Rsg.values, rsg, rtol=1e-6)
    np.testing.assert_allclose(out.Rw_toa.values, rw, rtol=1e-5, atol=1e-7, equal_nan=True)
    np.testing.assert_allclose(out.Rrs.values, rw / trans, rtol=1e-5, atol=1e-7, equal_nan=True)
    np.testing.assert_allclose(out.PAR.values, 1500 - 300 * y / NY - 200 * aot, rtol=1e-6, equal_nan=True)
    assert np.isnan(out.Rrs.values[:, 3, 4]).all() and np.isnan(out.PAR.values[3, 4])
    assert bool(out.aot_clipped.values[5, 6]) and int(out.aot_clipped.sum()) == 1


def test_skipped_nodes_not_used(case):
    prod, nodes = case
    # a few valid pixels: inside a cell, on a node row, on a node and in the last row and column
    T865 = np.full(prod.T865.shape, np.nan, dtype=np.float32)
    for j, i in [(5, 7), (16, 20), (32, 48), (36, 49), (20, 49)]:
        T865[j, i] = 0.1
    prod = prod.assign(T865=(('y', 'x'), T865))
    full = raster.apply_nodes(prod, nodes, chunks=11).Rrs.compute()
    needed = raster.needed_nodes(np.isfinite(T865), nodes.y_node.values, nodes.x_node.values)
    assert 0 < needed.sum() < needed.size
    sparse = nodes.copy()
    for name in raster.TERMS + ('PAR',):
        sparse[name] = sparse[name].where(xr.DataArray(needed, dims=('y_node', 'x_node')))
    out = raster.apply_nodes(prod, sparse, chunks=11).Rrs.compute()
    np.testing.assert_array_equal(np.isfinite(out.values), np.isfinite(full.values))
    np.testing.assert_allclose(out.values, full.values, equal_nan=True)


def test_sunglint_iso_same_as_radcalnet_oc():
    rng = np.random.default_rng(0)
    n = 300
    sza, vza, azi, ws = rng.uniform(5, 75, n), rng.uniform(0, 60, n), rng.uniform(0, 360, n), rng.uniform(0, 15, n)
    # near-specular and nadir geometries
    sza[:100] = vza[:100] = rng.uniform(10, 50, 100)
    azi[:100] = 180 + rng.normal(0, 3, 100)
    vza[100:120] = 0
    ref = [max(float(radoc.Sunglint(a, b, c, m=1.334).sunglint(w, 0, stats='cm_iso', shadow=False)[0]), 0.)
           for a, b, c, w in zip(sza, vza, azi, ws)]
    np.testing.assert_allclose(raster.sunglint_iso(sza, vza, azi, ws), ref, rtol=1e-10, atol=1e-14)


def test_unsupported_options(case):
    prod, _ = case
    with pytest.raises(NotImplementedError):
        raster.solve_pixels(prod, [0], [0], execute_kwargs={'stats': 'cm_dir'})
    with pytest.raises(ValueError):
        raster.solve_pixels(prod, [0], [0], scheduler='threads')


def test_wrong_raster(case):
    prod, nodes = case
    with pytest.raises(ValueError):
        raster.apply_nodes(prod.isel(y=slice(1, None)), nodes)


def test_node_positions():
    np.testing.assert_array_equal(raster.node_positions(10, 4), [0, 4, 8, 9])
    np.testing.assert_array_equal(raster.node_positions(9, 4), [0, 4, 8])
