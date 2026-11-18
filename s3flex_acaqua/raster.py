'''
Atmospheric correction of a whole raster: the radiative transfer is solved on a coarse grid of nodes and
interpolated to the full resolution.

The terms of the TOA reflectance in the sensor bands depend on the pixel through the geometry and the
meteorological fields of the L2A product, which vary smoothly in space, and through the aerosol optical
thickness, which varies from pixel to pixel. The computation is in two steps:

1. :py:func:`solve_nodes`: :py:class:`~s3flex_acaqua.Process` at nodes every ``step`` pixels (and on the last row
   and column), for a grid of aerosol optical thicknesses at 550 nm ``aot_grid``, in parallel over processes
   (dask); terms ``Ratm_sat``, ``Tdir_sat`` (direct transmittance), ``trans_w_sat`` and ``PAR``;
2. :py:func:`apply_nodes`: at each pixel, the terms are interpolated (bilinear in space, linear in aerosol optical
   thickness, for the aerosol optical thickness of the L2A product), the sunglint reflectance at the surface
   ``Rsg`` is computed exactly from the geometry and the wind of the pixel (it is too peaked to be interpolated),
   and, as in :py:meth:`~s3flex_acaqua.Process.execute`,

       Rw_toa = Rtoa - Ratm_sat - Tdir_sat * Rsg,   Rrs = Rw_toa / trans_w_sat

   (dask array computed block by block with a numba kernel).

Accuracy of the interpolation, against Process at the centre of the cells (worst case) of 40 water cells of
an OLCI scene (2018-07-02), in Rrs units (error / trans_w):

- space, node step 32 px: max 2.7e-5 sr-1 (95th percentile 7e-6), trans_w 6e-4 relative;
  node step 64 px: max 1.1e-4 sr-1 (95th percentile 2.3e-5), trans_w 2.3e-3 relative;
- sunglint (exact Rsg, interpolated Tdir_sat): max 4e-6 sr-1 (32 px), 1.8e-5 sr-1 (64 px);
- aerosol optical thickness, grid of 0.025 (AOT_GRID): max 4e-6 sr-1, trans_w 6e-5 relative.

Example::

    prod = xr.open_zarr(path, decode_coords='all')
    nodes = solve_nodes(prod, step=32, num_workers=20)
    nodes.to_netcdf('nodes.nc')
    l2 = apply_nodes(prod, nodes)
    l2.to_zarr('l2.zarr')
'''

import logging
import os
import warnings

import numba
import numpy as np
import xarray as xr
from threadpoolctl import threadpool_limits

from .process import VZA_MAX

logger = logging.getLogger(__name__)

#: aerosol optical thicknesses at 550 nm of the nodes (the values of the pixels outside are clipped and flagged)
AOT_GRID = np.round(np.arange(0., 1.0001, 0.025), 4)
#: terms in the sensor bands computed at the nodes
TERMS = ('Ratm_sat', 'Tdir_sat', 'trans_w_sat')
#: variables of the product read by Process.execute
INPUT_VARS = ['Rtoa', 'fwhm', 'sza', 'vza', 'raa', 'T865', 'A865', 'horizontal_wind', 'sea_level_pressure',
              'total_ozone', 'total_columnar_water_vapour']
#: options of Process.execute supported by the raster processing (the sunglint is computed by sunglint_iso)
SUPPORTED_SUNGLINT = dict(stats='cm_iso', shadow=False, slope=False)
WL_REF = 550.

# Process instance of the worker process, kept between the batches (look-up tables and operators loaded once)
_PROCESS = {}


def node_positions(size, step):
    '''Indices of the nodes along a dimension of ``size`` pixels: every ``step`` pixels and the last one.'''
    pos = np.arange(0, size, step)
    if pos[-1] != size - 1:
        pos = np.append(pos, size - 1)
    return pos


def aot550(prod):
    '''Aerosol optical thickness at 550 nm of the L2A product, from T865 and the Angstrom exponent A865.'''
    return (prod.T865 * (WL_REF / 865.) ** -prod.A865).rename('aot550')


def wind_speed(prod):
    '''Wind speed (m s-1) of the L2A product.'''
    return np.hypot(prod.horizontal_wind.isel(wind_vectors=0), prod.horizontal_wind.isel(wind_vectors=1))


def sunglint_iso(sza, vza, azi, wind, m=1.334, sigma2_coef=(3e-3, 512e-5)):
    '''
    Sunglint reflectance at the surface (intensity) of :py:meth:`radcalnet_oc.Sunglint.sunglint` for the
    isotropic Cox-Munk statistics without shadowing (``stats='cm_iso'``, ``shadow=False``, the defaults of
    :py:meth:`~s3flex_acaqua.Process.execute`), vectorized over arrays of any shape (the rotation of the reference
    plane does not change the intensity), clipped at zero as in Process.

    :param sza, vza: solar and viewing zenith angles (deg)
    :param azi: relative azimuth (deg, 180 deg when the Sun and the sensor are in opposition)
    :param wind: wind speed (m s-1)
    '''
    rad = np.pi / 180.
    sza, vza, azi = (np.asarray(a, dtype=np.float64) * rad for a in (sza, vza, azi))
    mu0, muv = np.cos(sza), np.cos(vza)
    cos_scat = np.clip(-mu0 * muv - np.sin(sza) * np.sin(vza) * np.cos(azi), -1., 1.)
    omega = (np.pi - np.arccos(cos_scat)) / 2
    cos_n = np.clip((mu0 + muv) / (2. * np.cos(omega)), -1., 1.)
    # Fresnel reflectance of the facets for the incidence omega
    root = np.sqrt(m ** 2 - np.sin(omega) ** 2)
    rl = (root - m ** 2 * np.cos(omega)) / (root + m ** 2 * np.cos(omega))
    rr = (np.cos(omega) - root) / (np.cos(omega) + root)
    rf = 0.5 * (rl ** 2 + rr ** 2)
    sigma2 = sigma2_coef[0] + sigma2_coef[1] * np.asarray(wind, dtype=np.float64)
    tan2 = (1. - cos_n ** 2) / cos_n ** 2
    proba = np.exp(-tan2 / sigma2) / (np.pi * sigma2)
    return np.maximum(rf * np.pi * proba / (4. * mu0 * muv * cos_n ** 4), 0.)


def _solve_batch(batch, process_kwargs, execute_kwargs, aot_grid):
    '''Solve the pixels of a batch in the worker process; returns a list of (index, dict of arrays or None).'''
    from .process import Process

    key = tuple(sorted(process_kwargs.items()))
    if key not in _PROCESS:
        _PROCESS[key] = Process(**process_kwargs)
    process = _PROCESS[key]
    out = []
    for index, pixel in batch:
        try:
            # one thread per worker process (the parallelism is over the processes): no oversubscription by BLAS
            with warnings.catch_warnings(), threadpool_limits(1):
                warnings.simplefilter('ignore')
                process.execute(pixel, aot_refs=aot_grid, **execute_kwargs)
            res = {term: getattr(process, term).transpose('aot_ref', 'wl').values.astype(np.float32)
                   for term in TERMS}
            res['PAR'] = np.broadcast_to(np.asarray(process.PAR.values, dtype=np.float32), aot_grid.shape)
        except Exception as exc:  # a pixel that fails is left NaN (logged)
            logger.warning('pixel %s failed: %s', index, exc)
            res = None
        out.append((index, res))
    return out


def solve_pixels(prod, iy, ix, aot_grid=AOT_GRID, process_kwargs=None, execute_kwargs=None, num_workers=None,
                 batch_size=8, scheduler='processes'):
    '''
    Terms in the sensor bands of a list of pixels, for the aerosol optical thicknesses ``aot_grid``.

    :param prod: co-registered L1 and L2A product (see :py:meth:`~s3flex_acaqua.Process.execute`), dimensions
        ``y``, ``x`` and ``wl``
    :param iy, ix: indices of the pixels along ``y`` and ``x`` (1-D arrays of the same length)
    :param aot_grid: aerosol optical thicknesses at 550 nm
    :param process_kwargs: arguments of :py:class:`~s3flex_acaqua.Process` (``gas_coupling``, ``o2_model``)
    :param execute_kwargs: other arguments of :py:meth:`~s3flex_acaqua.Process.execute` (not ``aot_refs``; the
        sunglint options must be those of :py:data:`SUPPORTED_SUNGLINT`)
    :param num_workers: number of worker processes (default: number of CPUs, at most 20; ~1 GB each)
    :param batch_size: number of pixels per dask task
    :param scheduler: dask scheduler: 'processes' ('threads' is not supported: Process is not thread-safe),
        'synchronous' to debug, or a dask.distributed Client
    :return: :py:class:`xarray.Dataset` of dimensions (``pixel``, ``aot_ref``, ``wl``)
    '''
    import dask.bag as db

    process_kwargs = dict(process_kwargs or {})
    execute_kwargs = dict(execute_kwargs or {})
    for name, value in SUPPORTED_SUNGLINT.items():
        if execute_kwargs.get(name, value) != value:
            raise NotImplementedError(f'raster processing only for {name}={value!r} (see sunglint_iso)')
    if scheduler == 'threads':
        raise ValueError('Process is not thread-safe: use the processes scheduler')
    aot_grid = np.asarray(aot_grid, dtype=float)
    iy, ix = np.asarray(iy, dtype=int), np.asarray(ix, dtype=int)
    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 20)

    # inputs of the pixels (point indexing: the chunks of the product that contain them are read once)
    variables = [v for v in INPUT_VARS if v in prod.variables]
    pixels = prod[variables].isel(y=xr.DataArray(iy, dims='pixel'), x=xr.DataArray(ix, dims='pixel')).load()
    wl = pixels.wl.values
    items = [(i, pixels.isel(pixel=i)) for i in range(iy.size)]
    npartitions = max(1, int(np.ceil(len(items) / batch_size)))
    bag = db.from_sequence(items, npartitions=npartitions)
    compute_kwargs = dict(scheduler=scheduler)
    if scheduler == 'processes':
        compute_kwargs['num_workers'] = num_workers
    results = bag.map_partitions(_solve_batch, process_kwargs, execute_kwargs, aot_grid).compute(**compute_kwargs)

    shape = (iy.size, aot_grid.size, wl.size)
    data = {term: np.full(shape, np.nan, dtype=np.float32) for term in TERMS}
    data['PAR'] = np.full(shape[:2], np.nan, dtype=np.float32)
    failed = 0
    for index, res in results:
        if res is None:
            failed += 1
            continue
        for name, values in res.items():
            data[name][index] = values
    if failed:
        logger.warning('%d pixel(s) of %d failed', failed, iy.size)
    dims = ('pixel', 'aot_ref', 'wl')
    ds = xr.Dataset({term: (dims, data[term]) for term in TERMS},
                    coords={'aot_ref': aot_grid, 'wl': wl, 'iy': ('pixel', iy), 'ix': ('pixel', ix)})
    ds['PAR'] = (dims[:2], data['PAR'])
    return ds


def needed_nodes(valid, y_node, x_node):
    '''
    Nodes used by the interpolation of the valid pixels: corners of the cells that contain at least one valid
    pixel (the cell of a pixel is the one chosen by :py:func:`apply_nodes`).

    :param valid: boolean array (y, x) of the pixels to process
    :return: boolean array (y_node, x_node)
    '''
    counts = np.add.reduceat(np.add.reduceat(valid.astype(np.int64), y_node[:-1], axis=0), x_node[:-1], axis=1)
    cells = counts > 0
    needed = np.zeros((y_node.size, x_node.size), dtype=bool)
    for dy in (0, 1):
        for dx in (0, 1):
            needed[dy:dy + cells.shape[0], dx:dx + cells.shape[1]] |= cells
    return needed


def solve_nodes(prod, step=32, aot_grid=AOT_GRID, valid=None, **kwargs):
    '''
    Terms in the sensor bands at the nodes of the raster, every ``step`` pixels and on the last row and column;
    see :py:func:`solve_pixels` for the other arguments.

    :param valid: boolean (y, x) array of the pixels to process (default: aerosol optical thickness of the L2A
        product defined); the nodes that no valid pixel uses are not solved (NaN)
    :return: :py:class:`xarray.Dataset` of dimensions (``y_node``, ``x_node``, ``aot_ref``, ``wl``); the
        coordinates ``y_node``, ``x_node`` are the indices of the nodes in the raster
    '''
    y_node = node_positions(prod.sizes['y'], step)
    x_node = node_positions(prod.sizes['x'], step)
    if valid is None:
        valid = np.isfinite(aot550(prod).transpose('y', 'x').values)
    needed = needed_nodes(np.asarray(valid), y_node, x_node)
    iy, ix = (a[needed] for a in np.meshgrid(y_node, x_node, indexing='ij'))
    logger.info('solving %d nodes of %d (%d x %d)', iy.size, needed.size, y_node.size, x_node.size)
    ds = solve_pixels(prod, iy, ix, aot_grid=aot_grid, **kwargs)
    out = xr.Dataset(coords={'y_node': y_node, 'x_node': x_node, 'aot_ref': ds.aot_ref, 'wl': ds.wl})
    for name, var in ds.data_vars.items():
        values = np.full((y_node.size, x_node.size) + var.shape[1:], np.nan, dtype=var.dtype)
        values[needed] = var.values
        out[name] = (('y_node', 'x_node') + var.dims[1:], values)
    out['needed'] = (('y_node', 'x_node'), needed)
    out.attrs.update(step=step, ny=prod.sizes['y'], nx=prod.sizes['x'],
                     process_kwargs=repr(kwargs.get('process_kwargs')),
                     execute_kwargs=repr(kwargs.get('execute_kwargs')))
    return out


def _bracket(nodes, positions):
    '''Lower node index and weight of the upper node for each position (linear interpolation, clipped).'''
    i0 = np.clip(np.searchsorted(nodes, positions, side='right') - 1, 0, nodes.size - 2)
    w = (positions - nodes[i0]) / (nodes[i0 + 1] - nodes[i0])
    return i0.astype(np.int64), np.clip(w, 0., 1.)


@numba.njit(nogil=True, cache=True)
def _interp_kernel(rtoa, aot, rsg, iy0, wy, ix0, wx, aot_grid, ratm, tdir, trans, par, rrs, rw, par_out):
    '''
    Trilinear interpolation of the terms (y_node, x_node, aot, wl) at each pixel of a block and water-leaving
    terms; rtoa (wl, y, x), aot and rsg (y, x); outputs rrs, rw (wl, y, x), par_out (y, x).
    '''
    nwl, ny, nx = rtoa.shape
    naot = aot_grid.size
    ys = np.empty(8, dtype=np.int64)
    xs = np.empty(8, dtype=np.int64)
    aas = np.empty(8, dtype=np.int64)
    ws = np.empty(8)
    for j in range(ny):
        y0, fy = iy0[j], wy[j]
        for i in range(nx):
            a = aot[j, i]
            if not np.isfinite(a):
                for k in range(nwl):
                    rrs[k, j, i] = np.nan
                    rw[k, j, i] = np.nan
                par_out[j, i] = np.nan
                continue
            # aerosol optical thickness bracket (clipped to the grid)
            if a <= aot_grid[0]:
                a0, fa = 0, 0.
            elif a >= aot_grid[naot - 1]:
                a0, fa = naot - 2, 1.
            else:
                a0 = np.searchsorted(aot_grid, a, side='right') - 1
                fa = (a - aot_grid[a0]) / (aot_grid[a0 + 1] - aot_grid[a0])
            x0, fx = ix0[i], wx[i]
            # the 8 corners and their weights
            c = 0
            for dy in range(2):
                for dx in range(2):
                    for da in range(2):
                        ys[c], xs[c], aas[c] = y0 + dy, x0 + dx, a0 + da
                        ws[c] = ((fy if dy else 1. - fy) * (fx if dx else 1. - fx) * (fa if da else 1. - fa))
                        c += 1
            p = 0.
            for c in range(8):
                p += ws[c] * par[ys[c], xs[c], aas[c]]
            par_out[j, i] = p
            glint = rsg[j, i]
            for k in range(nwl):
                path = 0.
                tw = 0.
                for c in range(8):
                    path += ws[c] * (ratm[ys[c], xs[c], aas[c], k] + glint * tdir[ys[c], xs[c], aas[c], k])
                    tw += ws[c] * trans[ys[c], xs[c], aas[c], k]
                r = rtoa[k, j, i] - path
                rw[k, j, i] = r
                rrs[k, j, i] = r / tw


def apply_nodes(prod, nodes, chunks=512):
    '''
    Remote-sensing reflectance of all the pixels from the terms solved at the nodes (:py:func:`solve_nodes`).

    :param prod: co-registered L1 and L2A product, dimensions ``wl``, ``y``, ``x`` (``Rtoa``, ``T865``,
        ``A865``, ``sza``, ``vza``, ``raa``, ``horizontal_wind``)
    :param nodes: output of :py:func:`solve_nodes` for this product
    :param chunks: size of the blocks along ``y`` and ``x``
    :return: lazy :py:class:`xarray.Dataset` (dask): ``Rrs`` (sr-1), ``Rw_toa`` (wl, y, x), ``PAR``, ``Rsg``
        (sunglint reflectance at the surface), ``aot550`` and ``aot_clipped`` (aerosol optical thickness outside
        the grid of the nodes, clipped) (y, x); NaN where T865 or A865 is not defined
    '''
    import dask.array as da

    if (nodes.attrs.get('ny'), nodes.attrs.get('nx')) != (prod.sizes['y'], prod.sizes['x']):
        raise ValueError('nodes not computed for this raster')
    if not np.array_equal(nodes.wl.values, prod.wl.values):
        raise ValueError('nodes not computed for the bands of this product')
    y_node = nodes.y_node.values.astype(np.float64)
    x_node = nodes.x_node.values.astype(np.float64)
    aot_grid = nodes.aot_ref.values.astype(np.float64)
    # float32 to halve the memory (the sums are in float64 in the kernel)
    coefs = [np.ascontiguousarray(nodes[name].transpose('y_node', 'x_node', 'aot_ref', 'wl').values,
                                  dtype=np.float32) for name in TERMS]
    par = np.ascontiguousarray(nodes.PAR.transpose('y_node', 'x_node', 'aot_ref').values, dtype=np.float32)

    def dask_data(da_x, dims):
        data = da_x.transpose(*dims).data
        return data if isinstance(data, da.Array) else da.from_array(data)

    block_chunks = (chunks, chunks)
    rtoa = dask_data(prod.Rtoa, ('wl', 'y', 'x')).rechunk((-1,) + block_chunks)
    aot = dask_data(aot550(prod), ('y', 'x')).rechunk(block_chunks)
    # sunglint reflectance at the surface of each pixel, as in Process (viewing zenith angle of the LUT)
    rsg = xr.apply_ufunc(sunglint_iso, prod.sza, np.minimum(prod.vza, VZA_MAX), prod.raa, wind_speed(prod),
                         dask='parallelized', output_dtypes=[np.float64])
    rsg = dask_data(rsg, ('y', 'x')).rechunk(block_chunks)

    def block(rtoa_b, aot_b, rsg_b, block_info=None):
        (y_start, _), (x_start, _) = block_info[1]['array-location']
        ny, nx = aot_b.shape
        iy0, wy = _bracket(y_node, np.arange(y_start, y_start + ny, dtype=np.float64))
        ix0, wx = _bracket(x_node, np.arange(x_start, x_start + nx, dtype=np.float64))
        nwl = rtoa_b.shape[0]
        # one output array: Rrs, Rw_toa and the PAR as an extra band (split below)
        out = np.empty((2 * nwl + 1, ny, nx), dtype=np.float32)
        _interp_kernel(np.ascontiguousarray(rtoa_b, dtype=np.float32), np.ascontiguousarray(aot_b, dtype=np.float32),
                       np.ascontiguousarray(rsg_b, dtype=np.float64), iy0, wy, ix0, wx, aot_grid, *coefs, par,
                       out[:nwl], out[nwl:2 * nwl], out[2 * nwl])
        return out

    nwl = rtoa.shape[0]
    stacked = da.map_blocks(block, rtoa, aot, rsg, dtype=np.float32,
                            chunks=((2 * nwl + 1,), aot.chunks[0], aot.chunks[1]))
    dims = ('wl', 'y', 'x')
    out = xr.Dataset({
        'Rrs': (dims, stacked[:nwl]),
        'Rw_toa': (dims, stacked[nwl:2 * nwl]),
        'PAR': (dims[1:], stacked[2 * nwl]),
        'Rsg': (dims[1:], rsg.astype(np.float32)),
        'aot550': (dims[1:], aot.astype(np.float32)),
        'aot_clipped': (dims[1:], (aot > aot_grid[-1]) | (aot < aot_grid[0])),
    }, coords={'wl': prod.wl, 'y': prod.y, 'x': prod.x})
    # the encoding of the coordinates of the input (e.g. its zarr compressor) may not be valid for the output
    out = out.drop_encoding()
    out.Rrs.attrs.update(units='sr-1', long_name='remote-sensing reflectance')
    out.Rw_toa.attrs.update(long_name='water-leaving term at the TOA, Rtoa - Ratm - Rtoa_sunglint')
    out.PAR.attrs.update(units='umol photons m-2 s-1')
    out.Rsg.attrs.update(long_name='sunglint reflectance at the surface (isotropic Cox-Munk)')
    out.attrs.update(node_step=int(nodes.attrs.get('step', -1)), aot_grid=list(map(float, aot_grid)))
    return out
