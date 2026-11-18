# S3FLEX_ACaqua package
## Prototype code to perform atmospheric correction of FLEX above water

The atmospheric correction of a FLEX (or Sentinel-3 FLEX-mode) pixel uses the atmospheric parameters of the
Sentinel-3 L2A products (aerosol optical thickness and Angström exponent, water vapour, ozone, pressure, wind speed)
and the radiative transfer simulator [radcalnet_oc](https://radcalnet-oc.readthedocs.io) (version >= 1.1):

    R_toa = T_g^atm R_atm + T_dir R_sg + pi T_down t_up T_g^w R_rs

with the gaseous absorption coupled with the scattering (`gas_coupling='path_length'`), line-by-line O2
(`o2_model='lbl'`) and the sunglint of the direct sunlight (Cox and Munk). The theoretical basis is described in the
ATBD *Radiative transfer model for FLEX simulations and atmospheric correction above water* (PHY2FLEX-ATBD-MAG-23,
issue V1.1).

### Principles
![p](/static/principle_eq.png)

![s](/static/toa_simu.png)

### Installation
First install [radcalnet_oc](https://github.com/Tristanovsk/radcalnet_oc) (>= 1.1) and its look-up table, downloaded
from Zenodo (https://doi.org/10.5281/zenodo.23394202):
```commandline
pip install radcalnet_oc
radcalnet_oc_config --init /path/to/lut
radcalnet_oc_config --download
```

And then, in the repertory of your clone type:
```commandline
pip install .
```

### Usage
```python
import s3flex_acaqua

# l1_pt: pixel of the co-registered FLEX (L1) and Sentinel-3 L2A products
l2a_pt = s3flex_acaqua.Process().execute(l1_pt, aot_refs=[0.05, 0.1], aerosol_combination=[0, 0, 1, 0, 0, 0])
l2a_pt.Rrs, l2a_pt.Ed, l2a_pt.PAR
```

#### Whole raster
The radiative transfer is solved at nodes every `step` pixels for a grid of aerosol optical thicknesses (in parallel
over processes with dask), then interpolated to all the pixels (bilinear in space, linear in aerosol optical
thickness) with the exact sunglint of each pixel (dask array, numba kernel):
```python
import xarray as xr
from s3flex_acaqua import raster

prod = xr.open_zarr(path, decode_coords='all')  # co-registered L1 and L2A products (y, x, wl)
nodes = raster.solve_nodes(prod, step=32, num_workers=20,
                           execute_kwargs={'aerosol_combination': [0, 0, 1, 0, 0, 0]})
nodes.to_netcdf('nodes.nc')
raster.apply_nodes(prod, nodes).to_zarr('l2.zarr')  # Rrs, Rw_toa, PAR, Rsg, aot550, aot_clipped
```
On an OLCI scene of 4065 x 4865 pixels (7 million water pixels), with 20 workers on a shared 40-core machine: nodes
(step 64: 3527 nodes over water) in 20 min, interpolation of the whole scene in 1.6 min. Difference with
`Process.execute` at 40 random water pixels: max 3.2e-5 sr-1 (median 5e-7 sr-1) with step 64; the step 32 (default)
divides the interpolation error by ~4 for ~3.5 times more nodes. Only the isotropic Cox-Munk sunglint without
shadowing (the defaults) is supported.

The notebook `notebook/atbd_v1.1_figures.ipynb` makes the figures of the ATBD V1.1 from a Sentinel-3 FLEX-mode image.
