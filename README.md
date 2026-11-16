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

The notebook `notebook/atbd_v1.1_figures.ipynb` makes the figures of the ATBD V1.1 from a Sentinel-3 FLEX-mode image.
