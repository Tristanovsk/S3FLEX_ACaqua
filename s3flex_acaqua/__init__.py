'''

Version history
==================

0.1.0:
    - radcalnet_oc >= 1.1: gaseous transmittances of the path reflectance, of the water-leaving term and of the
      downwelling irradiance coupled with the scattering (photon path-length model, option ``gas_coupling``),
      line-by-line O2 (option ``o2_model``), look-up tables interpolated at the wind speed of the L2A product
    - remote-sensing reflectance divided by the diffuse transmittances (``Rrs``), water term at the TOA
      ``Rw_toa``, downwelling irradiance ``Ed`` (with the Earth-Sun distance) and ``PAR``
    - packaging: requirements.txt, README (2026/11/16)

0.0.1:
    - initial version (2026/09/01)

'''

__package__ = 's3flex_acaqua'
__version__ = '0.1.0'


from .process import Process