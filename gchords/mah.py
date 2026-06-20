import numpy as np
from scipy.integrate import solve_ivp
from colossus.cosmology import cosmology

params = {'Om0': 0.286,
 'flat': True,
 'H0': 70.0,
 'Ob0': 0.049,
 'sigma8': 0.82,
 'ns': 0.95}

cosmo = cosmology.setCosmology('', params)

'''
These functions are approximations for the halo growth rate taken from Fakhouri+2010 (https://ui.adsabs.harvard.edu/abs/2010MNRAS.406.2267F/abstract)
These are used to model the evolution of the GC-halo connection for some classes in `sampler.py`
'''

def dmdz_mean(z, m):
    a = 46.1 * (m/1e12)**(1.1) * (1 + 1.11*z)
    rad = np.sqrt(0.286 * (1+z)**3 + (1 - 0.286))
    dtdz = cosmo.age(z, derivative=1)
    return [a * rad * dtdz * 1e9]

# Mean rates
def mean_mah(m0, zmax=20, z_eval=None):
    if z_eval is None:
        z_eval = np.linspace(0, zmax, 500)

    sol = solve_ivp(
        dmdz_mean,
        (0, zmax),
        y0=[m0],
        t_eval=z_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-10,
        dense_output=True,
    )

    return sol.t, sol.y[0]

# Median rates
def dmdz_median(z, m):
    a = 25.3 * (m/1e12)**(1.1) * (1 + 1.65*z)
    rad = np.sqrt(0.286 * (1+z)**3 + (1 - 0.286))
    dtdz = cosmo.age(z, derivative=1)
    return [a * rad * dtdz * 1e9]

def median_mah(m0, zmax=20, z_eval=None):
    if z_eval is None:
        z_eval = np.linspace(0, zmax, 500)

    sol = solve_ivp(
        dmdz_median,
        (0, zmax),
        y0=[m0],
        t_eval=z_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-10,
        dense_output=True,
    )

    return sol.t, sol.y[0]
