import abc
import os
import numpy as np
from scipy.stats import uniform
from scipy.interpolate import interp1d, LinearNDInterpolator

# sometimes I have to use older versions of numpy that don't have _trapezoid, so I import it from scipy...
try:
    from scipy.integrate import trapezoid as _trapezoid
except ImportError:
    from scipy.integrate import trapz as _trapezoid

from scipy.stats import lognorm
import symlib
from scipy.stats import norm, truncnorm
from gchords.tag import GlobularClusterRhalf
from gchords.mah import mean_mah, cosmo


class AgeModel(abc.ABC):
    """
    Base class for GC formation-time distributions.

    Subclasses supply a p(t) distribution in cosmic time (Gyr since Big Bang).
    """

    @abc.abstractmethod
    def p_time(self, t):
        """Probability density p(t_form) evaluated at cosmic time t (Gyr)."""
        pass

    def p_age(self, age):
        """Probability density p(age) evaluated at lookback time age (Gyr)."""
        raise NotImplementedError

    def cdf(self, t):
        """
        Fraction of GC formation occurring before cosmic time t:
            S(t) = integral_0^t p(t') dt'

        Default implementation integrates p_time numerically.
        Subclasses may override for analytic efficiency.
        """
        t_grid = np.linspace(0, t, 2000)
        return _trapezoid(self.p_time(t_grid), t_grid)


class KruijssenAgeModel(AgeModel):
    """
    p(t_form) based on the mean age distribution in Kruijssen (2019),
    converted from the z-space KDE table to cosmic time via the mah.py cosmology.
    """

    def __init__(self):
        _data_path = os.path.join(os.path.dirname(__file__), "data", "zform_pdf_table.csv")
        table = np.genfromtxt(_data_path, delimiter=",", skip_header=1)
        z_grid = table[:, 0]
        pdf_z  = table[:, 1]

        # convert z → cosmic time; z increases → t decreases, so reverse
        t_grid = np.array([cosmo.age(z) for z in z_grid])
        dtdz   = np.abs(np.gradient(t_grid, z_grid))   # |dt/dz|

        # change of variables: p(t) = p(z) / |dt/dz|
        pdf_t  = pdf_z / dtdz
        t_grid = t_grid[::-1]
        pdf_t  = pdf_t[::-1]

        norm_factor    = _trapezoid(pdf_t, t_grid)
        self._pdf_norm = pdf_t / norm_factor
        self._t_grid   = t_grid

        self._pdf_interp = interp1d(
            t_grid, self._pdf_norm,
            bounds_error=False, fill_value=0.0,
        )

        # precomputed CDF: S(t) = fraction of GCs formed before time t
        cdf_vals = np.zeros_like(self._pdf_norm)
        for i in range(1, len(t_grid)):
            cdf_vals[i] = _trapezoid(self._pdf_norm[:i+1], t_grid[:i+1])
        self._sf_interp = interp1d(
            t_grid, cdf_vals,
            bounds_error=False, fill_value=(0.0, 1.0),
        )

    def p_time(self, t):
        return self._pdf_interp(t)

    def cdf(self, t):
        return self._sf_interp(t)


class ValcinAgeModel(AgeModel):
    """
    p(t_form) derived from ages in Valcin+2025, parameterized as a Gaussian
    in cosmic time truncated to [0, t_max].
    (source: https://arxiv.org/abs/2503.19481)
    """

    def __init__(self, mu=11.89, sigma=0.98, t_max=13.8):
        self._mu = mu
        self._sigma = sigma
        self._tmax = t_max

        loc = t_max - mu
        a = (0 - loc) / sigma
        b = (t_max - loc) / sigma
        self._dist = truncnorm(a, b, loc=loc, scale=sigma)

    def p_time(self, t):
        return self._dist.pdf(t)

    def p_age(self, age):
        return self._dist.pdf(self._tmax - age)

    def cdf(self, t):
        return self._dist.cdf(np.clip(t, 0, self._tmax))


"""
Mass-to-light ratios are taken from N-body modeling of globular clusters:
https://arxiv.org/abs/1609.08794
"""


def magnitude_to_luminosity(magnitude, zero_point=5.12):
    """Convert magnitude to luminosity using the zero point."""
    return 10 ** ((zero_point - magnitude) / 2.5)


def luminosity_to_mass(luminosity, ratio=3.0):
    """Convert luminosity to mass using the mass-to-light ratio."""
    return luminosity * ratio


class OccupationModel(abc.ABC):
    """
    Base class for GC occupation models
    """

    def __init__(self, seed=None):
        self.kind = None
        if seed is not None:
            np.random.seed(seed)

    @abc.abstractmethod
    def var_names(self):
        pass

    @abc.abstractmethod
    def p_gc(self, mass, z=None):
        pass

    @abc.abstractmethod
    def has_gc(self, mass, z=None):
        pass


class GCSMassModel(abc.ABC):
    """
    Base class for GC system mass models
    """

    def __init__(self, seed=None):
        self.kind = None
        self.evolving = False
        self.mean_mass = 0

        if seed is not None:
            np.random.seed(seed)

    @abc.abstractmethod
    def var_names(self):
        pass

    @abc.abstractmethod
    def mass(self, **kwargs):
        pass


class GCLuminosityFunction(abc.ABC):
    """
    Base class for GC luminosity functions
    """

    def __init__(self, seed=None):
        self.kind = None
        self.evolving = False

        if seed is not None:
            np.random.seed(seed)

    @abc.abstractmethod
    def var_names(self):
        pass

    @abc.abstractmethod
    def sample(self, n_draws, return_L=False, **kwargs):
        pass

    @abc.abstractmethod
    def set_halo_mass(self, halo_mass, seed=None):
        pass

class MassLightRatioModel(abc.ABC):
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
    @abc.abstractmethod
    def var_names(self):
        pass

    @abc.abstractmethod
    def sample(self, n_draws):
        pass


def sample_truncated_normal(loc, scale, xmin=-np.inf, xmax=np.inf):
    a = (xmin - loc) / scale
    b = (xmax - loc) / scale
    return truncnorm.rvs(a, b, loc=loc, scale=scale)


def _lognormal_icdf(logmu, sigma):
    return lognorm(s=sigma * np.log(10), scale=10**logmu).ppf

class GaussianGCLF(GCLuminosityFunction):
    def __init__(self, mu=-7.0, sigma=1.0, M_sun=5.12, log_ml=0.3, log_ml_sigma=0.0, seed=None):
        """
        Generic Gaussian GCLF
        """

        super().__init__(seed=seed)

        # The luminosity function is defined in magnitudes
        self.mu = mu # mean magnitude
        self.sigma = sigma # sd of magnitude dist
        self.M_sun = M_sun # absolute magnitude of the sun in the same band as mu
        self.log_ml = log_ml
        self.log_ml_sigma = log_ml_sigma

        log_M_star, sigma_logM, mean_mass = self.compute_mass_parameters(mu, sigma, log_ml, log_ml_sigma)
        self.mean_mass = mean_mass
        self.sigma_M = sigma_logM
        
        # precompute the inverse CDF for sampling
        self.icdf = _lognormal_icdf(log_M_star, self.sigma_M)

        self.kind = None  # Kind of dependence of the GCLF parameters
        self.evolving = False  # Whether the GCLF parameters evolve over redshift

    def var_names(self):
        return ['mu', 'sigma', 'M_sun', 'log_ml', 'log_ml_sigma']
    
    def compute_mass_parameters(self, mu, sigma, log_ml, log_ml_sigma):
        log_L_star = - mu / 2.5 + self.M_sun / 2.5 # turnover luminosity
        sigma_logL = sigma / 2.5

        log_M_star = log_L_star + log_ml        
        sigma_logM = np.sqrt(sigma_logL**2 + log_ml_sigma**2)
        mean_mass = 10**(log_M_star) * np.exp(0.5 * (np.log(10) * sigma_logM) ** 2)
        return log_M_star, sigma_logM, mean_mass

    def sample_mag(self, n_draws):
        """
        Samples magnitudes from the GCLF
        """
        return np.random.normal(self.mu, self.sigma, size=n_draws)

    def sample(self, n_draws, return_L=False, **kwargs):
        """
        Samples masses from the GCLF
        """
        u = np.random.uniform(0, 1, size=n_draws)
        masses = self.icdf(u)

        if return_L:
            if self.log_ml_sigma > 0:
                log_ml = np.random.normal(self.log_ml, self.log_ml_sigma, size=n_draws)
            else:
                log_ml = self.log_ml
            luminosities = masses / (10 ** log_ml)
            return masses, luminosities

        return masses

class FlexibleMassLightRatioGCLF(GCLuminosityFunction):
    '''
    This luminosity function class samples GC masses and luminosities by:

    1. sampling a distribution of mean and variances of the GCMF for each specified halo mass
    2. sampling GC masses from the distribution defined by the two random variables (defined in the last step)
    3. samples an empirical M/L ratio distribution to assign luminosities

    '''
    def __init__(self, ml_model=None, seed=None):
        super().__init__(seed=seed)
        self.ml_model = ml_model if ml_model is not None else GChordsMassLightRatioModel()

        # parameters are fit simultaneously with the GChordsMassLightRatioModel model
        # describe Gaussian distributions in log10(M) and log10(sigma)
        self.gcmf_mean_mu = 5.22
        self.gcmf_mean_scale = 0.02
        self.gcmf_std_mu = 0.33
        self.gcmf_std_scale = 0.19

        self.set_sampler()

    def var_names(self):
        return ['ml_model']

    def set_halo_mass(self, halo_mass, seed=None):
        pass

    def set_sampler(self):
        # sample the GCMF parameter random variables
        self.mass_mu = np.random.normal(loc=self.gcmf_mean_mu, scale=self.gcmf_mean_scale)
        self.mass_sigma = sample_truncated_normal(self.gcmf_std_mu, self.gcmf_std_scale, xmin=0.0)

        # sample GC masses from the lognormal GCMF
        self.icdf = _lognormal_icdf(self.mass_mu, self.mass_sigma)

        self.mean_mass = 10**self.mass_mu * np.exp(0.5 * (np.log(10) * self.mass_sigma) ** 2)


    def sample(self, n_draws, return_L=False, **kwargs):
        """
        Samples masses from the GCLF
        """

        u = np.random.uniform(0, 1, size=n_draws)
        masses = self.icdf(u)

        if return_L:
            # sample log M/L ratios and convert masses to luminosities
            log_ml = self.ml_model.sample(n_draws)
            luminosities = masses / (10 ** log_ml)
            return masses, luminosities

        return masses


"""
Occupation models
"""


class EadieOccupationModel(OccupationModel):
    def __init__(self, b0=-10.83, b1=1.59, seed=None):
        super().__init__(seed=seed)
        self.kind = "stellar"
        self.b0 = b0
        self.b1 = b1

    def var_names(self):
        return ["b0", "b1"]

    def p_gc(self, stellar_mass, z=None):
        p = (1 + np.exp(-1 * (self.b0 + self.b1 * np.log10(stellar_mass)))) ** (-1)
        return p

    def has_gc(self, stellar_mass, z=None):
        p = self.p_gc(stellar_mass, z=z)
        return uniform.rvs() < p


class DornanOccupationModel(OccupationModel):
    def __init__(self, b0=-31.86, b1=3.0, seed=None):
        super().__init__(seed=seed)
        self.kind = "halo"
        self.b0 = b0
        self.b1 = b1

    def var_names(self):
        return ["b0", "b1"]

    def p_gc(self, halo_mass, z=None):
        p = 1 / (1 + np.exp(-(self.b0 + self.b1 * np.log10(halo_mass))))
        return p

    def has_gc(self, halo_mass, z=None):
        p = self.p_gc(halo_mass, z=z)
        return uniform.rvs() < p


class DornanOccupationModelInSitu(OccupationModel):
    """
    In-situ variant of DornanOccupationModel.

    Builds an interpolator over (log_mhalo, lookback_time) → p_gc by tracing
    each halo's mean MAH and anchoring p_gc to the z=0 peak halo mass via the
    Dornan logistic relation.
    """

    def __init__(self, b0=-31.86, b1=3.0, masses=None, z_eval=None, seed=None):
        super().__init__(seed=seed)
        self.kind = "halo"
        self.b0 = b0
        self.b1 = b1

        if masses is None:
            masses = np.logspace(7, 14, 50)
        if z_eval is None:
            z_eval = np.linspace(0, 20, 100)

        self._interp = self._build_interpolator(masses, z_eval)

    def _p_gc_z0(self, halo_mass):
        return 1 / (1 + np.exp(-(self.b0 + self.b1 * np.log10(halo_mass))))

    def _build_interpolator(self, masses, z_eval):
        log_mhalo_pts, t_pts, p_gc_vals = [], [], []

        for mass in masses:
            zs, m_track = mean_mah(mass, z_eval=z_eval)
            p_z0 = self._p_gc_z0(m_track[0])
            t = np.array([cosmo.age(z) for z in zs])
            log_mhalo_pts.extend(np.log10(m_track))
            t_pts.extend(t)
            p_gc_vals.extend(np.full(len(zs), p_z0))

        return LinearNDInterpolator(
            np.column_stack([log_mhalo_pts, t_pts]),
            np.array(p_gc_vals),
        )

    def var_names(self):
        return ["b0", "b1"]

    def p_gc(self, halo_mass, z=0.0):
        halo_mass = np.atleast_1d(np.asarray(halo_mass, dtype=float))
        t = np.full_like(halo_mass, cosmo.age(z if z is not None else 0.0))
        return self._interp(np.log10(halo_mass), t)

    def has_gc(self, halo_mass, z=0.0):
        p = self.p_gc(halo_mass, z=z)
        return uniform.rvs() < float(p)


class DornanMstarOccupationModel(OccupationModel):
    def __init__(self, b0=-10.64, b1=1.38, seed=None):
        super().__init__(seed=seed)
        self.kind = "stellar"
        self.b0 = b0
        self.b1 = b1

    def var_names(self):
        return ["b0", "b1"]

    def p_gc(self, stellar_mass, z=None):
        p = 1 / (1 + np.exp(-(self.b0 + self.b1 * np.log10(stellar_mass))))
        return p

    def has_gc(self, stellar_mass, z=None):
        p = self.p_gc(stellar_mass, z=z)
        return uniform.rvs() < p


"""
GC system mass models
"""


class GCSMassLinearModel(GCSMassModel):
    def __init__(self, g0=-0.725, g1=0.788, seed=None):
        """
        Implementation of the linear regression model from Eadie+2022
        Source: https://iopscience.iop.org/article/10.3847/153
        """
        super().__init__(seed=seed)

        self.g0 = g0
        self.g1 = g1
        self.kind = "stellar"

    def var_names(self):
        return ["g0", "g1"]

    def mass(self, stellar_mass):
        return 10 ** (self.g0 + self.g1 * np.log10(stellar_mass))


class GCSMassHarrisModel(GCSMassModel):
    def __init__(self, g0=-0.725, g1=0.788, scatter=0.0, seed=None):
        """
        Implementation of the Harris halo mass–GCS mass relation from
        Harris, Blakeslee, & Harris (2017) paper.
        """

        super().__init__(seed=seed)

        self.g0 = g0
        self.g1 = g1
        self.scatter = scatter
        self.kind = "halo"

    def var_names(self):
        return ["g0", "g1", "scatter"]

    def mass(self, halo_mass):
        eta = 2.9e-5
        mhalo = eta * halo_mass

        if self.scatter > 0:
            log_scatter = self.scatter * np.random.normal(0, 1, size=np.shape(mhalo))
            log_eta = np.log10(eta) + log_scatter
            return 10**log_eta * halo_mass
        else:
            return eta * halo_mass


class GCSMassDornanModel(GCSMassModel):
    def __init__(self, slope=0.9257, intercept=-3.5645, scatter=0.3, seed=None):
        """
        Implementation of the Dornan and Harris (2026) halo mass–GCS mass relation
        refit to the Dornan dwarf galaxy catalog (except for Fornax Deep Survey)
        """

        super().__init__(seed=seed)

        self.slope = slope
        self.intercept = intercept
        self.scatter = scatter
        self.kind = "halo"

    def var_names(self):
        return ["slope", "intercept", "scatter"]

    def mass(self, halo_mass, seed=None):
        if seed is not None:
            np.random.seed(seed)

        if self.scatter > 0:
            log_scatter = self.scatter * np.random.normal(
                0, 1, size=np.shape(halo_mass)
            )
            log_mgc = self.intercept + self.slope * np.log10(halo_mass) + log_scatter
            return 10**log_mgc
        else:
            return 10 ** (self.intercept + self.slope * np.log10(halo_mass))


class GCSDornanMassInSitu(GCSMassModel):
    def __init__(
        self,
        slope=0.9257,
        intercept=-3.5645,
        scatter=0.3,
        masses=None,
        z_eval=None,
        seed=None,
    ):
        """
        In-situ GCS mass model using the Dornan mass-halo relation evaluated along
        each halo's mean MAH, then interpolated over (halo_mass, redshift).
        """
        super().__init__(seed=seed)
        self.kind = "halo"
        self.slope = slope
        self.intercept = intercept
        self.scatter = scatter

        if masses is None:
            masses = np.logspace(7, 14, 50)
        if z_eval is None:
            z_eval = np.linspace(0, 20, 100)

        self._interp = self._build_interpolator(masses, z_eval)

    def _build_interpolator(self, masses, z_eval):
        dornan = GCSMassDornanModel(slope=self.slope, intercept=self.intercept, scatter=0.0)
        log_mhalo_pts, t_pts, log_mgcs_vals = [], [], []

        for mass in masses:
            zs, m_track = mean_mah(mass, z_eval=z_eval)
            log_mgcs = np.log10(dornan.mass(m_track[0]))
            t = np.array([cosmo.age(z) for z in zs])
            log_mhalo_pts.extend(np.log10(m_track))
            t_pts.extend(t)
            log_mgcs_vals.extend(np.full(len(zs), log_mgcs))

        return LinearNDInterpolator(
            np.column_stack([log_mhalo_pts, t_pts]),
            np.array(log_mgcs_vals),
        )

    def var_names(self):
        return ["slope", "intercept", "scatter"]

    def mass(self, halo_mass, z=0.0, cosmology=None):
        if cosmology is None:
            cosmology = cosmo
        halo_mass = np.atleast_1d(np.asarray(halo_mass, dtype=float))
        t = np.full_like(halo_mass, cosmology.age(z))
        log_mgcs = self._interp(np.log10(halo_mass), t)

        if self.scatter > 0:
            log_mgcs += self.scatter * np.random.normal(0, 1, size=log_mgcs.shape)

        return 10**log_mgcs


class GCSDornanMixture(GCSMassModel):
    def __init__(
        self,
        alpha=0.5,
        slope=0.9257,
        intercept=-3.5645,
        scatter=0.3,
        masses=None,
        z_eval=None,
        z_form_weight=False,
        age_model=None,
        seed=None,
    ):
        """
        Mixture of the Dornan in-situ and ex-situ GCS mass models.

        log_mgcs = log_insitu * (log_exsitu / log_insitu) ** alpha

        alpha=0 recovers the pure in-situ model; alpha=1 recovers the pure ex-situ model.

        If z_form_weight=True, the GCS mass is weighted by the fraction of GC formation
        occurring after infall:

            M_gcs_weighted = M_gcs(M_peak, z_infall) * S(z_infall)

        where S(z) = integral_z^{inf} p(z_form) dz_form comes from `age_model`
        (defaults to ValcinAgeModel).
        """
        super().__init__(seed=seed)
        self.kind = "halo"
        self.alpha = alpha
        self.z_form_weight = z_form_weight
        self.age_model = age_model if age_model is not None else ValcinAgeModel()

        self._insitu = GCSDornanMassInSitu(
            slope=slope, intercept=intercept, scatter=0.0,
            masses=masses, z_eval=z_eval, seed=seed,
        )
        self._exsitu = GCSMassDornanModel(slope=slope, intercept=intercept, scatter=0.0)
        self.scatter = scatter

    def var_names(self):
        return ["alpha", "slope", "intercept", "scatter", "z_form_weight"]

    def mass(self, halo_mass, z=0.0, cosmology=None):
        if cosmology is None:
            cosmology = cosmo
        halo_mass = np.atleast_1d(np.asarray(halo_mass, dtype=float))
        t = cosmology.age(z)

        log_insitu = np.log10(self._insitu.mass(halo_mass, z=z, cosmology=cosmology))
        log_exsitu = np.log10(self._exsitu.mass(halo_mass))

        # M_0 * (M_inf / M_0)^alpha in log space
        log_mgcs = log_insitu + self.alpha * (log_exsitu - log_insitu)

        if self.scatter > 0:
            log_mgcs += self.scatter * np.random.normal(0, 1, size=log_mgcs.shape)

        if self.z_form_weight:
            weight = self.age_model.cdf(t)
            log_mgcs += np.log10(weight)

        return 10**log_mgcs


"""
GC luminosity functions
"""


class GCMFGeorgiev(GaussianGCLF):
    def __init__(self, log_ml=0.3, log_ml_sigma=0.0, seed=None):
        """
        Implementation of the GCMF from Georgiev+2009
        Source: https://ui.adsabs.harvard.edu/abs/2009MNRAS.392..879G/abstract
        => Valid up to galaxy stellar masses of ~1e9 Msun
        """
        mu_V = -7.04
        sigma_V = 1.15
        V_sun = 4.80

        super().__init__(
            mu=mu_V, sigma=sigma_V, M_sun=V_sun, log_ml=log_ml, log_ml_sigma=log_ml_sigma, seed=seed
        )


class GCMFElves(GaussianGCLF):
    def __init__(self, log_ml=0.3, log_ml_sigma=0.0, seed=None):
        """
        Implementation of the GCMF from ELVES (Carlsten+22a; https://www.arxiv.org/abs/2105.03440).
        """
        mu_g = -7.02
        sigma_g = 0.57
        g_sun = 5.05

        super().__init__(
            mu=mu_g, sigma=sigma_g, M_sun=g_sun, log_ml=log_ml, log_ml_sigma=log_ml_sigma, seed=seed
        )


class GCMFVillegas(GaussianGCLF):
    def __init__(self, halo_mass=1e12, log_ml=0.297, log_ml_sigma=0.0, seed=None):
        """
        Implementation of the GCMF from Villegas+2010, which depends on galaxy mass
        """
        self.kind = "halo"
        mu_g = self._mu_g(halo_mass, seed=seed)
        sigma_g = self._sigma_g(halo_mass, seed=seed)
        g_sun = 5.09 # HST F475W
        super().__init__(
            mu=mu_g, sigma=sigma_g, M_sun=g_sun, log_ml=log_ml, log_ml_sigma=log_ml_sigma, seed=seed
        )
        self.kind = "halo"

    """
    Fits to the Villegas+2010 GCLF parameters as a function of galaxy mass, with scatter
    """

    def _mu_g(self, mpeak, seed=None):
        if seed is not None:
            np.random.seed(seed)
        x0 = 11.8
        y0 = -7.29
        m = -0.09
        sqrtV = 0.18
        scatter = np.random.normal(0, 1) * sqrtV
        return y0 + m * (np.log10(mpeak) - x0) + scatter

    def _sigma_g(self, mpeak, seed=None):
        if seed is not None:
            np.random.seed(seed)
        x0 = 11.8
        y0 = 0.95
        m = 0.34
        sqrtV = 0.15
        scatter = np.random.normal(0, 1) * sqrtV
        return y0 + m * (np.log10(mpeak) - x0) + scatter
    
    def set_halo_mass(self, halo_mass, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.mu = self._mu_g(halo_mass, seed=seed)
        self.sigma = self._sigma_g(halo_mass, seed=seed)
        log_M_star, sigma_logM, mean_mass = self.compute_mass_parameters(self.mu, self.sigma, self.log_ml, self.log_ml_sigma)
        self.mean_mass = mean_mass
        self.icdf = _lognormal_icdf(log_M_star, sigma_logM)

class GChordsMassLightRatioModel(MassLightRatioModel):
    def __init__(self):
        self.mu1    = 0.612
        self.mu2    = 0.67
        self.sigma1 = 0.173
        self.sigma2 = 0.586
        self.w      = 0.864

    def pdf(self, y):
        p1 = self.w * norm.pdf(y, loc=self.mu1, scale=self.sigma1)
        p2 = (1 - self.w) * norm.pdf(y, loc=self.mu2, scale=self.sigma2)
        return p1 + p2

    def sample(self, n_draws=1):
        component = np.random.choice([0, 1], size=n_draws, p=[self.w, 1 - self.w])
        return np.where(
            component == 0,
            np.random.normal(self.mu1, self.sigma1, size=n_draws),
            np.random.normal(self.mu2, self.sigma2, size=n_draws),
        )

    def var_names(self):
        return ["mu1", "mu2", "sigma1", "sigma2", "w"]

class GCHaloModel:
    def __init__(self,
                 occupation_model,
                 mass_model,
                 gclf_model,
                 nimbus_model, 
                 mass_to_light_ratio_model,
                 seed=None):
        
        self.occupation_model = occupation_model
        self.mass_model = mass_model
        self.gclf_model = gclf_model
        self.nimbus_model = nimbus_model
        self.mass_to_light_ratio_model = mass_to_light_ratio_model

        if seed is not None:
            np.random.seed(seed)

        self.required_inputs = ["halo_mass", "stellar_mass"]

    def var_names(self):
        return {
            "occupation_model": self.occupation_model.var_names(),
            "mass_model": self.mass_model.var_names(),
            "gclf_model": self.gclf_model.var_names(),
            "mass_to_light_ratio_model": self.mass_to_light_ratio_model.var_names(),
        }

    def generate(self, **kwargs):
        """
        Returns a tuple (bool: has_gc, int: gc_count, list: gc_masses, list: gc_luminosities)
        """

        halo_mass = kwargs.get("halo_mass")
        stellar_mass = kwargs.get("stellar_mass")
        z = kwargs.get("z", None)

        if halo_mass is None:
            raise ValueError("halo_mass not supplied")
        if stellar_mass is None:
            raise ValueError("stellar_mass not supplied")

        occ_mass = halo_mass if self.occupation_model.kind == "halo" else stellar_mass
        has_gc = self.occupation_model.has_gc(occ_mass, z=z)

        _empty = np.array([])

        if not has_gc:
            return False, 0, _empty, _empty

        mass_input = halo_mass if self.mass_model.kind == "halo" else stellar_mass
        mass_kwargs = {"z": z} if z is not None else {}
        gc_mass = self.mass_model.mass(mass_input, **mass_kwargs)

        if gc_mass <= 0:
            return True, 0, _empty, _empty

        if self.gclf_model.kind == "halo":
            self.gclf_model.set_halo_mass(halo_mass)

        lam = gc_mass / self.gclf_model.mean_mass

        if lam <= 0 or np.isnan(lam):
            print(f"ERROR: gc_mass / mean_gc_mass = {lam}")
            return True, 0, _empty, _empty

        n_draws = np.random.poisson(lam)

        gc_masses, gc_luminosities = self.gclf_model.sample(n_draws, return_L=True)

        return True, n_draws, gc_masses, gc_luminosities
    

GC_HALO_MODEL = symlib.GalaxyHaloModel(
    symlib.StellarMassModel(symlib.UniverseMachineMStarFit(), symlib.DarkMatterSFH()),
    symlib.ProfileModel(GlobularClusterRhalf(), symlib.PlummerProfile()),
    symlib.MetalModel(
        symlib.Kirby2013Metallicity(),
        symlib.Kirby2013MDF(model_type="gaussian"),
        symlib.GaussianCoupalaCorrelation(),
    ),
)

class FiducialGCHaloModel(GCHaloModel):
    def __init__(self):
        super().__init__(
            occupation_model=EadieOccupationModel(),
            mass_model=GCSMassLinearModel(),
            gclf_model=GCMFVillegas(),
            nimbus_model=GC_HALO_MODEL,
            mass_to_light_ratio_model=FlexibleMassLightRatioGCLF()
        )
