import abc
import numpy as np
from scipy.stats import uniform
from scipy.interpolate import interp1d, LinearNDInterpolator
from scipy.stats import lognorm
import symlib
from scipy.stats import norm
from gchords.tag import GlobularClusterRhalf
from gchords.mah import mean_mah

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
    def p_gc(self, **kwargs):
        pass

    @abc.abstractmethod
    def has_gc(self, **kwargs):
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
        self.mass_sigma = abs(np.random.normal(loc=self.gcmf_std_mu, scale=self.gcmf_std_scale))

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

    def p_gc(self, stellar_mass):
        p = (1 + np.exp(-1 * (self.b0 + self.b1 * np.log10(stellar_mass)))) ** (-1)
        return p

    def has_gc(self, stellar_mass):
        p = self.p_gc(stellar_mass)
        return uniform.rvs() < p


class DornanOccupationModel(OccupationModel):
    def __init__(self, b0=-31.86, b1=3.0, seed=None):
        super().__init__(seed=seed)
        self.kind = "halo"
        self.b0 = b0
        self.b1 = b1

    def var_names(self):
        return ["b0", "b1"]

    def p_gc(self, halo_mass):
        p = 1 / (1 + np.exp(-(self.b0 + self.b1 * np.log10(halo_mass))))
        return p

    def has_gc(self, halo_mass):
        p = self.p_gc(halo_mass)
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
            masses = np.logspace(7, 14, 100)
        if z_eval is None:
            z_eval = np.linspace(0, 12, 100)

        self._interp = self._build_interpolator(masses, z_eval)

    def _build_interpolator(self, masses, z_eval):
        dornan = GCSMassDornanModel(slope=self.slope, intercept=self.intercept, scatter=0.0)
        log_mhalo_pts, a_pts, log_mgcs_vals = [], [], []

        for mass in masses:
            zs, m_track = mean_mah(mass, z_eval=z_eval)
            log_mgcs = np.log10(dornan.mass(m_track[0]))
            a = 1.0 / (1.0 + zs)
            log_mhalo_pts.extend(np.log10(m_track))
            a_pts.extend(a)
            log_mgcs_vals.extend(np.full(len(zs), log_mgcs))

        return LinearNDInterpolator(
            np.column_stack([log_mhalo_pts, a_pts]),
            np.array(log_mgcs_vals),
        )

    def var_names(self):
        return ["slope", "intercept", "scatter"]

    def mass(self, halo_mass, z=0.0):
        halo_mass = np.atleast_1d(np.asarray(halo_mass, dtype=float))
        a = np.full_like(halo_mass, 1.0 / (1.0 + z))
        log_mgcs = self._interp(np.log10(halo_mass), a)

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
        seed=None,
    ):
        """
        Mixture of the Dornan in-situ and ex-situ GCS mass models.

        log_mgcs = log_insitu * (log_exsitu / log_insitu) ** alpha

        alpha=0 recovers the pure in-situ model; alpha=1 recovers the pure ex-situ model.
        """
        super().__init__(seed=seed)
        self.kind = "halo"
        self.alpha = alpha

        self._insitu = GCSDornanMassInSitu(
            slope=slope, intercept=intercept, scatter=0.0,
            masses=masses, z_eval=z_eval, seed=seed,
        )
        self._exsitu = GCSMassDornanModel(slope=slope, intercept=intercept, scatter=0.0)
        self.scatter = scatter

    def var_names(self):
        return ["alpha", "slope", "intercept", "scatter"]

    def mass(self, halo_mass, z=0.0):
        halo_mass = np.atleast_1d(np.asarray(halo_mass, dtype=float))

        log_insitu = np.log10(self._insitu.mass(halo_mass, z=z))
        log_exsitu = np.log10(self._exsitu.mass(halo_mass))
        log_mgcs = log_insitu * (log_exsitu / log_insitu) ** self.alpha

        if self.scatter > 0:
            log_mgcs += self.scatter * np.random.normal(0, 1, size=log_mgcs.shape)

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

class BaumgardtMassLightRatioModel(MassLightRatioModel):
    '''
    '''
    def __init__(self):
        self.mu1 = 0.62
        self.sigma1 = 0.2
        self.mu2 = 0.69
        self.sigma2 = 0.88
        self.w = 0.91

    def pdf(self, y):
        p1 = self.w * norm.pdf(y, loc=self.mu1, scale=self.sigma1)
        p2 = (1 - self.w) * norm.pdf(y, loc=self.mu2, scale=self.sigma2)
        return p1 + p2

    def sample(self, n_draws=1):

        # first choose which Gaussian to sample from
        component = np.random.choice([0, 1], size=n_draws)
        
        # then sample from that Gaussian
        samples = np.where(
            component == 0,
            np.random.normal(self.mu1, self.sigma1, size=n_draws),
            np.random.normal(self.mu2, self.sigma2, size=n_draws)
        )

        return samples  
    
    def var_names(self):
        return super().var_names()

class GChordsMassLightRatioModel(BaumgardtMassLightRatioModel):
    
    def __init__(self):
        self.mu1 = 0.612
        self.mu2 = 0.67
        self.sigma1 = 0.173
        self.sigma2 = 0.586
        self.w = 0.864

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
        # TODO: add redshift evolution

        if halo_mass is None:
            raise ValueError("halo_mass not supplied")
        if stellar_mass is None:
            raise ValueError("stellar_mass not supplied")

        has_gc = self.occupation_model.has_gc(
            halo_mass
            if self.occupation_model.kind == "halo"
            else stellar_mass
        )

        if not has_gc:
            return False, 0, [], []

        gc_mass = self.mass_model.mass(
            halo_mass if self.mass_model.kind == "halo" else stellar_mass
        )

        if gc_mass <= 0:
            return True, 0, None, None

        if self.gclf_model.kind == "halo":
            # update GCLF parameters
            self.gclf_model.set_halo_mass(halo_mass)

        lam = gc_mass / self.gclf_model.mean_mass

        if lam <= 0 or np.isnan(lam):
            print(f"ERROR: gc_mass / mean_gc_mass = {lam}")
            return True, 0, None, None

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
            mass_to_light_ratio_model=GChordsMassLightRatioModel()
        )
