from abc import ABC, abstractmethod
import numpy as np
import agama
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

agama.setUnits(mass=1.0, length=1.0, velocity=1.0)

class MassLossModel(ABC):
    def __init__(self, relaxation_model, stellar_evolution_model):
        self.relaxation_model = relaxation_model
        self.stellar_evolution_model = stellar_evolution_model
        self.has_closed_form = False

    @abstractmethod    
    def evolve_mass(self, initial_mass, time, tidal_strength, integrated=False):
        if integrated and not self.has_closed_form:
            raise ValueError("This mass loss model does not have a closed-form solution for the integrated mass loss. Set `integrated=False`.")

class MassLossComponent(ABC):
    @abstractmethod
    def mass_loss_rate(self, cluster_mass, tidal_strength):
        pass


class StellarEvolutionMassLossModel(MassLossComponent):
    def __init__(self):
        pass


class RelaxationMassLossModel(MassLossComponent):
    def __init__(self):
        pass

    def mass_loss_rate(self, cluster_mass, tidal_strength, *args, **kwargs):
        # compressive (extensive=False) tidal fields do not drive mass loss
        if tidal_strength <= 0:
            return 0.0
        return self._mass_loss_rate(cluster_mass, tidal_strength, *args, **kwargs)

    @abstractmethod
    def _mass_loss_rate(self, cluster_mass, tidal_strength, *args, **kwargs):
        pass


class NoStellarEvolutionModel(StellarEvolutionMassLossModel):
    '''
    A mass loss model that does not include stellar evolution effects.
    Valid if you assume that all cluster masses in your simulation have "accounted" for
    the mass loss from stellar evolution (i.e., using some prescription for the 'evolved' GCLF after instantaneous stellar evolution).
    '''
    def mass_loss_rate(self, cluster_mass, tidal_strength):
        return 0.0


class FiducialRelaxationMassLossModel(RelaxationMassLossModel):
    """
    Original equation:
    M(t) = \left(M_0^{2/3} - \frac{2A}{3} \int_{t_0}^{t} \sqrt{\frac{\kappa}{3} \lambda (\tau)} d\tau\right)^{3/2}

    I absorb sqrt(kappa / 3) into the A term:
    M(t) = \left(M_0^{2/3} - \frac{2A}{3} \int_{t_0}^{t} \sqrt{\lambda (\tau)} d\tau\right)^{3/2}

    And I save the spline integral of sqrt(lambda(tau)) dtau as:
    X(t)=\int_{t_0}^{t} \sqrt{\lambda (\tau)} d\tau

    M(t) = \left(M_0^{2/3} - \frac{2AX(t)}{3} \right)^{3/2}
    """

    def __init__(self, kappa=1.0):
        self.kappa = kappa

    def _mass_loss_rate(self, m_cl, tidal_strength):
        return -m_cl / (10 * (m_cl / 2e5) ** (2.0 / 3.0) / (np.sqrt(self.kappa * tidal_strength / 3) / 100.0))


class MamikonyanRelaxationMassLossModel(RelaxationMassLossModel):
    """
    Based on Mamikonyan+2017 (ApJ 837, 70), continuous form of Eq. 41/44:

        dM/dt = -A*sqrt(alpha) - B*(dalpha/dt)/sqrt(alpha)*M

    active only while alpha > 0; the B term only engages while alpha is increasing.
    """
    def __init__(self, A=3.4, B=0.135):
        self.A = A
        self.B = B

    def _mass_loss_rate(self, m_cl, alpha, dalpha_dt):
        if m_cl <= 0:
            return 0.0
        rate = self.A * np.sqrt(alpha)
        if dalpha_dt > 0:
            rate += self.B * dalpha_dt / np.sqrt(alpha) * m_cl
        return -rate

class FiducialMassLossModel(MassLossModel):
    def __init__(self, kappa=1.0):
        super().__init__(relaxation_model=FiducialRelaxationMassLossModel(kappa=kappa), stellar_evolution_model=NoStellarEvolutionModel())
        self.has_closed_form = True

    def evolve_mass(self, initial_mass, time, tidal_strength, integrated=False):

        A = np.sqrt(self.relaxation_model.kappa / 3.0) * (2e5)**(2/3) / 1000

        if integrated:
            # `tidal_strength` is the time-integrated tidal strength
            x = np.maximum(0.0, initial_mass**(2/3) - (2 * A / 3) * tidal_strength)
            return x**(3/2)
        else:
            # time is the time when the tidal field is evaluated, and `tidal_strength` is the instantaneous tidal strength at that time
            # numerically integrate the mass loss rate over time to get the evolved mass at the given time

            interpolated_strength = PchipInterpolator(time, tidal_strength)

            def dMdt(t, m_cl):
                return self.relaxation_model.mass_loss_rate(m_cl, interpolated_strength(t))

            sol = solve_ivp(dMdt, (time[0], time[-1]), [initial_mass], t_eval=[time[-1]])
            return float(sol.y[0][0])


class MamikonyanMassLossModel(MassLossModel):
    def __init__(self, A=3.4, B=0.135):
        super().__init__(
            relaxation_model=MamikonyanRelaxationMassLossModel(A=A, B=B),
            stellar_evolution_model=NoStellarEvolutionModel(),
        )

    def evolve_mass(self, initial_mass, time, tidal_strength, integrated=False):
        super().evolve_mass(initial_mass, time, tidal_strength, integrated)

        alpha_spline = PchipInterpolator(time, tidal_strength)
        dalpha_spline = alpha_spline.derivative()

        def dMdt(t, M):
            if M[0] <= 0:
                return [0.0]
            alpha = float(alpha_spline(t))
            if alpha <= 0:
                return [0.0]
            dalpha_dt = float(dalpha_spline(t))
            return [self.relaxation_model.mass_loss_rate(M[0], alpha, dalpha_dt)]

        def dissolved(t, M):
            return M[0]
        dissolved.terminal = True
        dissolved.direction = -1

        dt = time[1] - time[0]
        sol = solve_ivp(
            dMdt,
            (time[0], time[-1]),
            [initial_mass],
            method='RK45',
            events=dissolved,
            max_step=dt,
        )

        return float(np.clip(sol.y[0][-1], 0.0, None))


class CircularOrbitRelaxationMassLossModel(RelaxationMassLossModel):
    r"""
    Relaxation-driven mass loss in the Gieles & Baumgardt (2008) form, with the tidal frequency replaced by an
    effective one that interpolates between the cluster's own tidal field and that of the circular orbit with the
    same energy:

        \dot{M}/M = - \tilde{\omega}(t) / (\sqrt{2} A_GB) * (\ln(\gamma M / \bar{m}) / (M / M_sun))^{3/4}
        \tilde{\omega}(t) = \omega_circ(t, E) * (\omega(t) / \omega_circ(t, E))^k

    with omega = sqrt(tidal strength) in Gyr^-1 along the cluster's orbit, and omega_circ = sqrt(tidal strength) of
    the circular orbit at the radius R_c(E, t) whose circular energy equals the cluster's energy E (see
    GChords.compute_cluster_circular_tidal_field). k = 1 recovers the instantaneous tidal field, k = 0 the circular
    orbit's. A_GB ~ 10^-0.55 and k ~ 2 are fit from N-body simulations.
    """
    def __init__(self, A_GB=10**-0.55, k=2.0, gamma=0.02, m_bar=0.5):
        self.A_GB = A_GB
        self.k = k
        self.gamma = gamma
        self.m_bar = m_bar

    def omega_tilde(self, omega, omega_circ):
        return omega_circ**(1 - self.k) * omega**self.k

    def mass_loss_rate(self, cluster_mass, omega, omega_circ):
        # dM/dt in Msun / Gyr for omega, omega_circ in Gyr^-1
        if omega <= 0 or omega_circ <= 0:
            return 0.0
        return self._mass_loss_rate(cluster_mass, omega, omega_circ)

    def _mass_loss_rate(self, m_cl, omega, omega_circ):
        log_n = np.log(self.gamma * m_cl / self.m_bar)
        if m_cl <= 0 or log_n <= 0:
            return 0.0
        return -m_cl * self.omega_tilde(omega, omega_circ) / (np.sqrt(2) * self.A_GB) * (log_n / m_cl)**0.75


class CircularOrbitMassLossModel(MassLossModel):
    r"""
    Mass loss from CircularOrbitRelaxationMassLossModel (no stellar evolution).

    The equation is separable, dM / (M^{1/4} \ln(\gamma M/\bar{m})^{3/4}) = -\tilde{\omega}(t) dt / (\sqrt{2} A_GB),
    so M(t) is found exactly by inverting
        F(M) = \int_{\bar{m}/\gamma}^{M} dm / (m^{1/4} \ln(\gamma m/\bar{m})^{3/4})
             = 4 (\bar{m}/\gamma)^{3/4} \int_0^{w} e^{3 w'^4 / 4} dw',   w = \ln(\gamma M/\bar{m})^{1/4},
    i.e. F(M(t)) = F(M_0) - X(t) / (\sqrt{2} A_GB) with X(t) = \int \tilde{\omega} dt (dimensionless). The cluster
    has dissolved (mass 0) once F reaches 0, i.e. at M = \bar{m}/\gamma where the Coulomb logarithm vanishes.

    evolve_mass needs the circular-orbit tidal strength at the cluster's tracked snapshots (circular_time,
    circular_tidal_strength; from GChords.compute_cluster_circular_tidal_field), which is interpolated linearly in
    log(tidal strength) onto the orbit's time samples. Times are in Agama units (as written by
    compute_cluster_tidal_field) and converted with time_unit_gyr; tidal strengths are in Gyr^-2.
    """
    requires_circular_tidal_field = True

    def __init__(self, A_GB=10**-0.55, k=2.0, gamma=0.02, m_bar=0.5, time_unit_gyr=0.97779222, m_max=1e10,
                 n_table=20001):
        super().__init__(
            relaxation_model=CircularOrbitRelaxationMassLossModel(A_GB=A_GB, k=k, gamma=gamma, m_bar=m_bar),
            stellar_evolution_model=NoStellarEvolutionModel(),
        )
        self.time_unit_gyr = time_unit_gyr
        self.m_dissolve = m_bar / gamma
        # F tabulated on w = ln(gamma M / m_bar)^(1/4), where the integrand 4 e^{3 w^4 / 4} is smooth
        from scipy.integrate import cumulative_simpson
        self._w = np.linspace(0.0, np.log(m_max / self.m_dissolve)**0.25, n_table)
        self._F = self.m_dissolve**0.75 * cumulative_simpson(4 * np.exp(0.75 * self._w**4), x=self._w, initial=0.0)

    def F(self, m):
        m = np.asarray(m, dtype=float)
        w = np.log(np.maximum(m, self.m_dissolve) / self.m_dissolve)**0.25
        return np.interp(w, self._w, self._F)

    def F_inverse(self, F):
        w = np.interp(F, self._F, self._w)
        return self.m_dissolve * np.exp(w**4)

    def omega_tilde(self, time, tidal_strength, circular_time, circular_tidal_strength):
        '''omega_tilde (Gyr^-1) at the orbit's time samples.'''
        lam = np.clip(np.asarray(tidal_strength, dtype=float), 0.0, None)
        t_c = np.asarray(circular_time, dtype=float)
        lam_c = np.asarray(circular_tidal_strength, dtype=float)
        ok = np.isfinite(t_c) & np.isfinite(lam_c) & (lam_c > 0)
        if not np.any(ok):
            # no circular-orbit information (e.g. no circular table at these snapshots): fall back to omega
            return np.sqrt(lam)
        lam_c = np.exp(np.interp(time, t_c[ok], np.log(lam_c[ok])))
        return self.relaxation_model.omega_tilde(np.sqrt(lam), np.sqrt(lam_c))

    def evolve_mass(self, initial_mass, time, tidal_strength, integrated=False, circular_time=None,
                    circular_tidal_strength=None):
        '''
        Mass at time[-1] of a cluster of initial_mass at time[0].

        integrated=True: tidal_strength is X = \int omega_tilde dt (dimensionless, time in Gyr), and initial_mass
        and X may be arrays. Otherwise time / tidal_strength are the orbit's samples (Agama time units, Gyr^-2) and
        circular_time / circular_tidal_strength the circular-orbit tidal strength at its snapshots.
        '''
        if integrated:
            X = np.asarray(tidal_strength, dtype=float)
        else:
            if circular_time is None or circular_tidal_strength is None:
                raise ValueError("CircularOrbitMassLossModel needs circular_time and circular_tidal_strength "
                                 "(see GChords.compute_cluster_circular_tidal_field).")
            time = np.asarray(time, dtype=float)
            g = self.omega_tilde(time, tidal_strength, circular_time, circular_tidal_strength)
            X = np.trapz(g, time) * self.time_unit_gyr

        m0 = np.asarray(initial_mass, dtype=float)
        F_t = self.F(m0) - X / (np.sqrt(2) * self.relaxation_model.A_GB)
        m = np.where((F_t > 0) & (m0 > self.m_dissolve), self.F_inverse(np.maximum(F_t, 0.0)), 0.0)
        return float(m) if m.ndim == 0 else m
