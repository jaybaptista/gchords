# Globular Clusters in Halos with Orbital Disruption for Streams

`gchords` is a subhalo infall model for accreted globular cluster (GC) formation and evolution. It used to interface with the Symphony suite of Milky Way mass simulations
[(Nadler et al. 2023)](http://arxiv.org/abs/2209.02675) 

## the `Gchords` object

The `Gchords` object contains all the infall modeling methods to perform globular cluster (GC) formation and evolution.

The (main) classes that generate the GC population are:
- `Interface`: a generic interface to the simulation suite to determine the infall properties of subhalos and their particle tracking data.
- `GCHaloModel`: a class that handles the generation of GC populations based on halo (or stellar) mass. It is largely defined by the following subclasses: 
    - `OccupationModel`: is a subhalo massive enough to host a GC system?
    - `GCSMassModel`: what is the infall mass of a GC system?
    - `GCLuminosityFunction`: what is the luminosity distribution of the GC systems?

There are additional classes that facilitate the evolution of the GC systems:
- `AgeModel`: a class that defines the GC formation-redshift distribution p(z_form). Available models:
    - `ValcinAgeModel` (default): Gaussian in lookback time (μ=11.89 Gyr, σ=0.98 Gyr) transformed to redshift space via the mah.py cosmology, with a hard cut at z=20. Based on [Valcin et al. 2025](https://arxiv.org/abs/2503.19481).
    - `KruijssenAgeModel`: tabulated p(z_form) from [Kruijssen et al. 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.486.3180K).
- `Potential`: a class that uses the particle tracking data in the `Interface` to produce a time-evolving potential and evaluate tidal fields.
- `MassLossModel`: a class that defines how a cluster loses mass.

See `examples/example_smw.py` for how these components are used to:

(1) generates an accreted GC population based on infall parameters

(2) constructs a time-evolving potential

(3) evaluates the tidal mass loss of the GC population
