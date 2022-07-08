"""Two-dim. demonstration of model recovery for a nested motion stimulus."""

import numpy as np
from datetime import datetime
from strinf import WorldOUSources, ObservationGeneratorVelo, \
                   FilterStrinfNaturalParameters, FilterStrinfAdiabaticDiagonalSigma,\
                   inf_method_MAP_sliding_lam2
from strinf import create_dsl

np.random.seed(0xBADC0DE)

# # # # # # # # # # # # # #
# # # CORE PARAMETERS # # #
# # # # # # # # # # # # # #

# Load default parameters (tau_s, tau_lam, fps, sig_obs,...)
from default_parameters_object_indexed import *

# A base name for the dataset label
human_readable_dsl = "001_nested_model_recovery"

D = 2                   # number of spatial dimensions
C = np.array([          # Motion features
 [1,+1, 0, 1,0,0,0,0,0,0,0],
 [1,+1, 0, 0,1,0,0,0,0,0,0],
 [1,-1, 0, 0,0,1,0,0,0,0,0],
 [1,-1, 0, 0,0,0,1,0,0,0,0],
 [1, 0,+1, 0,0,0,0,1,0,0,0],
 [1, 0,+1, 0,0,0,0,0,1,0,0],
 [1, 0,-1, 0,0,0,0,0,0,1,0],
 [1, 0,-1, 0,0,0,0,0,0,0,1],
], dtype=np.float64)

lam_star = np.array([4.0, 2.25, 1.75] + [1.00]*8)     # True motion strengths

# tau_lam *= 5.
# nu = -2./D               # For the demo, we use the maximum likelihood estimator (no bias in lambda)

# # # # # # # # # # # # # #
# # # AUTO PARAMETERS # # #
# # # # # # # # # # # # # #

dsl = create_dsl(human_readable_dsl)    # Create dataset label
K,M = C.shape                           # number of objects, motion features


# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # #    BUILD THE ACTUAL CONFIG DICTIONARY   # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# The actual config dict used by the simulation
cfg = {
    # GLOABAL PARAMETERS
    "glo" : dict(
        DRYRUN = False,           # If true, nothing will be saved to disk.
        PLOTAFTER = True,        # If true, some basic results will be plotted after the simulation
        loglevel = "INFO",       # level of logging ["DEBUG", "INFO", "WARNING", "ERROR"]
        dsl = dsl,               # dataset label
        outdir = "./data/%s" % dsl,   # output directory
        R = 1,                   # num repetitions (trials)
        T = 20.,                 # total time per trial
        PRINTEVERY = 1.,         # print info to console
        ),
    # WORLD SIMULATION (dot movement)
    "wld" : dict(
        cls = WorldOUSources,         # World class
        D = D,
        C = C,
        lam = lam_star,
        tau_s = tau_s,
        dt = 0.001,                   # sim time step in the world sim can be finer than the obs. times.
        seed = 0xC0FFEE1A,
        Mrot = None,
        ),
    # OBSERVATIONS (from world state)
    "obs" : dict(
        cls = ObservationGeneratorVelo,
        dt = 1./fps,
        seed = 0xDECAFBAD,
        sig_obs = sig_obs,
        final_precise = False,
        tInvisible = None,
        idxInvisible = None,
        ),
    # FILTERS (dict of dicts "filtername" = {param dict} AND one dict "default_params" = {param dict} )
    "fil" : {
        "exact" : dict(
            cls = FilterStrinfNaturalParameters,
            inf_method = inf_method_MAP_sliding_lam2,
            ),
        "adiab" : dict(
            cls = FilterStrinfAdiabaticDiagonalSigma,
            inf_method = inf_method_MAP_sliding_lam2,
            ),
        "default_params" : dict(
            D = D,
            C = C,
            lam0 = lam0 * np.ones(M),
            tau_s = tau_s,
            sig_obs = sig_obs,
            inf_method_kwargs = dict(fps=fps, tau_s=tau_s, tau_lam=tau_lam, kappa=kappa, nu=nu),
            )
        },

    }

