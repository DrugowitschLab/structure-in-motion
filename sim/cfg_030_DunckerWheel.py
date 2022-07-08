"""Decomposition into shared linear motion and an orbiting dot, as in the Duncker Wheel"""

import numpy as np
from datetime import datetime
from strinf import WorldVeloFunction, ObservationGeneratorVelo, \
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
human_readable_dsl = "020_DunckerWheel"

D = 2                   # number of spatial dimensions
speedmultilier = 1.0    # convenience parameter

lam0 = 0.1              # To demonstrate how the components emerge, we reduce the initial strengths
sig_obs = 0.15          # and increase the noise

C = np.array([
    [1, 1,0],
    [1, 0,1],
    ], dtype=np.float64)

# Velocity function
R, vx = 1.0, 2 * np.pi       # Radius and shared velocity
def f_v(t):
    w = vx / R              # derive rotation frequency for "slip-free" rolling (just physics)
    v = np.array([[vx + R * w * np.cos(w*t), vx],
                  [0. - R * w * np.sin(w*t), 0.]])
    return v * speedmultilier


# # # # # # # # # # # # # #
# # # AUTO PARAMETERS # # #
# # # # # # # # # # # # # #

dsl = create_dsl(human_readable_dsl)     # Create dataset label
K,M = C.shape                            # number of objects, motion features


# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # #    BUILD THE ACTUAL CONFIG DICTIONARY   # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# The actual config dict
cfg = {
    # GLOABAL PARAMETERS
    "glo" : dict(
        DRYRUN = False,           # If true, nothing will be saved to disk.
        PLOTAFTER = False,        # If true, some basic results will be plotted after the simulation
        loglevel = "INFO",      # level of logging ["DEBUG", "INFO", "WARNING", "ERROR"]
        dsl = dsl,               # dataset label
        outdir = "./data/%s" % dsl,   # output directory
        R = 1,                   # num repetitions (trials)
        T = 20.,                  # total time per trial
        PRINTEVERY = 1.,
        ),
    # WORLD SIMULATION (dot movement)
    "wld" : dict(
        cls = WorldVeloFunction,         # World class
        M = M,                           # for shape-compatibility only
        dt = 0.001,
        f_v = f_v,
        seed = None,
        Mrot = None,
        ),
    # OBSERVATIONS (from world state)
    "obs" : dict(
        cls = ObservationGeneratorVelo,
        dt = 1./fps,
        seed = 0xDECAFBAD,
        sig_obs = sig_obs,
        final_precise = False,
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

