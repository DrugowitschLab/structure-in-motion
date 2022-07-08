"""3-dot motion structure decomposition like in Johansson (1973) to demonstrate the interaction prior"""

import numpy as np
from datetime import datetime
from strinf import WorldVeloFunction, ObservationGeneratorVelo, \
                   FilterStrinfNaturalParameters, FilterStrinfAdiabaticDiagonalSigma,\
                   inf_method_MAP_sliding_lam2, \
                   inf_method_MAP_sliding_lam2_manual_interaction
from strinf import create_dsl

np.random.seed(0xBADC0DE)

# # # # # # # # # # # # # #
# # # CORE PARAMETERS # # #
# # # # # # # # # # # # # #

# Load default parameters (tau_s, tau_lam, fps, sig_obs,...)
from default_parameters_object_indexed import *

# A base name for the dataset label
human_readable_dsl = "021_3_dots_Johansson_1973_sparsity"

D = 2                   # number of spatial dimensions
lamequiv = 2.0          # lam equivalent of total speed
freq = 0.5              # oscil. freq [Hz]

C = np.array([
    [1,1, 1,0,0],
    [1,1, 0,1,0],
    [1,1, 0,0,1],
    ], dtype=np.float64)

lam0 = lam0 * np.ones(5)
lam0 *= np.array([1.2, 0.8] + [1.0]*3)

phirot = 0.         # np.pi/3   # for the SI, we demonstrate rotation invariance
Mrot = np.array([(np.cos(phirot), -np.sin(phirot)),
                 (np.sin(phirot),  np.cos(phirot))])

alpha = 45 * (np.pi/180)   # angle of central dot

# Velocity function
def f_v(t):
    vtot =  np.sqrt(D * tau_s * lamequiv**2 / 2)
    v = np.array([[1,  1,             1  ],
                  [0., np.cos(alpha), 0. ]])
    return v * np.sin(2*np.pi * freq * t) * vtot


# # # # # # # # # # # # # #
# # # AUTO PARAMETERS # # #
# # # # # # # # # # # # # #

dsl = create_dsl(human_readable_dsl)   # Create dataset label
K,M = C.shape                          # number of objects, motion features

Jinter = np.zeros((M,M))               # HERE IS THE INTERACTION MATRIX
Jinter[0,1] = Jinter[1,0] = 1


# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # #    BUILD THE ACTUAL CONFIG DICTIONARY   # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# The actual config dict
cfg = {
    # GLOABAL PARAMETERS
    "glo" : dict(
        DRYRUN = True,           # If true, nothing will be saved to disk.
        PLOTAFTER = True,        # If true, some basic results will be plotted after the simulation
        loglevel = "INFO",      # level of logging ["DEBUG", "INFO", "WARNING", "ERROR"]
        dsl = dsl,               # dataset label
        outdir = "./data/%s" % dsl,   # output directory
        R = 1,                   # num repetitions (trials)
        T = 120.,                  # total time per trial
        PRINTEVERY = 1.,
        ),
    # WORLD SIMULATION (dot movement)
    "wld" : dict(
        cls = WorldVeloFunction,         # World class
        M = M,                           # for shape-compatibility only
        dt = 0.001,
        f_v = f_v,
        seed = None,
        Mrot = Mrot,
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
        "interaction" : dict(
            cls = FilterStrinfAdiabaticDiagonalSigma,
            inf_method = inf_method_MAP_sliding_lam2_manual_interaction,
            inf_method_kwargs = dict(fps=fps, tau_s=tau_s, tau_lam=tau_lam, J0=10.0, Jinter=Jinter, kappa=kappa, nu=nu),
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

