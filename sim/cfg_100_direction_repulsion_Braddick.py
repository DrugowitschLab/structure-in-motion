"""Motion direction repulsion as a function of opening angle, as in Braddick et al. (2002)"""

import numpy as np
from datetime import datetime
from strinf import WorldVeloFunction, ObservationGeneratorVelo, \
                   FilterStrinfNaturalParameters, FilterStrinfAdiabaticDiagonalSigma,\
                   inf_method_MAP_sliding_lam2
from strinf import create_dsl, trialNumberAware

np.random.seed(0xBADC0DE)

# # # # # # # # # # # # # #
# # # CORE PARAMETERS # # #
# # # # # # # # # # # # # #

# Load default parameters
from default_parameters_location_indexed import *

# A base name for the dataset label
human_readable_dsl = "100_repulsion_by_angle_Braddick"

D = 2                  # number of spatial dimensions
lam_tot = 2.0          # equivalent speed of stim
nAngle = 33           # number of tested opening angles (spaced equally from 0..180 degrees)
reps = 1              # trial repetitions (for each angle)


sig_obs = sig_obs * np.ones((2,3))
sig_obs[:,2] *= 3                 # <-- noise of proprioception
sig_obs = sig_obs.flatten()


C = np.array([
    [1,1,1,0],
    [1,1,0,1],
    [1,0,0,0],          # <-- self-motion
    ], dtype=np.float64)


# Braddick tests the following angles (we sample more densely):
# Alpha = np.array([  5.625,  11.25 ,  22.5  ,  45.   ,  67.5  ,  90.   , 112.5  ,  135.   , 157.5  , 169.   , 174.   , 180.   ])

@trialNumberAware
def f_v(t, trialNumber):
    alpha = np.linspace(0., 180 * (np.pi/180), nAngle)[trialNumber//reps]  # outer loop: angles, inner loop: reps
    vtot =  np.sqrt(D * tau_s * lam_tot**2 / 2)
    v = np.array([[vtot * np.cos(alpha/2), vtot * np.cos(alpha/2), 0.],
                  [vtot * np.sin(alpha/2),-vtot * np.sin(alpha/2), 0.]])
    return v


# # # # # # # # # # # # # #
# # # AUTO PARAMETERS # # #
# # # # # # # # # # # # # #

# Create dataset label
dsl = create_dsl(human_readable_dsl)       # Create dataset label
K,M = C.shape                              # number of objects, motion features


# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # #    BUILD THE ACTUAL CONFIG DICTIONARY   # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# The actual config dict
cfg = {
    # GLOABAL PARAMETERS
    "glo" : dict(
        DRYRUN = False,           # If true, nothing will be saved to disk.
        PLOTAFTER = True,        # If true, some basic results will be plotted after the simulation
        loglevel = "INFO",      # level of logging ["DEBUG", "INFO", "WARNING", "ERROR"]
        dsl = dsl,               # dataset label
        outdir = "./data/%s" % dsl,   # output directory
        R = nAngle * reps,        # num repetitions (trials)
        T = 30.,                  # total time per trial
        PRINTEVERY = 1.,
        ),
    # WORLD SIMULATION (dot movement)
    "wld" : dict(
        cls = WorldVeloFunction,         # World class
        M = M,    # for shape-compatibility only
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
        tInvisible = None,
        idxInvisible = None,
        ),
    # FILTERS (dict of dicts "filtername" = {param dict} AND one dict "default_params" = {param dict} )
    "fil" : {
        # "exact" : dict(
        #     cls = FilterStrinfNaturalParameters,
        #     # forceOmDiagonal = True,
        #     inf_method = inf_method_MAP_sliding_lam2,
        #     # only additional kwargs need to be given (updates default)
        #     # inf_method_kwargs = dict(fps=fps, tau_s=tau_s, tau_lam=5*tau_s, useSigma=True, kappa=0., nu=-2.),
        #     ),
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
            inf_method_kwargs = dict(fps=fps, tau_s=tau_s, tau_lam=tau_lam, kappa=kappa, nu=np.array([-2./D,nu,nu,nu]) ),
            )
        },

    }

