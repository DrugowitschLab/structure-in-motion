"""Motion direction repulsion as a function of contrast of the 2nd component (here: noise), as in Chen et al. (2005), Fig. 7"""

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

# Factors: 0.001, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0
noise_factor = 10.0      # for technical reasons, we have to run each noise separately

# A base name for the dataset label
human_readable_dsl = f"103_repulsion_by_contrast__noise_factor_{noise_factor:.3f}"

# Load default parameters
from default_parameters_location_indexed import *

D = 2                  # number of spatial dimensions
Alpha = np.array([15., 20., 22.5, 25., 30., 45., 60., 75., 90.]) * (np.pi/180)  # at which opening angle
lam_tot = 2.0          # equivalent speed of stim
reps = 20              # trial repetitions (for each angle)


sig_obs = sig_obs * np.ones((2,3))
sig_obs[:,2] *= 3                            # <-- noise of proprioception
sig_obs[:,1] *= 1./np.sqrt(noise_factor)     # only the 2nd component gets affected by the noise
sig_obs = sig_obs.flatten()

C = np.array([
    [1,1,1,0],
    [1,1,0,1],
    [1,0,0,0],          # <-- self-motion
    ], dtype=np.float64)


@trialNumberAware
def f_v(t, trialNumber):
    # 2-level loop: alpha, repetition
    alpha = Alpha[trialNumber//reps]
    # print(trialNumber, alpha/np.pi*180, )
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
        PLOTAFTER = False,        # If true, some basic results will be plotted after the simulation
        loglevel = "INFO",      # level of logging ["DEBUG", "INFO", "WARNING", "ERROR"]
        dsl = dsl,               # dataset label
        outdir = "./data/%s" % dsl,   # output directory
        R = len(Alpha) * reps,        # num repetitions (trials)
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
            lam0 = lam0 * np.ones(M), # (0.5*np.ones(M) + np.random.uniform(-0.05,0.05,M)) * f_speed,
            tau_s = tau_s,
            sig_obs = sig_obs,
            inf_method_kwargs = dict(fps=fps, tau_s=tau_s, tau_lam=tau_lam, kappa=kappa, nu=np.array([-2./D,nu,nu,nu]) ),
            )
        },

    }

