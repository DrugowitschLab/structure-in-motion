"""Noise-dependent motion illusion from Lorenceau (1996)"""

import numpy as np
from datetime import datetime
from strinf import ObservationGeneratorVelo, WorldVeloFunction,\
                   FilterStrinfNaturalParameters, FilterStrinfAdiabaticDiagonalSigma,\
                   inf_method_MAP_sliding_lam2
from strinf import create_dsl

np.random.seed(0xBADC0DE)

# # # # # # # # # # # # # #
# # # CORE PARAMETERS # # #
# # # # # # # # # # # # # #

# Load default parameters
from default_parameters_location_indexed import *

# A base name for the dataset label
human_readable_dsl = "110_lorenceau_1996_low_noise"

D = 2                   # number of spatial dimensions
K = 2*10+1                  # Observables: (K-1) dots plus 1 proprioceptive input
noise_factor = 1.0  #1:low, 25:high    # multiplier for sig_obs of visible dots (not the proprioceptive input)

# # # Draw groups
# Random
# frac = 0.5
# idx = np.random.choice(np.arange(K-1), int(round(frac*(K-1))), replace=False)
# Equal
idx = np.arange((K-1)//2)
G = idx, np.setdiff1d(np.arange(K-1), idx)  # vertical, horizontal

# Obs. noise for dots
sig_obs = sig_obs * np.ones((D,K))
sig_obs[:,-1] *= 3                  # proprioception
sig_obs[:,:-1] *= noise_factor      # dots
sig_obs = sig_obs.flatten()

# Build C matrix: self, global, group 0, group 1, individual
# Cartoon:
# 1 1 1   1
# 1 1 1     1
# 1 1   1     1
# 1 1   1       1
# 1
C = np.hstack(( np.ones(4*K).reshape(K,4), np.vstack((np.eye(K-1), np.zeros(K-1))) ))
C[-1,1:] = 0.
C[G[1],2] = 0.
C[G[0],3] = 0.


a = 0.5                     # oscillation amplitude
na = 0.0                    # noise amplitude (only for rendering videos)
omega = 2*np.pi* 0.83        # main frequency
nomega = 2*np.pi * omega    # noise frequency (= some irrational multiple of omega)
phi = 2 * np.pi * np.random.rand(K)   # noise phase

def f_v(t):
    # main component ( v(t) = dx(t) / dt )
    vx = omega * a * np.cos(omega * t) * np.ones(K)
    vx[G[0]] = 0
    vy = - omega * a * np.sin(omega * t) * np.ones(K)
    vy[G[1]] = 0.
    # noise (orthogonal to main direction)
    vnx = nomega * na * np.cos(nomega * t + phi) * np.ones(K)
    vnx[G[1]] = 0
    vny = nomega * na * np.cos(nomega * t + phi) * np.ones(K)
    vny[G[0]] = 0
    # combine
    v = np.concatenate([vx+vnx, vy+vny]).reshape(D,K)
    v[:,-1] = 0.      # proprio
    return v


# # # # # # # # # # # # # #
# # # AUTO PARAMETERS # # #
# # # # # # # # # # # # # #

# Create dataset label
dsl = create_dsl(human_readable_dsl)         # Create dataset label
K,M = C.shape                                # number of objects, motion features


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
        R = 1,                   # num repetitions (trials)
        T = 30.,                  # total time per trial
        PRINTEVERY = 1.,
        ),
    # WORLD SIMULATION (dot movement)
    "wld" : dict(
        cls = WorldVeloFunction,
        # only for class WorldVeloFunction
        M = M,
        f_v = f_v,
        seed = None,
        dt = 0.001,
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
            inf_method_kwargs = dict(fps=fps, tau_s=tau_s, tau_lam=tau_lam, kappa=kappa, nu=np.array([-2./D,nu,nu,nu] + [nu]*(K-1) )),
            )
        },
    }

