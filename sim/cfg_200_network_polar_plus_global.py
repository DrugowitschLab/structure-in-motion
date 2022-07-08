"""Neural network model. Demo in Figure 7."""

import numpy as np
from datetime import datetime
from strinf import WorldOUSources, WorldOUSourcesLamlist, ObservationGeneratorVelo, WorldVeloFunction,\
                   FilterStrinfNaturalParameters, FilterStrinfAdiabaticDiagonalSigma,\
                   RateNetworkMTInputStrinfAdiabaticDiagonalSigma,\
                   inf_method_none, inf_method_MAP_sliding_lam2, inf_method_MAP_sliding_lam2_CTC_interaction
from strinf import create_dsl

np.random.seed(0xBADC0DE)

# # # # # # # # # # # # # #
# # # CORE PARAMETERS # # #
# # # # # # # # # # # # # #

# Load default parameters
from default_parameters_location_indexed import *
# Add network-specific defaults
from default_parameters_specific_to_network import *

# A base name for the dataset label
human_readable_dsl = "network_demo_polarCoords_and_global"

alpha = 1.0              # Volatility factor
f_speed = 1.0           # Speed factor
tau_s /= alpha

fps *= 2                # Get finer temporal samples (otherwise the firing rates look a bit choppy) 

D = 2                   # number of spatial dimensions
K = 6                   # num receptive fields

sig_obs = sig_obs * np.ones((2,K))
sig_obs = sig_obs.flatten()

polarIdx = [1]  # list(np.arange(1,K+2))
RFCoord = [(1., k * 2*np.pi/K) for k in range(K)]

C = np.hstack(( np.ones((K,2)), np.eye(K)))

lam_star = np.array([2.0] + [0.1]*K)
lam_star *= np.sqrt(alpha) * f_speed

def f_v(t):
    # counter-clockwise global rotation
    v = - np.array([ (R * np.sin(th), - R * np.cos(th)) for k,(R,th) in enumerate(RFCoord)]).T
    if t > 1.0:
        # clockwise rotation
        v = -v
        # optional: expansion
        # v += 1.0 * np.array([ (R * np.cos(th), R * np.sin(th)) for k,(R,th) in enumerate(RFCoord)]).T
    if t > 2.0:
        # add translation in x-direction
        v[0] += 1.0
    return 2 * v


# # # # # # # # # # # # # #
# # # AUTO PARAMETERS # # #
# # # # # # # # # # # # # #

dsl = create_dsl(human_readable_dsl)    # Create dataset label
K,M = C.shape                           # number of objects, motion features


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
        T = 3.0,                  # total time per trial
        PRINTEVERY = 0.1,
        ),
    # WORLD SIMULATION (dot movement)
    "wld" : dict(
        cls = WorldVeloFunction,
        # only for class WorldVeloFunction
        M = C.shape[1],
        f_v = f_v,
        seed = None,
        # for all wld classes
        dt = 0.001,
        polarIdx=polarIdx,
        RFCoord=RFCoord,
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
        "adiab" : dict(
            cls = FilterStrinfAdiabaticDiagonalSigma,
            inf_method = inf_method_MAP_sliding_lam2_CTC_interaction,
            # only additional kwargs need to be given (updates default)
            inf_method_kwargs = dict(fps=fps, tau_s=tau_s, tau_lam=tau_lam, kappa=kappa, nu=nu),
            # inf_method_kwargs = dict(tau_s=tau_s, tau_lam=1*tau_s, kappa=-0.05, nu=-1.5, fps=fps, J0=0.0, useSigma=True),
            ),
        "net" : dict(
            cls = RateNetworkMTInputStrinfAdiabaticDiagonalSigma,
            inf_method = inf_method_MAP_sliding_lam2_CTC_interaction,
            # only additional kwargs need to be given (updates default)
            # inf_method_kwargs = dict(tau_s=tau_s, tau_lam=1*tau_s, kappa=-0.05, nu=-1.5),
            inf_method_kwargs = dict(tau_lam=tau_lam, kappa=kappa, nu=nu),
            # Neural network and coding
            inputNeuronKwargs = inputNeuronKwargs,
            latentNeuronKwargs = latentNeuronKwargs,
            ),
        "default_params" : dict(
            D = D,
            C = C,
            lam0 = lam0 * np.ones(M), 
            tau_s = tau_s,
            sig_obs = sig_obs,
            polarIdx = polarIdx,
            RFCoord = RFCoord,
            inf_method_kwargs = dict(),
            )
        },
    }

