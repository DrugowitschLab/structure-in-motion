"""Neural network model. Proposed experiment with varying fraction of shared motion, in Figure 7."""

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
human_readable_dsl = "network_global_sweep_lamG"

lam2Tot = 2.0**2        # Expected Total squared speed, will be split between lamG (global) and lamI (individual)
nLam = 7                # number of lambda's  --> frac := lam2G/lam2Tot = linspace(0, 1, nLam)
TperLam = 10.0           # Time each lambda-frac is presented

fps *= 2                # Get finer temporal samples (otherwise the firing rates look a bit choppy) 

reps = 10                # number of repetitions
D = 2                   # number of spatial dimensions
K = 6                   # <-- increase me later?

sig_obs = sig_obs * np.ones((2,K))
sig_obs = sig_obs.flatten()

polarIdx = None
RFCoord = [(1., k * 2*np.pi/K) for k in range(K)]  # THIS IS ONLY NEEDED IF WE USE ROTATIONAL INPUTS

# Just a global component and the individual components
C = np.hstack(( np.ones((K,1)), np.eye(K)))

# # #   Build the lamList (t_i, lam_i)  # # # 
lamList = [(i*TperLam, np.array([np.sqrt(qi * lam2Tot)] + [np.sqrt((1-qi) * lam2Tot)]*K)) for (i,qi) in enumerate(np.linspace(0, 1, nLam+2)[1:-1])]

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
        PLOTAFTER = True,        # If true, some basic results will be plotted after the simulation 
        loglevel = "INFO",       # level of logging ["DEBUG", "INFO", "WARNING", "ERROR"]
        dsl = dsl,               # dataset label
        outdir = "./data/%s" % dsl,   # output directory
        R = reps,                 # num repetitions (trials)
        T = TperLam * nLam,       # total time per trial
        PRINTEVERY = 0.1,
        ),
    # WORLD SIMULATION (dot movement)
    "wld" : dict(
        # cls = WorldOUSources,         # World class
        cls = WorldOUSourcesLamlist,
        # not for class WorldVeloFunction
        D = D,
        C = C,
        lamList = lamList,
        tau_s = tau_s,
        seed = 0xCAFE,
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
            ),
        "net" : dict(
            cls = RateNetworkMTInputStrinfAdiabaticDiagonalSigma,
            inf_method = inf_method_MAP_sliding_lam2_CTC_interaction,
            # only additional kwargs need to be given (updates default)
            inf_method_kwargs = dict(tau_s=tau_s, tau_lam=tau_lam, kappa=kappa, nu=nu),
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

