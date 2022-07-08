"""Motion structure classification based on the trials from Yang et al. (2021)"""

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

# # #
# Usage: Enter the participant id (pid) below, then run the "simulation" which will use the trials from (Yang et al., 2021).
#        Note down the dataset label (DSL), e.g., in the dict in the file '../ana/YangETAL_dsl.py'
#        Repeat for all 12 participants. These DSL's will be used later for data analysis.
# # # 

# Load default parameters (tau_s, tau_lam, fps, sig_obs,...)
from default_parameters_object_indexed import *

# A base name for the dataset label
human_readable_dsl = "040_Yang_2021"

D = 1                   # number of spatial dimensions
fps = 50.0              # observations per second (from Yang et al's experiments / stimuli)
# We run a grid search over a set of reasonable values
sig_obs = [0.020, 0.025, 0.030, 0.035, 0.040]      # location observation noise (dt-independent)

C = np.array([[1,1,0,1,1,0,0],                          # Motion features
              [1,1,1,0,0,1,0],
              [1,0,1,1,0,0,1]], dtype=np.float64)

# Calculate the average lambda values across structures
lamTot = 2.                                 # Total speed; = ind. component in I condition
lamI = 1./4.                                # Small independent component in G, C, H condition
lamGC = np.sqrt(lamTot**2 - lamI**2)        # global/cluster component in G and C condition
lamGH = np.sqrt(3/4)*lamGC                  # global component in H
lamCH = np.sqrt(1/4)*lamGC                  # cluster component in H
lamIH = np.sqrt(lamTot**2 - lamGH**2)       # Maverick's independent component in H
# lam0 = prior = average across structures (for C, only 1 of 3 will be nonzero)
lam0 = np.array([(lamGC + lamGH)/4]*1 + [(lamGC+lamCH)/(3*4)]*3 + [(lamTot+lamI+(2*lamI+lamTot)/3+(2*lamI+lamIH)/3)/4]*3)


# # #  SELECT THE PARTICIPANT (pid=1..12) HERE  # # #
from strinf.forYangETAL import PresentYangTrials
PYT = PresentYangTrials(pid=12, K=3)

@trialNumberAware
def f_v(t, trialNumber):
    return PYT(t, trialNumber)



# # # # # # # # # # # # # #
# # # AUTO PARAMETERS # # #
# # # # # # # # # # # # # #

# Create dataset label
human_readable_dsl += f"_{PYT.pid:02d}"
dsl = create_dsl(human_readable_dsl)
K,M = C.shape                   # number of objects, motion features


# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # #    BUILD THE ACTUAL CONFIG DICTIONARY   # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# The actual config dict used by the simulation
cfg = {
    # GLOABAL PARAMETERS
    "glo" : dict(
        DRYRUN = False,           # If true, nothing will be saved to disk.
        PLOTAFTER = True,        # If true, some basic results will be plotted after the simulation
        loglevel = "INFO",      # level of logging ["DEBUG", "INFO", "WARNING", "ERROR"]
        dsl = dsl,               # dataset label
        outdir = "./data/%s" % dsl,   # output directory
        R = 200,                   # num repetitions (trials)
        T = 4.0,                  # total time per trial
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
        sig_obs = 0.,                       # Like Yang et al, sig_obs=0 for the presentation
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
        "default_params" : dict(
            D = D,
            C = C,
            lam0 = lam0,
            tau_s = tau_s,
            inf_method = inf_method_MAP_sliding_lam2,
            inf_method_kwargs = dict(fps=fps, tau_s=tau_s, tau_lam=tau_lam, kappa=kappa, nu=nu),
            )
        },

    }

# ADD THE FILTERS FOR DIFFERENT SIG_OBS
for s in sig_obs:
    cfg["fil"][f"adiab_{s:.3f}"] = dict(cls = FilterStrinfAdiabaticDiagonalSigma,
                                        sig_obs = s,
                                       )



# Specific to Yang (not used for inferring the lambda; only stored for later analysis)
labeldict = {
    'ground_truth' : PYT.ground_truth,
    'choice' : PYT.choice,
    'confidence' : PYT.confidence,
    }

cfg['label'] = labeldict
