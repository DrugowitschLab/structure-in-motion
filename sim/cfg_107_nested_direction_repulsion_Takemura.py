"""Motion direction repulsion with 2 inner and 2 outer RDK, as in Takemura et al. (2011)"""

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
human_readable_dsl = "107_direction_repulsion_Takemura"

D = 2                  # number of spatial dimensions
lam_tot = 2.0          # equivalent speed of stim
reps = 200             # trial repetitions
# In Experiment 1, the inner dots have no vertical component.
# In Experiment 2, all inner dots have the same x-speed, and a multiplier of this value on their y-speed.
multy = 1       # in {0,1,2,3,4}, or choose 0 for experiment 1; choose 1 for experiment 2, 90 degree condition
outersign = 1  # in {-1, 0, 1}, -1:down, +1:up, 0:bidirectional (one up, one down)

# We have K=5 observables: 1st inner dots, 2nd inner dots, 2 outer dots, vestibular signals
# Note: By having two outer dots, we support the "bi-directional surround" condition. This further makes
# sense because the outer dots cover a larger area.
sig_obs = sig_obs * np.ones((D,5))
sig_obs[:,:2] *= 1.               # <-- Inner dots
sig_obs[:,2:4] /= 6.                # <-- noise of outer RDK (higher dot density = smaller noise)
sig_obs[:,4] *= 3                 # <-- noise of proprioception
sig_obs = sig_obs.flatten()


C = np.array([
    [1,1,1,0,1,0,0,0],          # <-- 1st inner (leftward) dots
    [1,1,1,0,0,1,0,0],          # <-- 2nd inner (rightward) dots
    [1,1,0,1,0,0,1,0],          # <-- outer dots 1 
    [1,1,0,1,0,0,0,1],          # <-- outer dots 2 
    [1,0,0,0,0,0,0,0],          # <-- proprioception
    ], dtype=np.float64)


@trialNumberAware
def f_v(t, trialNumber):
    vtot =  np.sqrt(D * tau_s * lam_tot**2 / 2)
    if outersign == 0:
        vy_out_1 = +vtot
        vy_out_2 = -vtot
    elif outersign in (-1, 1):
        vy_out_1 = outersign * vtot
        vy_out_2 = outersign * vtot
    else:
        raise Exception(f"outersign {outersign} not in (-1, 0, +1).")
    # Velocities (x/y): inner 1,      inner 2,  outer 1,  outer 2, vestibular 
    v = np.array([[       -vtot,         vtot,       0.,       0.,         0.],     # x-direction
                  [multy * vtot, multy * vtot, vy_out_1, vy_out_2,         0.]])    # y-direction
    return v

human_readable_dsl += f"_inner_y_{multy:d}_outer_" + {1:"up", -1:"down", 0:"bi"}[outersign]


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
        R = reps,                # num repetitions (trials)
        T = 30., # CHANGE BACK TO: 30.                 # total time per trial
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
        seed = 0xDECAFBAD + multy + 10 * outersign,
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
            inf_method_kwargs = dict(fps=fps, tau_s=tau_s, tau_lam=tau_lam, kappa=kappa, nu=np.array([-2./D] + [nu]*7) ),
            )
        },

    }


def plot_perceived_directions(wld, fil, last_tn=60, savefig=False):
    """A quick analysis function"""
    # True velos
    v_true = wld.get_v(wld.S[wld.index_of_t(t)]).reshape(fil.D,fil.K)  # 't' comes from globals()
    # Perceived velos
    s_mean = np.array(fil.archive['mu'][-last_tn:]).mean(0).reshape(fil.D,fil.M)
    s_mean[:,0] = 0.   # disregard self-motion
    v_perc = (fil.C @ s_mean.flatten()).reshape(fil.D, fil.K)
    # Plotting
    import pylab as pl
    fig = pl.figure(figsize=(2.,2.))
    ax = fig.add_axes( (0.17, 0.17, 0.81, 0.81), aspect='equal')
    # plot true velocities
    for k,(vxk,vyk) in enumerate(v_true.T[:-1]):
        kwargs = dict(lw=3.)
        c = pl.cm.Reds((fil.K-k)/ (fil.K+1))
        l = "True velo." if k==1 else None
        ax.plot([0., vxk], [0., vyk], color=c, label=l, **kwargs)
    # plot perceived velocities
    for k,(vxk,vyk) in enumerate(v_perc.T[:-1]):
        kwargs = dict(lw=1.5)
        c = pl.cm.Blues((fil.K-k)/ (fil.K+1))
        l = "Perceived" if k==1 else None
        ax.plot([0., vxk], [0., vyk], color=c, label=l, **kwargs)
    vmax = max(np.max(v_true), np.max(v_perc))
    ax.set_xlim(-1.1*vmax, 1.1*vmax)
    ax.set_ylim(-1.1*vmax, 1.1*vmax)
    ax.set_xlabel("Horizontal velocity, $v_x$", labelpad=2.)
    ax.set_ylabel("Vertical velocity, $v_y$", labelpad=0.)
    leg = ax.legend(loc='best', fontsize=5.)
    if savefig:
        fnamebase = "./fig/fig_sourround_repulsion_Takemura"
        fig.savefig(fnamebase + ".pdf")
        fig.savefig(fnamebase + ".png", dpi=300)
    
    
    
