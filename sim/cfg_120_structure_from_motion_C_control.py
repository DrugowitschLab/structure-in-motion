"""Structure-from-motion (SfM). Control experiment with additional motion components (= SI Figure)"""

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
human_readable_dsl = "120_SFM_C_control" # <-- With more components, will the algo choose the rotation?

D = 2                   # number of spatial dimensions
K0 = 18                 # Observables for a cylinder with r > rfxmax; K will be K_outer + K_inner + 1 (vestibular input)
rfxmax = 1.6            # determines RF range

# # #  SELECT EXPERIMENT HERE  # # #
# Experiment 1: Only the outer cylinder
Rs = (1.5,)          # radii
Omega = np.array([1.0]) * np.pi/2. / 1.   # rotational speed
noise_factor = 20. # 30.


# # # We have to calculate the RFs this early based on geometry to determine K  # # #
# rfxloc = np.linspace(Rs[0], -Rs[0], K0//2 + 1)
# WARNING: rf's should NOT fall on the left or right edge of a cylinder.
rfxloc = np.linspace(rfxmax, -rfxmax, K0//2)  


# location for k=0..(K-2); phi=0 <==> (x,z) = (r,0); the last one is a dummy for vestibular
phi, Ks = [], []
for ri in Rs: # For every cylinder:
    # Select the RF locs within the cylinder
    rfxloc_i = np.array( [ x for x in rfxloc if -ri < x <= ri ] )
    # Calculate their polar angles (and add a second set of fields at +pi, one for the front and one for the back)
    phi += [  np.arccos(rfxloc_i / ri), np.arccos(rfxloc_i / ri) + np.pi ]
    Ks.append(2 * len(rfxloc_i))       # num fields per cylinder
phi += [ [0.] ]  # dummy for vestibular input
phi = np.concatenate(phi)

K = len(phi)

# Init the phi's of the polar coords; we'll set the radii next
RFCoord = np.array([ (0., phik) for phik in phi ])    # vestibular can have any value since C[-1,1] = 0
# Now we set the radii:
R_unrolled = np.ones(K)   # This data format will come in handy for stimulus generation, too.
Omega_unrolled = np.ones(K)
for (om,ri,idxmin,idxmax) in zip(Omega, Rs, np.cumsum([0]+Ks[:-1]), np.cumsum(Ks) ):
    R_unrolled[idxmin:idxmax] = ri
    Omega_unrolled[idxmin:idxmax] = om
# Set the radii
RFCoord[:,0] = R_unrolled
RFCoord = RFCoord.tolist()  # For backward compatibility

# Observation noise
sig_obs = sig_obs * np.ones((D,K))
sig_obs[:,-1] *= 3.            # vestibular (defaut: *= 3)
sig_obs[:,:-1] *= noise_factor # dots
sig_obs = sig_obs.flatten()


# Build C matrix: self, global_rot, global_trans, front trans, back trans, individual
assert len(Rs) == 1  # This is only the control
# Cartoon:
# 1 1 1 1   1
# 1 1 1 1     1
# 1 1 1 1       1
# 1 1 1   1       1
# 1 1 1   1         1
# 1 1 1   1           1
# 1
C = np.hstack(( np.ones((K,5)), np.vstack((np.eye(K-1), np.zeros(K-1))) ))
C[Ks[0]//2:,3] = 0.
C[:Ks[0]//2,4] = 0.
C[-1,1:] = 0.   # vestibular row
polarIdx = [1] # global cylinder rotation is the only polar sources


# rotation around upward-directing y-axis (x points to the right; z towards the observer)
# observed are only the (x,y) coordinates, no depth perception.
def f_v(t):
    # main component ( v(t) = dx(t) / dt )
    vx = - Omega_unrolled * R_unrolled * np.sin(phi)
    vy = np.zeros(K)
    vz = - Omega_unrolled * R_unrolled * np.cos(phi)
    # combine (vz is not observed)
    v = np.concatenate([vx, vy]).reshape(D,K)
    v[:,-1] = 0.      # vestibular
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
        T = 100.,                  # total time per trial
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
            polarIdx = polarIdx,
            RFCoord = RFCoord,
            bSFM = True,
            inf_method_kwargs = dict(fps=fps, tau_s=tau_s, tau_lam=tau_lam, kappa=kappa, nu=np.array([-2./D] +[nu]*4 + [nu]*(K-1) )),
            )
        },
    }

