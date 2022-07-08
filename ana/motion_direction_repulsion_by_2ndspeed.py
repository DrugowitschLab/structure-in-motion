# # # # # # # # # # #
# # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # #
# # # 
# # #   Evaluate the perceived opening angle in motion direction repulsion 
# # #   experiments like Braddick et al. (2002), Figure 3
# # # 
# # #   1) We assume that the motion decomposition includes a self-motion component
# # #      which is not part of the perceived object velocity. Remove this component.
# # #   2) Project the remaining sources back into velocity space.
# # #   3) Average the perceived velocity over the (stationary) end of the trial.
# # #   4) Calculate the perceived angle of the 1st component.
# # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import numpy as np
import pylab as pl
import strinf as si

# # #  PARAMETERS  # # #
PLOT = True                   # Plot some results right away? Make sure that "/ana/fig/" exists
SAVE = True                   # Save results to file?

DSL = "2021-08-19-15-18-47-392303_100_repulsion_by_2ndspeed_Braddick"  # Data to analyze
fil = "adiab"                                      # used filter
Alpha = np.array([15., 30., 45., 60., 75., 90.]) * (np.pi/180)     # opening angles 
v_2nd_factor = np.linspace(0, 2., 21)                  # Speed factor of the 2nd component
reps = 20                                               # trial repetitions (for each angle)

tavg = 10.                                         # Average percept over the last tavg seconds
mself = 0                                          # Index of the self-motion component 
kprop = 2                                          # Index of the proprioceptive input signal


# # # #
# # # # # #
# #   1) Load data
# # # # # # # # # # #
# # # # # # # # # # # # # # #

ds, cfg = si.load_dataset(DSL, F=fil)

nA = len(Alpha)
nV = len(v_2nd_factor)

assert cfg['glo']['R'] == nA * nV * reps

# load C matrix
C = np.array(cfg['fil']['default_params']['C'])

# select time
tidx = ds.t >= (ds.t[-1] - tavg)
# load time and average (since everything is linear, averaging and projecting commutes)
S = ds.S_inf[:,tidx].mean('t')

# # # #
# # # # # #
# #   2) Project to velocities ignoring self-motion
# # # # # # # # # # #
# # # # # # # # # # # # # # #

# Erase self-motion (since projection is linear, this is the same as ignoring its contribution)
S[:,:,mself] = 0.
# Perceived velocities
V = np.delete(S.data @ C.T, kprop, axis=2)   # --> dims: (r, x/y, k)

# # # #
# # # # # #
# #   3) Calculate angle of each component
# # # # # # # # # # #
# # # # # # # # # # # # # # #

angle = np.arctan2( V[:,1,:] , V[:,0,:] )

# Only now we average over the trial repetitions of the same alpha (only for the first component [:,0])
mu_perceived_1st_angle = angle[:,0].reshape(nA, nV, reps).mean(2)
from scipy.stats import sem
sem_perceived_1st_angle = sem(angle[:,0].reshape(nA, nV, reps), axis=2)

out = dict(stimulus_angle = Alpha,
           v_2nd_factor = v_2nd_factor,
           reps = reps,
           perceived_1st_angle_mean = mu_perceived_1st_angle,
           perceived_1st_angle_sem = sem_perceived_1st_angle,
           perceived_angles = angle,  # These are the two individual angles, also before mean and sem
           )



# # # # # # # # # # #
# # #  S A V E  # # # 
# # # # # # # # # # #

if SAVE:
    fname = "./data/analysis_" + DSL + ".pkl"
    import pickle
    with open(fname, "wb") as f:
        pickle.dump(out, f)
    si.log.info(f"Data saved to file '{fname}'.")

# # # # # # # # # # #
# # #  P L O T  # # # 
# # # # # # # # # # #

if PLOT:
    fig = pl.figure(figsize=(3.25,2))
    x = v_2nd_factor
    y = (mu_perceived_1st_angle - (Alpha/2)[:,None]) / np.pi * 180
    for i,(ai,yi) in enumerate(zip(Alpha,y)):
        c = pl.cm.Blues((i+3)/(nA+2))
        l = f"Opening angle: {int(Alpha[i]/np.pi*180)}"
        pl.plot(x, yi, c=c, lw=1., marker='o', ms=3, label=l)
    pl.xlim(x[0], x[-1] * 1.65)
    pl.xticks([0, 1, 2])
    pl.xlabel("Speed of 2nd component (normalized to 1st)")
    pl.ylabel("Bias in perceived angle of 1st component")
    leg = pl.legend(loc='upper right', fontsize=6, labelspacing=0.2)
    pl.subplots_adjust(0.10, 0.16, 0.99, 0.99)
    fig.savefig("./fig/fig_braddick_by_2nd_component_speed.pdf")
    pl.show()

