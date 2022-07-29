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
# # #   4) Calculate the perceived opening angle.
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

DSL = "2021-08-19-12-14-27-196363_100_repulsion_by_angle_Braddick"  # Data to analyze
fil = "adiab"                                      # used filter
nAngle = 33                 # number of tested opening angles (spaced equally from 0..180 degrees)
reps = 20                   # trial repetitions (for each angle)

tavg = 10.                                         # Average percept over the last tavg seconds
mself = 0                                          # Index of the self-motion component 
kprop = 2                                          # Index of the proprioceptive input signal


# # # #
# # # # # #
# #   1) Load data
# # # # # # # # # # #
# # # # # # # # # # # # # # #

ds, cfg = si.load_dataset(DSL, F=fil)

assert cfg['glo']['R'] == nAngle * reps
alpha = np.linspace(0., 180 * (np.pi/180), nAngle)             # in radians

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
# #   3) Calculate opening angle
# # # # # # # # # # #
# # # # # # # # # # # # # # #

angle = np.arctan2( V[:,1,:] , V[:,0,:] )
openingAngle = angle[:,0] - angle[:,1]

# Only now we average over the trial repetitions of the same alpha
mu_opening_angle = openingAngle.reshape(nAngle, reps).mean(1)
from scipy.stats import sem
sem_opening_angle = sem(openingAngle.reshape(nAngle, reps), axis=1)

out = dict(stimulus_angle = alpha,
           reps = reps,
           perceived_opening_angle_mean = mu_opening_angle,
           perceived_opening_angle_sem = sem_opening_angle,
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
    fig = pl.figure(figsize=(3,2))
    x = alpha / np.pi * 180
    y = (mu_opening_angle - alpha) / np.pi * 180
    pl.errorbar(x, y, yerr=sem_opening_angle / np.pi * 180, color='b', lw=1., label="Model", zorder=0)
    try:
        B = np.loadtxt('./data/data_Braddick_2002_Fig3C.txt')
        pl.errorbar(B[:,0], B[:,1], yerr=B[:,2], fmt='o', color='k', ms=4, capsize=1.5, label="Braddick et al. (2002)" )
        leg = pl.legend(loc='lower right')
    except:
        si.log.warning("Cannot load experimental data.")
    pl.xlim(x[0], x[-1])
    pl.xlabel("Presented opening angle")
    pl.ylabel("Bias in perceived angle")
    pl.subplots_adjust(0.12, 0.14, 0.97, 0.98)
    fig.savefig("./fig/fig_braddick_by_angle.pdf")
    pl.show()

