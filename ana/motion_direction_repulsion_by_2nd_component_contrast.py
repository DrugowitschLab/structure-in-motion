# # # # # # # # # # #
# # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # #
# # # 
# # #   Evaluate the perceived angle of the 1st component in motion direction repulsion 
# # #   experiments like Chen et al. (2005), Figure 7
# # # 
# # #   1) We assume that the motion decomposition includes a self-motion component
# # #      which is not part of the perceived object velocity. Remove this component.
# # #   2) Project the remaining sources back into velocity space.
# # #   3) Average the perceived velocity over the (stationary) end of the trial.
# # #   4) Calculate the perceived angle of the first component.
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

from ChenETAL_direction_repulsion_by_contrast_dsl import DSL
fil = "adiab"                                      # used filter
Alpha = np.array([15., 20., 22.5, 25., 30., 45., 60., 75., 90.]) * (np.pi/180)
NoiseFactor = np.array([0.001, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0])
reps = 20                   # trial repetitions (for each angle)

tavg = 10.                                         # Average percept over the last tavg seconds
mself = 0                                          # Index of the self-motion component 
kprop = 2                                          # Index of the proprioceptive input signal


nA = len(Alpha)
nN = len(NoiseFactor)
assert nN == len(DSL)

# The DSL dict uses strings as keys
NoiseFactorKeys = [f"{nf:.3f}" for nf in NoiseFactor]

def calc_perceived_angle(dsl):
    # # # #
    # # # # # #
    # #   1) Load data
    # # # # # # # # # # #
    # # # # # # # # # # # # # # #
    ds, cfg = si.load_dataset(dsl, F=fil)
    assert cfg['glo']['R'] == nA * reps
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
    # #   3) Calculate the angle
    # # # # # # # # # # #
    # # # # # # # # # # # # # # #
    angle = np.arctan2( V[:,1,:] , V[:,0,:] )
    # Only now we average over the trial repetitions of the same alpha
    mu_perceived_1st_angle = angle[:,0].reshape(nA, reps).mean(1)
    from scipy.stats import sem
    sem_perceived_1st_angle = sem(angle[:,0].reshape(nA, reps), axis=1)

    return mu_perceived_1st_angle, sem_perceived_1st_angle


A_mu = np.zeros((nA, nN))
A_sem = np.zeros((nA, nN))

for i,d in enumerate(NoiseFactorKeys):
    m,s = calc_perceived_angle(DSL[d])
    A_mu[:,i] = m
    A_sem[:,i] = s
    

out = dict(stimulus_opening_angle = Alpha,
           noiseFactor = NoiseFactor,
           reps = reps,
           perceived_1st_angle_mean = A_mu,
           perceived_1st_angle_sem = A_sem,
           )



# # # # # # # # # # #
# # #  S A V E  # # # 
# # # # # # # # # # #

if SAVE:
    fname = "./data/analysis_direction_repulsion_by_2nd_component_contrast.pkl"
    import pickle
    with open(fname, "wb") as f:
        pickle.dump(out, f)
    si.log.info(f"Data saved to file '{fname}'.")

# # # # # # # # # # #
# # #  P L O T  # # # 
# # # # # # # # # # #

if PLOT:
    fig = pl.figure(figsize=(3.25,2.0))
    x = NoiseFactor
    y = (A_mu - Alpha[:,None]/2) / np.pi * 180
    for i,yi in enumerate(y):
        c = pl.cm.Reds((i+3)/(nA+2))
        l = f"Opening angle: {int(Alpha[i]/np.pi*180)}"
        pl.plot(x, yi, c=c, lw=1., marker='o', ms=3, label=l)
    pl.xlim(0., x[-1] * 1.65)
    pl.xticks([0, 1, 5, 10])
    pl.xlabel("1/$\sigma_\mathrm{obs}^2$ of 2nd component (normalized to 1st component)")
    pl.ylabel("Error of perceived angle of 1st component")
    leg = pl.legend(loc='upper right', fontsize=6., labelspacing=0.2)
    pl.subplots_adjust(0.10, 0.16, 0.99, 0.99)
    fig.savefig("./fig/fig_chen_by_2nd_component_contrast.pdf")
    pl.show()

