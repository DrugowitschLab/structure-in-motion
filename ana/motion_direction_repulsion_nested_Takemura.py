# # # # # # # # # # #
# # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # #
# # # 
# # #   Evaluate the perceived directions of the inner dots
# # #   in experiments 1 and 2 of Takemura et al., 2011. Cf. Figures 4 and 6.
# # # 
# # #   1) We assume that the motion decomposition includes a self-motion component
# # #      which is not part of the perceived object velocity. Remove this component.
# # #   2) Project the remaining sources back into velocity space.
# # #   3) Collect the perceived angles over different (noisy) repetitions.
# # #   
# # #   Inner dots' motion can be horizontal (multy=0) or with 90 deg opening angle (multy=1).
# # #   The surround motion can be up (+1, +1), down (-1, -1), or bidirectional (+1, -1).
# # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import numpy as np
import pylab as pl
import strinf as si

pl.rc('figure', dpi=2*pl.rcParams['figure.dpi'])


# # #  PARAMETERS  # # #
PLOT = True                   # Plot some results right away? Make sure that "/ana/fig/" exists
SAVE = True                   # Save results to file?

DSL = {"multy_0_surround_bi" : "2022-05-12-16-03-09-435421_107_direction_repulsion_Takemura_inner_y_0_outer_bi",
       "multy_0_surround_down" : "2022-05-12-16-03-25-555141_107_direction_repulsion_Takemura_inner_y_0_outer_down",
       "multy_1_surround_down" : "2022-05-12-16-03-46-053483_107_direction_repulsion_Takemura_inner_y_1_outer_down",
       "multy_1_surround_bi" : "2022-05-12-16-45-40-330033_107_direction_repulsion_Takemura_inner_y_1_outer_bi",
       "multy_1_surround_up" : "2022-05-12-16-46-03-495979_107_direction_repulsion_Takemura_inner_y_1_outer_up",
       }

fil = "adiab"                                      # used filter
reps = 200                                         # trial repetitions

tavg = 0.                                         # Average percept over the last tavg seconds
mself = 0                                          # Index of the self-motion component 
kinner = 0,1                                       # Indices of the inner dots (the ones of interest)

def calculate_inner_angles(DSL):
    # # # #
    # # # # # #
    # #   1) Load data
    # # # # # # # # # # #
    # # # # # # # # # # # # # #
    ds, cfg = si.load_dataset(DSL, F=fil)
    assert cfg['glo']['R'] == reps
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
    V = S.data @ C.T              # --> dims: (r, x/y, k)
    # # # #
    # # # # # #
    # #   3) Collect the perceptions
    # # # # # # # # # # #
    # # # # # # # # # # # # # # #
    angle = np.arctan2( V[:,1,:] , V[:,0,:] )
    inner_angles = np.zeros((len(kinner),reps))
    for ki, k in enumerate(kinner):
        inner_angles[ki] = angle[:,k]
    return inner_angles

# # # # # # # # # # #
# # #  P L O T  # # # 
# # # # # # # # # # #

def plot_inner_angles(inner_angles, bins=np.linspace(-22.5,202.5,46)/180*np.pi, key=None): 
    fig = pl.figure(figsize=(3,2))
    ax = fig.add_axes((0.10, 0.05, 0.85, 0.85), polar=True)
    ax.set_thetamin(-22.5)   # Interestingly, in degree
    ax.set_thetamax(180+22.5)
    theta = ( bins[1:] + bins[:-1] ) / 2
    for angle in inner_angles:
        # Bring angles into the range [-1/2 pi, +3/2 pi]
        angle += np.pi/2
        angle %= 2 * np.pi
        angle -= np.pi/2
        # count
        count,_ = np.histogram(angle, bins=bins)
        print(np.sum(count), count)
        kwargs = dict(bottom=0., width=np.diff(bins), zorder=5)
        ax.bar(theta, count, **kwargs)
    ax.set_xticks(np.linspace(0, 180, 7)/180*pi)
    ax.set_ylim(0, 55)
    ax.set_yticks([50])
    if key:
        ax.set_title(key)
    plot_setup_sketch(key, fig)
    pl.show()


def plot_setup_sketch(key, fig=None):
    if fig is None:
        fig = pl.figure(figsize=(3,2))
    ax = fig.add_axes((0.01,0.59,0.25,0.40), aspect="equal", xticks=[], yticks=[])
    # Circles
    kwargs = dict(ec='0.5', lw=0.5)
    ax.add_patch( pl.Circle(xy=(0.,0.), radius=0.95, fc='0.90', zorder=0., **kwargs) )
    ax.add_patch( pl.Circle(xy=(0.,0.), radius=0.55, fc='1.00', zorder=1., **kwargs) )
    # Inner arrows
    kwargs = dict(x=0., y=0., lw=0., width=0.08, length_includes_head=True, head_width=0.2, head_length=0.2)
    import strinf as si
    Arrow = pl.matplotlib.patches.FancyArrow
    for k in kinner:
        c = si.colors.get_color(si.colors.CMAP['velo'], k, 4)
        dy = 0.35 if "multy_1" in key else 0.
        dx = 0.35 if k==1 else -0.35
        ax.add_patch( Arrow(dx=dx, dy=dy, fc=c, ec=c, **kwargs) )
    # Outer arrows
    c = si.colors.get_color(si.colors.CMAP['velo'], 3, 4)
    kwargs = dict(dx=0., lw=0., width=0.08, fc=c, ec=c, length_includes_head=True, head_width=0.2, head_length=0.2)
    for x in (-0.72, 0.72):
        if "up" in key:
            ax.add_patch( Arrow(x=x, y=-0.30, dy=0.60, **kwargs) )
        elif "down" in key:
            ax.add_patch( Arrow(x=x, y=0.30, dy=-0.60, **kwargs) )
        elif "bi" in key:
            ax.add_patch( Arrow(x=x, y=0., dy=+0.40, **kwargs) )
            ax.add_patch( Arrow(x=x, y=0., dy=-0.40, **kwargs) )
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_frame_on(False)
    pl.show()
    


# # # # # # # # # # #
# # #  M A I N  # # # 
# # # # # # # # # # #


# Output dict
perceived_angles = dict()      # unit: radians

for key in DSL:
    perceived_angles[key] = calculate_inner_angles(DSL[key])
    if PLOT:
         plot_inner_angles(perceived_angles[key], key=key)



# # # # # # # # # # #
# # #  S A V E  # # # 
# # # # # # # # # # #

if SAVE:
    fname = "./data/analysis_direction_repulsion_Takemura.pkl"
    import pickle
    with open(fname, "wb") as f:
        pickle.dump(perceived_angles, f)
    si.log.info(f"Data saved to file '{fname}'.")


