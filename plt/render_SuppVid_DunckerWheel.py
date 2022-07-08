# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #
# # #     Render a 2d video of the input locations; velocities
# # #     and of the filter's estimated velocities; and of the
# # #     inferred motion strengths.
# # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import numpy as np
import pylab as pl
import strinf as si

np.random.seed(9834)

# # #  P A R A M  # # #
bShowOnScreen = False        # Show the rending process on screen? (slower)
SAVE = True                 # Save to video disk?

# # #  DUNCKER WHEEL
dsl = "2021-08-18-11-25-56-034025_020_DunckerWheel"  # both dots visible
x0 = np.array([(0.,0.), (1.,0.)])        # Assumed starting locations of shape (2,K)
tStart, tEnd = 0., 8.       # start and end time of the simulation (before speed factor)
fps = 60.                   # frames per second (ideally have a multiple of the simulated fps)

trial = 0                   # trial number
filname = "adiab"           # filter name
speed = 1.0                 # speed multiplier for the video
traceframes = 20            # length of the trace in S
bHideLinearDot = False       # Just for demo: show only the bouncing dot
bHideInference = True       # Just for demo: hide the axes with the inference part

# Overwrite the default colors
# si.plotting.groupColor = ( si.colors.CMAP['glo'], si.plotting.truncate_colormap(si.colors.CMAP['ind'], 0.9, 0.4) )
sourcecolors = np.vstack([si.colors.get_color(si.colors.CMAP['glo'], 0 ,1), si.colors.get_colors(si.colors.CMAP['ind'], 2)])
dotcolors = si.colors.get_colors(si.colors.CMAP['velo'], 3)[:2]


assert not(bHideLinearDot==True and bHideInference==False), "This setting makes no sense given the data file."

if not bShowOnScreen:
    pl.matplotlib.use("Agg")

# # #  LOAD DATA  # # #
import xarray as xr
from os import path
ds = xr.open_dataset(path.join("./data", dsl, "simdata.nc"))
# general config
import json
cfg = json.loads(ds._metadict_str)
C = np.array(cfg["fil"]["default_params"]["C"])
D = np.array(cfg["fil"]["default_params"]["D"])
# C = np.array(cfg["wld"]["C"])
# D = np.array(cfg["wld"]["D"])
K,M = C.shape
# times
T = ds.t
# Inferred motion strengths
L = ds.Lam_inf.loc[dict(R=trial, F=filname)]
# Noisy velos
V_obs = ds.V_obs.loc[dict(R=trial, F=filname)]
# Exact velos
S_wld = ds.S_wld.loc[dict(R=trial, F=filname)]
V_wld = ds.V_wld.loc[dict(R=trial, F=filname)]
# Inferred velocities
S_inf = ds.S_inf.loc[dict(R=trial, F=filname)]
V_inf = S_inf.values @ C.T
# Exact trajectories
if ("Mrot" in cfg["wld"]) and (cfg["wld"]["Mrot"] is not None):
    x0 = cfg["wld"]["Mrot"] @ x0
X = V_wld.cumsum(axis=0) * (T[1] - T[0]) + x0
X = X[(tStart <= ds.t) * (ds.t <= tEnd)]
# Some sanity checks
assert 0 <= tStart <= T[-1], "Start time must not be later than simdata."
assert tStart < tEnd <= T[-1], "End time must not be later than simdata."
assert D == 2, "Video is currently limited to 2-dimensional data."
# derive some global values
xxmax = np.abs(X[:,0]).max()
xymax = np.abs(X[:,1]).max()
# vxmax = max(np.abs(V_wld[:,0]).max(), np.abs(V_inf[:,0]).max())
# vymax = max(np.abs(V_wld[:,1]).max(), np.abs(V_inf[:,1]).max())
smax = np.abs(S_inf).max()
lmax = np.max(L)


def make_compact_ticks(ax):
    ax.set_xticks([ax.get_xticks()[0], ax.get_xticks()[-1]])
    ax.set_xlabel( ax.get_xlabel(), labelpad= -3  )
    ax.set_yticks([ax.get_yticks()[0], ax.get_yticks()[-1]])
    ax.set_ylabel( ax.get_ylabel(), labelpad= -2 * len(str(ax.get_yticks()[0]))  )


# # #  F I G U R E  # # #
fig = pl.figure(figsize=(7,4))
ar = fig.get_figwidth()/fig.get_figheight()
l,b,w,h = 0.01, 0.65, 0.98, 0.30
rect = l,b,w,h
ax_X = fig.add_axes(rect, aspect="equal", xticks=[], yticks=[], frame_on=False)
rect = 0.7, 0.10, 0.25, 0.25*ar
ax_S = fig.add_axes(rect, aspect="equal")
l = 0.08
rect = l, 0.10, 0.45, 0.25*ar
ax_L = fig.add_axes(rect)
wc, hc = 0.07 * M / K * fig.get_figheight()/fig.get_figwidth() , 0.07
rect = l+0.025, 0.10+0.24*ar-hc, wc, hc
ax_C = fig.add_axes(rect, aspect="equal", frame_on=True, xticks=[], yticks=[], xlim=(-0.5,M-0.5), ylim=(K-0.5,-0.5), zorder=10)

# Decorations
bolddict = dict(weight="bold")
facLim = 1.1
# ax_X
# ax_X.set_title("Location", pad=3, fontdict=bolddict)
# ax_X.set_xlabel("x")
# ax_X.set_ylabel("y")
ax_X.set_xlim(-1., xxmax + 1)
ax_X.set_ylim(-facLim*2*xymax, facLim*2*xymax)
# make_compact_ticks(ax_X)
# ax_S
ax_S.set_title("Latent motion sources", pad=3, fontdict=bolddict)
ax_S.set_xlabel("$s_{m,x}$")
ax_S.set_ylabel("$s_{m,y}$")
ax_S.set_xlim(-1.01 * smax, 1.01 * smax)
ax_S.set_ylim(-1.01 * smax, 1.01 * smax)
# make_compact_ticks(ax_S)
# ax_L
ax_L.set_title("Motion strengths", pad=3, fontdict=bolddict)
ax_L.set_xlabel("Time [s]")
ax_L.set_ylabel(r"Motion strength $\lambda_m$")
ax_L.set_xlim(tStart, tEnd)
ax_L.set_ylim(0., facLim*lmax)
# ax_C
si.imshow_C_matrix(C, ax_C, colors=sourcecolors, addVisibleCircles=True, viscolors=dotcolors)


# MAIN PLOTTING FUNCTION
lines = dict()  #  dict of the lines

def plot_frame(t):
    from strinf.plotting import assign_colors
    # time indices
    t0 = int(np.abs(T-tStart).argmin())
    tn = int(np.abs(T-t).argmin())
    # Location
    if "X" in lines:
        for k in range(K):
            lines["X"][k].set_data( X[tn,0,k], X[tn,1,k])
    else:
        lines["X"] = []
        colors = dotcolors
        for k in range(K):
            kwargs = dict(ms=7, mew=0.0, mec='k', mfc=colors[k], marker="o", lw=0., clip_on=False)
            lines["X"].append( ax_X.plot( X[tn,0,k] ,  X[tn,1,k] , **kwargs)[0] )
        if bHideLinearDot:
            lines["X"][1].set_visible(False)
        if bHideInference:
            ax_S.set_visible(False)
            ax_L.set_visible(False)
            ax_C.set_visible(False)
    if bHideInference:
        return lines['X']
    # Motion strengths
    if "L" in lines:
        for m in range(M):
            lines["L"][m].set_data(T[t0:tn+1], L[t0:tn+1,m])
    else:
        colors = sourcecolors
        ax_L.set_prop_cycle(c=colors)
        lines["L"] = ax_L.plot(T[t0:tn+1], L[t0:tn+1], lw=0.7)
    # Sources
    if "S" in lines:
        for m in range(M):
            lines["S"]["dots"][m].set_data( S_inf[tn,0,m], S_inf[tn,1,m] )
            tn0 = max(0, tn-traceframes)
            lines["S"]["trace"][m].set_data( S_inf[tn0:tn+1,0,m], S_inf[tn0:tn+1,1,m] )
    else:
        lines["S"] = dict()
        colors = sourcecolors
        lines["S"]["dots"] = []
        lines["S"]["trace"] = []
        for m in range(M):
            kwargs = dict(ms=5, mec=colors[m], mfc=colors[m], marker="o", lw=0., color="w", zorder=5)
            lines["S"]["dots"].append( ax_S.plot( S_inf[tn,0,m], S_inf[tn,1,m], **kwargs)[0] )
            tn0 = max(0, tn-traceframes)
            kwargs = dict(lw=0.7, color=colors[m], zorder=3)
            lines["S"]["trace"].append( ax_S.plot( S_inf[tn0:tn+1,0,m], S_inf[tn0:tn+1,1,m], **kwargs)[0] )
    # we need a flat version of lines
    blitlines = lines['L'] + lines['X'] + lines['S']['dots'] + lines['S']['trace']
    return blitlines


# # #  MAIN LOOP  # # #
targettimes = np.arange(tStart, tEnd+0.0001, speed/fps)
times = [ float(T[np.abs(T.values-t).argmin()])  for t in targettimes ]

from matplotlib.animation import FuncAnimation
ani = FuncAnimation(fig, plot_frame, blit=True, interval=1000/fps, frames=times, repeat=False)

if SAVE:
    fname = "vid/video_S2_DunckerWheel"
    if bHideInference:
        fname += "_stimulusOnly"
    if bHideLinearDot:
        fname += "_rimDotOnly"
    fname += ".mp4"
    ani.save(filename=fname, writer=None, fps=fps, dpi=150, codec=None, bitrate=2048, extra_args=['-pix_fmt', 'yuv420p'])  # -pix_fmt yuv420p should improve Mac compatibility

