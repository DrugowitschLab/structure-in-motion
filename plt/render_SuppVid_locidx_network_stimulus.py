# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #
# # #     Render a 2d video of the stimulus in Figure 7k.
# # #     
# # #     
# # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import numpy as np
import pylab as pl
import strinf as si

np.random.seed(9834)

# # #  P A R A M  # # #
bShowOnScreen = False        # Show the rending process on screen? (slower)
SAVE = True                 # Save to video disk?

# # #  Johansson
dsl = "2021-09-03-11-10-56-930384_network_global_sweep_lamG"  # both dots visible

# For all
tStart, tEnd = 0., 70.       # start and end time of the simulation (before speed factor)
# For independent
# tStart, tEnd = 0., 6.       # start and end time of the simulation (before speed factor)
# For intermediate correlation
# tStart, tEnd = 30., 36.       # start and end time of the simulation (before speed factor)
# For highly correlated
# tStart, tEnd = 60., 66.       # start and end time of the simulation (before speed factor)

fps = 60.                   # frames per second (ideally have a multiple of the simulated fps)
rStim = 0.40                 # radius of the stimulus circle (around RFCoord)
nDotPerStim = 25            # num dots per circle
unitfactor = 4.             # one spatial unit to inches

trial = 0                   # trial number
filname = "adiab"           # filter name
speed = 1.0                 # speed multiplier for the video


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
K,M = C.shape
# times
T = ds.t
# Exact velos
V_wld = ds.V_wld.loc[dict(R=trial, F=filname)]
X = (unitfactor*V_wld).cumsum(axis=0) * (T[1] - T[0])
X = X[(tStart <= ds.t) * (ds.t <= tEnd)]
# Some sanity checks
assert 0 <= tStart <= T[-1], "Start time must not be later than simdata."
assert tStart < tEnd <= T[-1], "End time must not be later than simdata."
assert D == 2, "Video is currently limited to 2-dimensional data."

# RF coords
RFCoord = np.array(cfg['wld']['RFCoord'])
# Stim in each axis
dotXY0 = rStim * 2 * (np.random.rand(2, K, nDotPerStim) - 0.5)
Xc = X.data.reshape(X.shape + (1,)) + dotXY0
Xc += rStim
Xc %= 2 * rStim
Xc -= rStim


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


# # #  F I G U R E  # # #
xymin = np.array(pol2cart(RFCoord[:,0], RFCoord[:,1])).min() - rStim - 0.25
xymax = np.array(pol2cart(RFCoord[:,0], RFCoord[:,1])).max() + rStim + 0.25

axlist = dict()
axkwargs = dict(xticks=[], yticks=[], aspect="equal", xlim=(-rStim,rStim), ylim=(-rStim,rStim), frame_on=True)
for k,(r,phi) in enumerate(RFCoord):
    x0,y1 = np.array(pol2cart(r, phi)) - rStim - xymin
    axlist[k] = dict(rect=(x0, y1, 2*rStim, 2*rStim), **axkwargs)

# Fixation cross
axlist['fix'] = dict(rect=(-0.1-xymin,-0.1-xymin,0.2,0.2), xticks=[], yticks=[], aspect="equal", frame_on=False)

fig, axes = si.plottools.init_figure_absolute_units(figsize=(xymax-xymin, xymax-xymin), axes=axlist)

# Fixation cross
kwargs = dict(color='0.3', lw=2.)
axes["fix"].hlines(0, -0.7, 0.7, **kwargs)
axes["fix"].vlines(0, -0.7, 0.7, **kwargs)

# Remove frame of axes (we need frame_on, though)
for k in range(K):
    for sp in axes[k].spines.values():
        sp.set_visible(False)

# MAIN PLOTTING FUNCTION
lines = dict()  #  dict of the lines

def plot_frame(t):
    for k in range(K):
        if (t % 10.) < 0.05:
            axes[k].patch.set_facecolor("0.7")
        else: 
            axes[k].patch.set_facecolor("1.0")
    # time index
    tn = int(np.abs(T-(t-tStart)).argmin())
    # Location
    for k in range(K):
        if k in lines:
            for i in range(nDotPerStim):
                lines[k][i].set_data( Xc[tn,0,k,i], Xc[tn,1,k,i])
        else:
            kwargs = dict(ms=3, mew=0.0, mec='k', mfc='k', marker="o", lw=0., clip_on=True)
            lines[k] = [ axes[k].plot( Xc[tn,0,k,i] ,  Xc[tn,1,k,i] , **kwargs)[0] for i in range(nDotPerStim) ]
        bIsvisible = np.linalg.norm(Xc[tn,:,k], axis=0) < rStim
        for li,vi in zip(lines[k], bIsvisible):
            li.set_visible(vi)
    # we need a flat version of lines
    blitlines = [ l[i] for l in lines.values() for i in range(nDotPerStim) ] + [ axes[k].patch for k in range(K) ]
    return blitlines


# # #  MAIN LOOP  # # #
targettimes = np.arange(tStart, tEnd+0.0001, speed/fps)
times = [ float(T[np.abs(T.values-t).argmin()])  for t in targettimes ]

from matplotlib.animation import FuncAnimation
ani = FuncAnimation(fig, plot_frame, blit=True, interval=1000/fps, frames=times, repeat=False)

if SAVE:
    ani.save(filename="vid/video_S5_locidx_network.mp4", writer=None, fps=fps, dpi=150, codec=None, bitrate=2048, extra_args=['-pix_fmt', 'yuv420p'])  # -pix_fmt yuv420p should improve Mac compatibility

