# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # 
# # # #    Render a structure-from-motion stimulus with nested cylinders
# # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import numpy as np
import pylab as pl
from matplotlib.animation import FuncAnimation
from datetime import datetime

np.random.seed(4321)

bShowOnScreen = False        # Show the rending process on screen? (slower)
SAVE = True                 # Save to video disk?

# # #  STIM PARAMS  # # #
T = 30.
fps = 60.
fnameSuffix = "fast_outer_slow_inner"
bHIGHQ = True               # Bitrate 2048 vs 512
bTIGHT = False                # If true, the video frame will be tight to the stimulus

density = 15               # num dots per 2D area = height x 2 x pi x r
h = 4.0                      # height; max: 8 
r = (1.5, 1.0)                  # radius of the cylinder; max = 4
avgLifetime = 0.250         # average dot life time  in sec (or None for infty)
omega = np.array([1.5, 1.0]) * np.pi/2 / 1.        # rotation speed in rad per sec
sobs = 0.                   # "observation noise"; std in rad (in each frame)
dotColor = "0.25"            # any matplotlib color for the dots


fixation = None # (0., 0.65*h)  # Fixation cross. None or (x,y)

# # #   END OF PARAMS  # # #

r = np.array(r)
N = (np.round(density * h * 2 * np.pi * r)).astype(int)  # list of dot numbers (per radius)

if not bShowOnScreen:
    pl.matplotlib.use("Agg")

# Initial y_i and phi_i
Y = -h/2 + h * np.random.rand(N.sum())
Phi = 2 * np.pi * np.random.rand(N.sum())

# Helper functions
R_unrolled = np.concatenate([ ri * np.ones(ni) for ri,ni in zip(r,N) ])
Omega_unrolled = np.concatenate([ omi * np.ones(ni) for omi,ni in zip(omega,N) ])

Phi2X = lambda Phi: R_unrolled * np.cos(Phi)


framecount = []

def update(i):
    global Y, Phi
    framecount.append(i)
    print("\b"*100 + f"Progress: {i/fps:.3f}s", end="", flush=True)
    Phi += Omega_unrolled / fps
    Phi %= 2 * np.pi
    # Kill and regenerate dots
    if avgLifetime is not None:
        # pnew = 1. / fps / avgLifetime   # Approximation if dt << avgLifetime
        pnew = 1. - np.exp(-1/(fps*avgLifetime))
        idx = np.random.rand(N.sum()) < pnew
        Y[idx] = -h/2 + h * np.random.rand(N.sum())[idx]
        Phi[idx] = 2 * np.pi * np.random.rand(N.sum())[idx]
    # draw the dots
    x,y = Phi2X(Phi + np.random.normal(0, sobs, N.sum())), Y
    ln.set_data(x, y)
    return ln,

t0 = None


fig = pl.figure(figsize=(7,7))
ax = fig.add_axes((0,0,1,1), aspect="equal", frame_on=False, xticks=[], yticks=[])
x,y = Phi2X(Phi), Y
ln, = ax.plot(x, y, 'o', ms=1., mfc=dotColor, mec=dotColor)
if fixation is not None:
    ax.text(fixation[0], fixation[1], "+", color="k", size=12, ha='center')
    
if bTIGHT:
    xymax = max( max(np.abs(ax.xaxis.get_data_interval())), max(np.abs(ax.yaxis.get_data_interval())) )
    xymax *= 1.1
    fig.set_size_inches(3.5, 3.5)
    ax.set_xlim(-xymax, +xymax)
    ax.set_ylim(-xymax, +xymax)
    fnameSuffix += "_tight"
else:
    ax.set_xlim(-5, +5)
    ax.set_ylim(-5, 5)

# run animation
ani = FuncAnimation(fig, update, blit=True, interval=1000/fps, frames=int(round(T*fps)), repeat=False)
if SAVE:
    ani.save(filename=f"vid/video_S4_Structure-from-motion_nested_{fnameSuffix}.mp4", writer=None, fps=fps, dpi=150, codec=None, bitrate=2048 if bHIGHQ else 512, extra_args=['-pix_fmt', 'yuv420p'])

print("\n")

