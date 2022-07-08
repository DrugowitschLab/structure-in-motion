# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # 
# # # #    Render a structure-from-motion stimulus
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
fnameSuffix = "one_cylinder"
bHIGHQ = True               # Bitrate 2048 vs 512
bTIGHT = False                # If true, the video frame will be tight to the stimulus

density = 15               # num dots per 2D area = height x 2 x pi x r
h = 1 * 4.0                      # height; max: 8
r = 1.5                  # radius of the cylinder; max = 4
avgLifetime = 0.250         # average dot life time  in sec (or None for infty)
omega = np.pi/2 / 1.        # rotation speed in rad per sec
sobs = 0.                   # "observation noise"; std in rad (in each frame)
dotColor = "0.25"            # any matplotlib color for the dots

translation = None #  (15., 24/fps, 2.0)  # horizontal translation. None or (v_move, t_move, t_rest)
fixation = None # (0., 0.65*h)  # Fixation cross. None or (x,y)

# # #   END OF PARAMS  # # #

N = int(round(density * h * 2 * np.pi * r))

if not bShowOnScreen:
    pl.matplotlib.use("Agg")

# Initial y_i and phi_i (Def.: 0 <= y_i < 1; 
Y = -h/2 + h * np.random.rand(N)
Phi = 2 * np.pi * np.random.rand(N)

# Helper functions
Phi2X = lambda Phi: r * np.cos(Phi)
x0 = 0.
_initphase = True    # Just a correction since mpl calls update 3x in the beginning with i=0

framecount = []

def update(i):
    global Y, Phi, x0, _initphase
    framecount.append(i)
    print("\b"*100 + f"Progress: {i/fps:.3f}s", end="", flush=True)
    Phi += omega / fps
    Phi %= 2 * np.pi
    if translation is not None:
        v_move, t_move, t_rest = translation
        t_full = 2 * (t_move + t_rest)
        t = ( ( i + fps*t_move/2 ) % (fps*t_full) ) / fps    # in sec mod full oscillation, right starts at 0
        if t < t_move:
            v0 = v_move
        elif t < t_move + t_rest:
            v0 = 0.
        elif t < 2 * t_move + t_rest:
            v0 = -v_move
        else:
            v0 = 0.
        if i > 0:
            _initphase = True
        x0 += _initphase * v0 / fps
        # print(f" {_initphase * v0} {x0:.3f}")
        if i == 0 and _initphase:
            _initphase = False
    # Kill and regenerate dots
    if avgLifetime is not None:
        # pnew = 1. / fps / avgLifetime   # Approximation if dt << avgLifetime
        pnew = 1. - np.exp(-1/(fps*avgLifetime))
        idx = np.random.rand(N) < pnew
        Y[idx] = -h/2 + h * np.random.rand(N)[idx]
        Phi[idx] = 2 * np.pi * np.random.rand(N)[idx]
    # draw the dots
    x,y = x0 + Phi2X(Phi + np.random.normal(0, sobs)), Y
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
    ani.save(filename=f"vid/video_S4_Structure-from-motion_height_{h:.1f}_{fnameSuffix}.mp4", writer=None, fps=fps, dpi=150, codec=None, bitrate=2048 if bHIGHQ else 512, extra_args=['-pix_fmt', 'yuv420p'])

print("\n")

