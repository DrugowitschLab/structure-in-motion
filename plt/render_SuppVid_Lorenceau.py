import numpy as np
import pylab as pl
from matplotlib.animation import FuncAnimation
from datetime import datetime

np.random.seed(12345)

bShowOnScreen = False        # Show the rending process on screen? (slower)
SAVE = True                 # Save to video disk?

N = 20      # num dots

a = 1.0     # amplitude
frac = 0.50  # fraction of dots in vertical group
omega = 2*np.pi* 0.83
fps = 60.
T = 10.

numNoise = 4

na1 = 0.15  # 0.5
nomega = 2 * np.pi * omega
phi = 2 * np.pi * np.random.rand(N)

if not bShowOnScreen:
    pl.matplotlib.use("Agg")

x0,y0 = np.meshgrid(np.linspace(-4.0,4.0,5), np.linspace(-3.0,3.0,4))
x0 = x0.flatten()
y0 = y0.flatten()
# x0 = np.random.rand(N)
# y0 = np.random.rand(N)

idx = np.random.choice(np.arange(N), int(round(frac*N)), replace=False)
G = idx, np.setdiff1d(np.arange(N), idx)
na = 0.0

def calc_xy(i):
    t = i / fps
    dx = a * np.sin(omega * t) * np.ones(N)
    dx[G[0]] = 0
    dy = a * np.cos(omega * t) * np.ones(N)
    dy[G[1]] = 0.
    dnx = na * np.sin(nomega * t + phi) * np.ones(N)
    dnx[G[1]] = 0
    dny = na * np.sin(nomega * t + phi) * np.ones(N)
    dny[G[0]] = 0
    return x0+dx+dnx, y0+dy+dny

def update(i):
    # i += 1
    global t0
    if t0 is None:
        t0 = datetime.now()
    t = (datetime.now() - t0)
    print("\b"*100 + f"Progress: {i/fps:.3f}s", end="", flush=True)
    x,y = calc_xy(i)
    ln.set_data(x, y)
    return ln,

t0 = None

fig = pl.figure(figsize=(4,3))
ax = fig.add_axes((0,0,1,1), aspect="equal", frame_on=False, xticks=[], yticks=[])
x,y = calc_xy(0)
ln, = ax.plot(x, y, 'ko')
ax.set_xlim(-2.0*a + x0.min(), x0.max() + 2.0*a)
ax.set_ylim(-2.0*a + y0.min(), y0.max() + 2.0*a)

for noise_i in range(numNoise):
    # next noise
    na = noise_i * na1
    # reassign groups
    idx = np.random.choice(np.arange(N), int(round(frac*N)), replace=False)
    G = idx, np.setdiff1d(np.arange(N), idx)
    # rerun animation
    ani = FuncAnimation(fig, update, blit=True, interval=1000/fps, frames=int(round(T*fps)), repeat=False)
    if SAVE:
        ani.save(filename=f"vid/video_S3_Lorenceau_noise_{noise_i}.mp4", writer=None, fps=fps, dpi=150, codec=None, bitrate=2048, extra_args=['-pix_fmt', 'yuv420p'])


