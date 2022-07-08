import numpy as np
import pylab as pl

import strinf as si

# # #  PARAMETERS  # # #
PLOT = True                   # Plot some results right away? Make sure that "/ana/fig/" exists
SAVE = True                   # Save results to file?

datadir = "./data/"
DSL = "2022-06-17-11-03-33-098346_121_SFM_one_cylinder_switching_distribution"  # Data to analyze
fil = "adiab"                                      # used filter

tmax = None         # None or float (for development)

midx = 1,1       # index of the rotational source (format: D,M)
threshrel = 1.00    # threshold for detecting a switch; unit: fraction of (autodetected) mode


# # # #
# # # # # #
# #   1) Load data
# # # # # # # # # # #
# # # # # # # # # # # # # # #

ds, cfg = si.load_dataset(DSL, datadir=datadir, F=fil, R=0)
# raise Exception
S = ds.S_inf[:,midx[0],midx[1]]
T = S.t

if tmax is not None:
    print(f" > WARNING: Using only t <= {tmax:.2f}s!")
    # select time
    tidx = T <= tmax
    S = S[tidx]
    T = T[tidx]


# # # #
# # # # # #
# #   2) Find the mode and binarize the sources based on threshold
# # # # # # # # # # #
# # # # # # # # # # # # # # #

smax = np.abs(S).max().data
hist, bin_edges = np.histogram(np.abs(S), bins=100, range=(0., smax))
smode = bin_edges[hist.argmax():hist.argmax()+2].mean()
print(f" > Mode detected at ±{smode:.2f}")

thresh = threshrel * smode
print(f" > Threshold set to ±{thresh:.2f}")

# Binarize S: +1 = above thresh, -1 = below negative thresh, 0 = neural territory 
B = np.zeros( len(S), dtype=int)
B[S >  thresh] = +1
B[S < -thresh] = -1

# # # #
# # # # # #
# #   3) Find switching times (-1 --> +1, or +1 --> -1)
# # # # # # # # # # #
# # # # # # # # # # # # # # #

tswitch = []
percept = []
current_percept = None
print(" > Detecting perceptual switches.")
for b,t in zip(B,T):
    # Neutral zone? --> ignore
    if b == 0:
        continue
    # First percept?
    if current_percept is None:
        current_percept = b
        tswitch.append(t)
        percept.append(b)
        continue
    # Here comes the typical test:
    # New percept?
    if b != current_percept:
        current_percept = b
        tswitch.append(t)
        percept.append(b)
        # print(f"{t.data:.2f}s", end=" ")

tswitch = np.array(tswitch)
print(f" > Found {len(tswitch)} switches.")

duration = np.diff(tswitch)

# # # #
# # # # # #
# #   4) Plot some information
# # # # # # # # # # #
# # # # # # # # # # # # # # #

pl.rc("figure", dpi=2*pl.rcParams["figure.dpi"])
if PLOT:
    fig = pl.figure(figsize=(7,1.50))

    rect = 0.05, 0.17, 0.65, 0.75 
    ax_s = fig.add_axes(rect)
    rect = 0.78, 0.17, 0.20, 0.75 
    ax_t = fig.add_axes(rect)

    plot_tmin, plot_tmax = 50., 150.
    tidx = (plot_tmin <= T) * (T <= plot_tmax)  # Usage: T[tidx], S[tidx]

    # # #  Sources  # # #
    ax = ax_s
    # Source
    ax.plot(T[tidx], S[tidx], c='royalblue', lw=0.75, zorder=5)
    # Threshold
    ax.hlines((-thresh,thresh), plot_tmin, plot_tmax, colors="0.4", linestyles="--", lw=0.5, zorder=10)
    ax.hlines((0.), plot_tmin, plot_tmax, colors="0.7", linestyles="--", lw=0.5, zorder=1)
    # Percept colors
    for t,w,b in zip(tswitch[:-1], duration, percept):
        if t + w < plot_tmin:
            continue
        if t > plot_tmax:
            break
        fc = "salmon" if b == +1 else "aquamarine"
        patch = pl.Rectangle((t,-2.5*smode), w, 5*smode, fc=fc, lw=0., zorder=0.)
        ax.add_patch(patch)
    # Beautify
    ax.set_xlim(plot_tmin, plot_tmax)
    ax.set_ylim(-1.75*smode, 1.75*smode)
    ax.set_xlabel("Time [s]", labelpad=1)
    ax.set_ylabel("Rotational motion source, s(t)")

    # # #  Histogram  # # #
    ax = ax_t
    tmean = duration.mean()
    w = tmean / 20.
    bins = np.concatenate(([0.], np.arange(w/2, 3.0*tmean, w)))
    # bins = np.arange(0., 3*tmean, w)
    kwargs = dict(color="royalblue")
    n,bins,patches = ax.hist(duration, bins, density=True, label="Simulation", **kwargs)
    # Fit (ML fitting will be available in scipy 1.9.0 ; for now, we use a simple approximation)
    from scipy import stats
    dist = stats.gamma
    # guess = dict(a=duration.mean()**2/duration.var(), scale=duration.var()/duration.mean())
    # res = stats.fit(dist, duration, ADD GUESS HERE)
    # Approx ML solution from: https://en.wikipedia.org/wiki/Gamma_distribution#Maximum_likelihood_estimation
    s = np.log(duration.mean()) - np.log(duration).mean()
    param = {"a" : (3 - s + np.sqrt((s-3)**2 + 24*s)) / (12*s)}
    param["scale"] = duration.mean() / param["a"]
    pdf = dist(**param).pdf
    x = np.linspace(bins[0], bins[-1], 101)
    ax.plot(x, pdf(x), c='magenta', lw=1., label=f"Gamma(a={param['a']:.2f}, θ={param['scale']:.2f})")
    # Beautify
    ax.set_xlim(bins[0], bins[-1])
    # ax.set_ylim()
    ax.set_xlabel("Duration [s]", labelpad=1)
    ax.set_ylabel("Frequency")
    leg = ax.legend(loc="upper right", fontsize=5.)
    ax.set_title(f"Mean duration: {duration.mean():.2f}s", pad=2)

    # # #  SAVE PLOT  # # #
    fname = "./fig/fig_SFM_perceptual_durations.pdf"
    print(f"> Save figure to '{fname}'")
    fig.savefig(fname)




# # # # # # # # # # #
# # #  S A V E  # # # 
# # # # # # # # # # #

out = dict(
    # First a bunch of config info
    DSL = DSL,
    fil = fil,
    tmax = tmax,
    midx = midx,
    threshrel = threshrel,
    # Here comes the analyzed data 
    threshabs = thresh,
    tswitch = tswitch,
    percept = percept,
    duration = duration
    )


if SAVE:
    import os
    fname = os.path.join(datadir, "analysis_SFM_" + DSL + ".pkl")
    import pickle
    with open(fname, "wb") as f:
        pickle.dump(out, f)
    si.log.info(f" > Data saved to file '{fname}'.")





