"""
Plot Figure 7 (Network)
"""
import numpy as np
import pylab as pl
import strinf as si
from strinf.plotting import truncate_colormap


# # #  PARAMETERS  # # # 
ZOOM = 1.5
SAVE = True
SHOW = True
PANEL = ("D", "G", "H", "I", "J", "M")   # The sketches in A, B, C, E, F, H, K, L are always plotted.

MT_tuning = {
    "neuronKwargs" : dict(Na=16, Nr=12, psi=0.1, rhomin=0.10, rhomax=8.0, \
                         exponent=1.25, kappaa = 1/0.35**2, sig2rho = 0.35**2),
    "mu_rho" : 5.,
    "mu_alpha" : np.pi/4.,
    "sig2" : (0.05/3)**2,    # default noise for location_indexed
}

net_demo = {
    "DSL" : "2021-09-02-12-03-15-153658_network_demo_polarCoords_and_global",
    "rng" : np.random.RandomState(1),     # rng for sampling a subset of neurons
    "stages" : (0.0, 1.0, 2.0, 3.0),        # for background coloring
    "fils" : ( ("net", "Network", "-") ,  ("adiab", "Online model", (0, (1,1))) ), # fmt: (filter, Legend entry, linestyle)
    "numPlot" : { 'inp' : 40, 'lin' : 25, 'one' : 8 },  # How many neurons in the subset
    "nLowPass" : { 'inp' : 6, 'lin' : 1, 'one' : 1 },  # How many time steps to low-pass-filter
    "yScale" : { 'inp' : 10, 'lin' : 8, 'one' : 4},  # y-axis scaling factor (it's arbitrary units, anyway)
    "rmin" : { 'inp' : 0.1, 'lin' : None, 'one' : None },  # only consider active neurons in subsample? (as fraction of max)
    "cmaps" : { 'inp' : si.colors.CMAP["velo"],         # color neurons from cmap
                'lin' : truncate_colormap(pl.cm.cool, 0.1, 0.6),
                'one' : truncate_colormap(pl.cm.bone, 0.85, 0.20),
                },
    "plotvars" : (#  which variables (not neurons) to plot
                  # fmt: dict(label for legend, key in ds, idx, color, skipincrement:plot on top of previous)
                  # idx=(spatial d, source m); spatial order: x/y OR r,phi
                 dict(label=r"$\mu_{\mathrm{ind}}$", key="mu", idx=(0,2), color=si.colors.get_color(si.colors.CMAP["ind"], 0, 4), skipincrement=False),
                 dict(label=r"$\mu_{\mathrm{ind}}$", key="mu", idx=(1,3), color=si.colors.get_color(si.colors.CMAP["ind"], 1, 4), skipincrement=True),
                 dict(label=r"$\mu_{\mathrm{ind}}$", key="mu", idx=(0,4), color=si.colors.get_color(si.colors.CMAP["ind"], 2, 4), skipincrement=True),
                 dict(label=r"$\mu_{\mathrm{ind}}$", key="mu", idx=(1,5), color=si.colors.get_color(si.colors.CMAP["ind"], 3, 4), skipincrement=True),
                 dict(label=r"$\mu_{\mathrm{polar,}r}$", key="mu", idx=(0,1), color=si.colors.SINGLE["rotation"]),
                 dict(label=r"$\mu_{\mathrm{polar,}\varphi}$", key="mu", idx=(1,1), color=si.colors.SINGLE["rotation"]),
                 dict(label=r"$\mu_{\mathrm{trans,}x}$", key="mu", idx=(0,0), color=si.colors.get_color(si.colors.CMAP["glo"], 0, 1)),
                 dict(label=r"$\mu_{\mathrm{trans,}y}$", key="mu", idx=(1,0), color=si.colors.get_color(si.colors.CMAP["glo"], 0, 1)),
               ) ,   
}


linear_readout = {
    "DSL" : "2021-09-03-11-10-56-930384_network_global_sweep_lamG",
    "fil" : "net",              # filter to use
    "pop" : "r_lin",            # neural population
    "duration" : 5.,            # length of the segement (t_end - duration, t_end) of neural activity to consider
    "idx_train" : (0, 6),       # which indices of lam_list are taken for training the regressor (remainder: prediction)
    "nPlot" : 500,              # avg. size of plotted subsample per q_i (to keep the file size acceptable)
}



# # #  END OF PARAMETERS  # # # 
# Choose backend
# pl.rcParams['toolbar'] = 'None'
if SHOW:
    pl.matplotlib.use("TkAgg")
else:
    pl.matplotlib.use("Agg")
    assert SAVE, "Neither showing (SHOW) nor saving (SAVE) the figure. This makes no sense."

si.plot.set_zoom(ZOOM)

# # #  AXES DEFINITIONS IN CM  # # # 
axeslayout = {
    "bg"     :     dict(rect=(0.0, 0.0, 18.0, 11.25), xticks=[], yticks=[], frame_on=False, zorder=-1),
    "sketch_net" : dict(rect=(0.25, 0.25, 4.25, 5.5), xticks=[], yticks=[]),
    "sketch_net_stim_and_centers" : dict(rect=(4.75, 0.25, 4.25, 5.50), xticks=[], yticks=[]),
    "example_tuning" : dict(rect=(7.035, 3.80, 1.70, 1.70), xticks=[], yticks=[], projection="polar"),
    "demo_tree":   dict(rect=(9.50, 0.25, 1.50, 2.20), xticks=[], yticks=[]),
    "demo_stim_1": dict(rect=(11.365, 0.25, 1.50, 1.75), xticks=[], yticks=[]),
    "demo_stim_2": dict(rect=(13.625, 0.25, 1.50, 1.75), xticks=[], yticks=[]),
    "demo_stim_3": dict(rect=(15.875, 0.25, 1.50, 1.75), xticks=[], yticks=[]),
    "demo_time_arrow" : dict(rect=(11.25, 2.25, 6.25, 0.30), xticks=[], yticks=[]),
    "demo_var" :   dict(rect=(11.00, 2.75, 6.75, 2.25), xlabel="", ylabel="Latent variables"),
    "demo_net_bg": dict(rect=(11.00, 5.50, 6.75, 5.00), xticks=[], yticks=[], zorder=0), # just for a white background
    "demo_one" :   dict(rect=(11.00, 5.50, 6.75, 0.75), xticks=[], yticks=[], zorder=2),
    "demo_lin" :   dict(rect=(11.00, 6.25, 6.75, 2.00), xticks=[], yticks=[], zorder=3),
    "demo_inp" :   dict(rect=(11.00, 8.25, 6.75, 2.25), xlabel="Time [s]", yticks=[], zorder=1),
    "lro_sketch_stim" : dict(rect=(1.00, 6.50, 2.25, 2.25), xticks=[], yticks=[]),
    "lro_sketch_tree" : dict(rect=(0.25, 9.25, 3.75, 1.75), xticks=[], yticks=[]),
    "lro_lam2" :   dict(rect=(5.25, 6.75, 3.50, 3.50), xlabel="True lam2", ylabel="Linear readout", aspect="equal"),
    }

fig, axes = si.plot.init_figure_absolute_units(figsize=(18., 11.25), axes=axeslayout, units="cm")
fig.canvas.manager.set_window_title('Figure 7')

# remove frame from network background
for loc in ("left", "top", "right", "bottom"):
    axes["demo_net_bg"].spines[loc].set_visible(False)


# # #  BACKGROUND COLORS AND PANEL LABELS  # # #

# good bg colors: papayawhip, lavenderblush, lavender, aliceblue, honeydew
c = np.array(pl.matplotlib.colors.to_rgba("cornsilk"))
c = 1. - 0.9 * (1 - c)
# Network demo
rect = (9.30, 0.05, 17.95-9.30, 11.20-0.05)
patch = si.plot.background_patch_absolute_units(axes["bg"], rect, color=c, units="cm")
# Linear readout experiment
rect = (0.05, 6.30, 9.00-0.05, 11.20-6.30)
patch = si.plot.background_patch_absolute_units(axes["bg"], rect, color="aliceblue", units="cm")

labels = {  # (label, x-pos, y-pos in absolute coords)  0.15, 0.32
    "sketch_net" :              ("a", 0.18, 0.32),
    "sketch_net_stim_and_centers" : ("b", 4.68, 0.32),
    "demo_time_arrow" :         ("c", 4.68, 3.20),  # This is a hack to have two labels for one axes
    "example_tuning" :          ("d", 6.92, 3.72),
    "demo_tree":                ("e", 9.43, 0.32),
    "demo_stim_1":              ("f",11.295,0.32),
    "demo_var" :                ("g", 9.63, 2.82),
    "demo_one" :                ("h", 9.63, 5.57),
    "demo_lin" :                ("i", 9.63, 6.32),
    "demo_inp" :                ("j", 9.63, 8.32),
    "lro_sketch_stim" :         ("k", 0.83, 6.72),
    "lro_sketch_tree" :         ("l", 0.15, 9.32),
    "lro_lam2" :                ("m", 4.08, 6.72),
   }

for axname, (label, x, y) in labels.items():
    ax = axes[axname]
    si.plot.print_panel_label_abcolute_units(ax, label, x, y, units="cm", fontkwargs=dict(fontsize=6., ha='center'))


# # #  AUXILIARY FUNCTIONS  # # #
def boxfilter(y, N):
    """
    Like a convolution with [1/N]*N, but with "shorter" kernel at the beginning to avoid boundary effects.
    """
    n = np.clip(np.arange(len(y)), 0, N-1) + 1
    return np.array([ y[i+1-ni:i+1].sum() / ni for i,ni in zip(range(len(y)), n) ])

def plot_random_subset(t, r, n, cmap, ax, scale=3.0, convolve=1, rmin=None):
    # remove silent neurons
    if rmin is not None:
        thresh = rmin * r.max()
        r = r[:, r.max(0) >= thresh]
    # Random subset of neurons
    idx = net_demo["rng"].choice(np.arange(r.shape[1]), n, replace=False)
    r = r[:,idx]
    # temporal smoothing of rates
    r = np.array([ boxfilter(ri, convolve) for ri in r.T]).T
    # Scale rates for plotting
    if 0 <= r.min():
        r *= 1 * scale / (r.max() - r.min())   # Scale = 1 --> maximally reach up to next neuron's baseline
    else:
        r *= 2 * scale / (r.max() - r.min())   # Scale = 1 --> maximally reach up to next neuron's baseline
    # shift the baselines
    for i in range(n):
        r[:,i] += i
    color = cmap(net_demo["rng"].rand(n))
    kwargs = dict(lw=0.5, clip_on=False)
    for ri,ci in zip(r.T, color):
        ax.plot(t, ri, color=ci, **kwargs)


def add_stages_bg(ax):
    for i,(l,r) in enumerate(zip(net_demo["stages"][:-1], net_demo["stages"][1:])):
        c = ("w", "k")[i%2]
        ax.axvspan(l, r, color=c, lw=0., alpha=0.075, zorder=-2)



# # #  PLOT THE SKETCHES  # # #


ax = axes["sketch_net"]
ax.imshow(pl.imread("./panel/sketch_net_connectome.png"))
ax.set_frame_on(False)

ax = axes["sketch_net_stim_and_centers"]
ax.imshow(pl.imread("./panel/sketch_net_stim_and_centers.png"))
ax.set_frame_on(False)

ax = axes["demo_tree"]
ax.imshow(pl.imread("./panel/sketch_net_demo_tree.png"))
ax.set_frame_on(False)

ax = axes["demo_stim_1"]
ax.imshow(pl.imread("./panel/sketch_net_demo_stim_1.png"))
ax.set_frame_on(False)

ax = axes["demo_stim_2"]
ax.imshow(pl.imread("./panel/sketch_net_demo_stim_2.png"))
ax.set_frame_on(False)

ax = axes["demo_stim_3"]
ax.imshow(pl.imread("./panel/sketch_net_demo_stim_3.png"))
ax.set_frame_on(False)

ax = axes["demo_time_arrow"]
ax.imshow(pl.imread("./panel/sketch_net_demo_time_arrow.png"))
ax.set_frame_on(False)


ax = axes["lro_sketch_stim"]
ax.imshow(pl.imread("./panel/sketch_lro_stim.png"))
ax.set_frame_on(False)

ax = axes["lro_sketch_tree"]
ax.imshow(pl.imread("./panel/sketch_lro_trees.png"))
ax.set_frame_on(False)


# # #  PANEL D (MT tuning example) # # #
if "D" in PANEL:
    ax = axes['example_tuning']
    # Tuning function for the example neuron
    f = si.make_MT_tuning_function(returnFsingle=True, **MT_tuning["neuronKwargs"])
    # Calculate neuron indices (na, nr). For the example, not necessarily integer-valued.
    na = MT_tuning["mu_alpha"] / (2*np.pi / MT_tuning["neuronKwargs"]["Na"])
    dr = (MT_tuning["neuronKwargs"]["rhomax"] - MT_tuning["neuronKwargs"]["rhomin"]) / (MT_tuning["neuronKwargs"]["Nr"]-1)**MT_tuning["neuronKwargs"]["exponent"]
    mu_rho_inv = lambda m: ((m - MT_tuning["neuronKwargs"]["rhomin"])/dr)**(1/MT_tuning["neuronKwargs"]["exponent"])
    nr = mu_rho_inv(MT_tuning["mu_rho"])
    # Meshgrid for plotting
    Alpha = np.linspace(0, 2*np.pi, 129)
    Rho = np.linspace(0.01, 9.0, 33)
    R, Theta = np.meshgrid(Rho, Alpha)
    Z = f(Theta, R, MT_tuning["sig2"], na, nr)
    si.log.info(f"Panel d: MT neuron max. firing rate: {Z.max():.2f} spk/s")
    # Plot
    from strinf.plotting import truncate_colormap
    kwargs = dict(vmin=0, vmax=Z.max(), cmap=truncate_colormap(pl.cm.hot_r, 0, 0.9), shading='nearest', zorder=-1, antialiased=True)
    ax.pcolormesh(Theta, R, Z, **kwargs)
    # Beautify
    ax.grid(lw=0.5, color='0.35', alpha=0.5)
    # Position the radial tick labels
    ax.set_ylim(0., 8.)
    ax.set_rlabel_position(-15.)
    ax.set_rticks([2, 5, 8])
    ax.set_yticklabels([2, 5, 8], fontdict=dict(va="top", fontsize=5, zorder=10))
    # Position the angluar tick labels
    ax.set_xticks(np.linspace(0,360,9)[:-1]/180*np.pi)
    ax.set_xticklabels([ f"{int(a)}°" if i%2==1 else ""  for i,a in enumerate(np.linspace(0,360,9)[:-1])], \
                         fontdict=dict(fontsize=5, va='center', ha='center'))
    ax.xaxis.set_tick_params(pad=-3)
    # set axes frame
    for spine in ax.spines.values():
        spine.set_edgecolor('0.7')
        spine.set_linewidth(0.5)



# # #  PANELs G - J (network demo) # # #
# if any(x in PANEL for x in ("G","H","I","J")):
# # #  LOAD DATA  # # #
fils = [fil for fil,_,_ in net_demo["fils"]]
ds, cfg = si.load_dataset(net_demo["DSL"], F=fils)
# Times
time = ds.t.data
# Load rates
rate = dict()
for pop in net_demo["numPlot"].keys(): # ("inp", "lin", "one"):
    r = ds["r_"+pop].loc[dict(R=0, F="net")].data
    rate[pop] = r
# Load latent variables
latent = dict()
for fil in fils:
    latent[fil] = dict()
    for mykey, key in (["lam","Lam_inf"], ["mu","S_inf"]):
        latent[fil][mykey] = ds[key].loc[dict(R=0, F=fil)].data
# # #  PLOTTING  # # #
# PLOT INPUT RATES
pop = "inp"
ax = axes["demo_" + pop]
plot_random_subset(time, rate[pop], net_demo["numPlot"][pop], net_demo["cmaps"][pop], \
                   ax, scale=net_demo["yScale"][pop], convolve=net_demo["nLowPass"][pop], rmin=net_demo["rmin"][pop])
add_stages_bg(ax)
ax.patch.set_alpha(0.0)
ax.set_xticks( np.arange(0, int(np.ceil(time[-1]))+1) )
ax.set_xlim(time[0], time[-1])
ax.set_ylim(-1, net_demo["numPlot"][pop])
for loc in ("top", "right"):
    ax.spines[loc].set_visible(False)
ax.set_xlabel("Time [s]", labelpad=2)
ax.set_ylabel("Input pop. [a.u.]")
# PLOT LINEAR RATES
pop = "lin"
ax = axes["demo_" + pop]
plot_random_subset(time, rate[pop], net_demo["numPlot"][pop], net_demo["cmaps"][pop], \
                   ax, scale=net_demo["yScale"][pop], convolve=net_demo["nLowPass"][pop], rmin=net_demo["rmin"][pop])
add_stages_bg(ax)
ax.patch.set_alpha(0.0)
ax.set_xlim(time[0], time[-1])
ax.set_ylim(-1, net_demo["numPlot"][pop])
for loc in ("top", "right", "bottom"):
    ax.spines[loc].set_visible(False)
ax.set_ylabel("Network activity\nDistributed pop.", linespacing=3.0)
# PLOT 1-to-1 RATES
pop = 'one'
ax = axes["demo_" + pop]
plot_random_subset(time, rate[pop], net_demo["numPlot"][pop], net_demo["cmaps"][pop], \
                   ax, scale=net_demo["yScale"][pop], convolve=net_demo["nLowPass"][pop], rmin=net_demo["rmin"][pop])
add_stages_bg(ax)
ax.patch.set_alpha(0.0)
ax.set_xlim(time[0], time[-1])
ax.set_ylim(0, net_demo["numPlot"][pop]+1)
for loc in ("top", "right", "bottom"):
    ax.spines[loc].set_visible(False)
ax.set_ylabel("1-to-1 pop.")
# PLOT SELECTION OF VARIABLES
ax = axes['demo_var']
ax.set_ylabel("Latent variables", labelpad=4)
add_stages_bg(ax)
# ax.patch.set_alpha(0.0)
# Iterate over filteres
plotvars = net_demo["plotvars"]
for fil,fillabel,fills in net_demo["fils"]:
    ticklabels = []
    nSkip = 0
    for i, d in enumerate(plotvars):
        scale = 3.0   # offset between vars is 1; delta-var=scale maps to delta-y=1 in the figure.
        s = tuple( [slice(None)] + [j for j in d['idx']] )  # a bit fancy slicing to support variable number of dims
        if ("skipincrement" in d) and (d["skipincrement"] is True):
            nSkip += 1
        else:
            ticklabels.append(d["label"])
        y = i - nSkip + latent[fil][d['key']][s] / scale
        l = fillabel if i==5 else None
        ax.plot(time, y, color=d['color'], ls=fills, label=l, clip_on=False)
# BEAUTIFY
leg = ax.legend(loc="upper left", fontsize=5, borderaxespad=0.35, borderpad=0.35, markerscale=1.5, handlelength=1)
leg.get_frame().set_linewidth(0.65)
ax.hlines( np.arange(len(plotvars)-nSkip), time[0], time[-1], color='0.7', lw=0.3, zorder=-1 )
ax.set_xticks( np.arange(0, int(np.ceil(time[-1]))+1) )
ax.set_xlim(time[0], time[-1])
ax.set_yticks(np.arange(len(plotvars)-nSkip))
ax.set_yticklabels(ticklabels)
ax.set_ylim(-0.25, len(plotvars)-nSkip-0.75)
for loc in ("top", "right"):
    ax.spines[loc].set_visible(False)

    

# # #  PANEL M (linear readout of lam_glo**2 / lam_tot**2) # # #
if "M" in PANEL:
    ax = axes['lro_lam2']
    # # #  Load data  # # #
    ds, cfg = si.load_dataset(linear_readout["DSL"], F=linear_readout["fil"])
    lam_list = cfg["wld"]["lamList"]
    reps = cfg["glo"]["R"]
    Lam = np.array([ lam for _,lam in lam_list ])
    T_end = np.array([ ti + lam_list[1][0] - lam_list[0][0] for ti,_ in lam_list ])
    lam2_tot = Lam[0,0]**2 + Lam[0,1]**2    # Total expected speed
    Q = Lam[:,0]**2 / lam2_tot              # fractions q_i := lam_glo**2 / lam_tot**2
    nQ = len(T_end)                         # number of fractions q_i
    for lam in Lam: # sanity check
        assert np.allclose(lam[0]**2 + lam[1:]**2, lam2_tot)
    # # #  Reshape the data array and select relevant times  # # #
    R = ds[linear_readout["pop"]]         # Firing rates; shape (reps, t_tot, N)
    N = R.shape[-1]                       # num Neurons
    nTperlam = int((ds.t < T_end[0]).sum())  # num time steps per lambda value
    R = R.data[:,:-1].reshape(reps, nQ, nTperlam, N)  # drop very last time step (for consistency with previous lam's), and reshape
    tidx = (T_end[0] - linear_readout["duration"] <= ds.t) * (ds.t <= T_end[0]) # selected duration for evaluation
    R = R[:,:,tidx[:nTperlam],:]  # ...and slice out the requested data
    R = R.swapaxes(0,1)   # --> This is the data to work with: shape = (nQ, reps, duration, neurons)
    # # #  Build the training set  # # # 
    idx_train = linear_readout["idx_train"]
    y = np.hstack([ [Q[i]]*(R.shape[1] * R.shape[2]) for i in idx_train ])
    X = np.vstack([ R[i,r] for i in idx_train for r in range(reps)] )
    # # #  Fit the linear regression model  # # #
    from sklearn import linear_model
    clf = linear_model.LinearRegression()
    clf.fit(X, y)
    # # #  Plot the preditions on the training data  # # # 
    y_pred =  clf.coef_ @ X.T + clf.intercept_
    kwargs = dict(marker='.', ms=1., mew=0., lw=0.)
    subset = np.random.choice(np.arange(len(y_pred)), size = len(idx_train) * linear_readout["nPlot"], replace=False)
    ax.plot(y[subset] + np.random.normal(0,0.01,len(y[subset])), y_pred[subset], mfc='b', label="Training data", **kwargs)
    # # #  Build the test set and plot # # #
    idx_test = set(range(nQ)).difference(idx_train)
    y_true = np.hstack([ [Q[i]]*(R.shape[1] * R.shape[2]) for i in idx_test ])
    X = np.vstack([ R[i,r] for i in idx_test for r in range(reps)] )
    y_pred = clf.coef_ @ X.T + clf.intercept_
    subset = np.random.choice(np.arange(len(y_pred)), size = len(idx_test) * linear_readout["nPlot"], replace=False)
    ax.plot(y_true[subset] + np.random.normal(0,0.01,len(y_true[subset])), y_pred[subset], mfc='r', label="Decoded", **kwargs)
    # Beautify
    ax.plot([0,1], [0,1], '0.5', lw=0.75, zorder=10)
    ax.set_xlabel("True fraction of shared motion\n(noise added for visibility)")
    ax.set_ylabel("Fraction linearly decoded\nfrom neural activity")
    ax.set_xlim(1/16, 1-1/16)
    ax.set_ylim(1/16, 1-1/16)
    ax.set_xticks(Q)
    ax.set_xticklabels([f"${i}\,/\,8$" if i in (1,4,7) else "" for i in range(1,8)])
    ax.set_yticks(Q)
    ax.set_yticklabels([f"${i}\,/\,8$" if i in (1,4,7) else "" for i in range(1,8)])
    leg = ax.legend(loc='upper left', markerscale=4.5, fontsize=6, borderpad=0.4, labelspacing=0.4, handlelength=1)
    leg.get_frame().set_linewidth(0.65)





# # #  SAVE AND/OR SHOW  # # #
if SAVE:
    si.plot.savefig(fig, basename="Figure_7", path="./fig/", fmt=(".png", ".pdf"), dpi=600, logger=si.log)
else:
    si.log.warning("Figure not saved!")

if SHOW:
    pl.show()
    
