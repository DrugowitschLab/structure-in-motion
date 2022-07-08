"""
Plot Figure 6 (Structure from Motion)
"""
import numpy as np
import strinf as si

# # #  PARAMETERS  # # # 
ZOOM = 2.0
SAVE = True
SHOW = True
PANEL = ("D", "E", "H", "I", "J")

param = {
    "datadir" : "data",
    "DSL_one_cylinder" : "2022-06-17-11-01-10-404662_121_SFM_one_cylinder_switching_distribution_short_demo_trace",
    "fname_switching_times" : "analysis_SFM_2022-06-17-11-03-33-098346_121_SFM_one_cylinder_switching_distribution.pkl",
    "switching_threshold" : 1.18,   # Enter output from analysis script: ./ana/SFM_switching_times.py
    "DSL_outer_slow_inner_slow" : "2022-05-31-17-07-55-799200_121_SFM_nested_cylinders_outer_slow_inner_slow",
    "DSL_outer_slow_inner_fast" : "2022-05-31-17-09-21-600215_121_SFM_nested_cylinders_outer_slow_inner_fast",
    "DSL_outer_fast_inner_slow" : "2022-05-31-17-10-26-795580_121_SFM_nested_cylinders_outer_fast_inner_slow",
    "m_rot_one" : [(1,1)],  # format (D,M)
    "color_one" : [si.colors.get_color(si.colors.CMAP['clu'], i=0, N=2)] ,
    "m_rot_nested" : [(1,1), (1,2), (1,3)],  # format [shared, outer, inner] with each (D,M)
    "color_nested" : [si.colors.get_color(si.colors.CMAP['glo'], i=0, N=1),
                      si.colors.get_color(si.colors.CMAP['clu'], i=0, N=2),
                      si.colors.get_color(si.colors.CMAP['clu'], i=1, N=2)],
    "t_max_one" : 42.,
    "t_max_nested" : "auto",  # if 'auto': adjust the figure scale to match axes['source_one']
    "ymax_nested" : "auto",      # None, 'auto' (like 'source_one') or value (for comparability) 
}

# # #  END OF PARAMETERS  # # # 
# Choose backend
import pylab as pl
# pl.rcParams['toolbar'] = 'None'
if SHOW:
    pl.matplotlib.use("TkAgg")
else:
    pl.matplotlib.use("Agg")
    assert SAVE, "Neither showing (SHOW) nor saving (SAVE) the figure. This makes no sense."

si.plot.set_zoom(ZOOM)

# # #  AXES DEFINITIONS IN CM  # # # 
axeslayout = {  # rect=(left, top, width, height) in cm or inches
    # 1st row
    "sketch_SFM"  :            dict(rect=( 0.25, 0.25, 3.60, 2.25), xticks=[], yticks=[]),
    "sketch_rotation" :        dict(rect=( 4.25, 0.25, 2.25, 2.25), xticks=[], yticks=[]),  # incl. human percept
    "sketch_graph_one"  :      dict(rect=( 6.75, 0.25, 2.25, 2.25), xticks=[], yticks=[]), 
    "source_one" :             dict(rect=(10.00, 0.25, 3.75, 1.75), xlabel="Time [s]", ylabel="Rotational source", yticks=[-2,0,2]),
    "switch_distribution":     dict(rect=(15.00, 0.25, 2.75, 1.75), xlabel="Switching time [s]", ylabel="Frequency", yticks=[0,0.10]),
    # Second row
    "sketch_nested_slow_slow": dict(rect=( 0.25, 3.00, 1.50, 2.25), xticks=[], yticks=[]),
    "sketch_nested_graph":     dict(rect=( 2.00, 3.00, 2.25, 2.25), xticks=[], yticks=[]),
    "source_nested_slow_slow": dict(rect=( 5.15, 3.00, 2.25, 1.75), xlabel="Time [s]", ylabel="Rotational source", yticks=[-2,0,2]),
    "sketch_nested_slow_fast": dict(rect=( 7.90, 3.00, 1.50, 2.25), xticks=[], yticks=[]),
    "source_nested_slow_fast": dict(rect=(10.25, 3.00, 2.25, 1.75), xlabel="Time [s]", ylabel="Rotational source", yticks=[-2,0,2]),
    "sketch_nested_fast_slow": dict(rect=(13.15, 3.00, 1.50, 2.25), xticks=[], yticks=[]),
    "source_nested_fast_slow": dict(rect=(15.50, 3.00, 2.25, 1.75), xlabel="Time [s]", ylabel="Rotational source", yticks=[-2,0,2]),
    }

fig, axes = si.plot.init_figure_absolute_units(figsize=(18., 5.50), axes=axeslayout, units="cm")
fig.canvas.manager.set_window_title('Figure 6')

labels = {  # (label, x-pos, y-pos in absolute coords)
   "sketch_SFM"       :  ("a", 0.15, 0.32),
   "sketch_rotation"  :  ("b", 4.10, 0.32),
   "sketch_graph_one" :  ("c", 6.55, 0.32),
   "source_one"       :  ("d", 9.15, 0.32),
   "switch_distribution":("e",14.15, 0.32),
   # 2nd row
   "sketch_nested_slow_slow" :  ("f", 0.15, 3.07),
   "sketch_nested_graph"     :  ("g", 1.80, 3.07),
   "source_nested_slow_slow" :  ("h", 4.35, 3.07),
   "sketch_nested_slow_fast" :  ("i", 7.80, 3.07),
   "sketch_nested_fast_slow" :  ("j",13.05, 3.07),
   }

for axname, (label, x, y) in labels.items():
    ax = axes[axname]
    si.plot.print_panel_label_abcolute_units(ax, label, x, y, units="cm", fontkwargs=dict(fontsize=6.))



# # #  PLOT THE SKETCHES  # # #

def make_ax_size_pixel_precise(ax, im, fig, dpi=600):
    """
    Adjust the width and height of axes ax to exactly fit image im. This is mostly for sub-pixel fine-adjustment.
    
    Sometimes this helps; sometimes it does not. The ways of MPL are inscrutable.
    """
    # We need a bunch of transformers
    from matplotlib.transforms import Bbox, BboxTransformFrom, TransformedBbox
    trabs = BboxTransformFrom(Bbox.from_bounds(0, 0, *fig.get_size_inches()))
    trfig = fig.transFigure.inverted()
    # current axes dimensions in inches 
    l,b,w,h =  ax.bbox.transformed(trfig).transformed(trabs.inverted()).bounds
    # image pixel width and height
    ph, pw = im.shape[:2]
    # target width and height in inches
    wnew, hnew = pw/dpi, ph/dpi
    # print("Old:", w,h, "\nNew:", wnew, hnew)
    # new dims
    rect = TransformedBbox(Bbox.from_bounds(l, b, wnew, hnew), trabs) # .bounds
    # apply
    ax.set_position(rect)


# TODO: remove """ aspect='auto' """ after DEV.

# Panel A
ax = axes["sketch_SFM"]
im = pl.imread("./panel/sketch_SFM_paradigm.png")
ax.imshow(im, aspect='auto')
make_ax_size_pixel_precise(ax, im, fig)
ax.set_frame_on(False)

# Panel B
ax = axes["sketch_rotation"]
im = pl.imread("./panel/sketch_SFM_top_view.png")
ax.imshow(im, aspect='auto')
# make_ax_size_pixel_precise(ax, im, fig)
ax.set_frame_on(False)

# Panel C
ax = axes["sketch_graph_one"]
im = pl.imread("./panel/sketch_SFM_graph_one.png")
ax.imshow(im, aspect='auto')
make_ax_size_pixel_precise(ax, im, fig)
ax.set_frame_on(False)

# Panel F
ax = axes["sketch_nested_slow_slow"]
im = pl.imread("./panel/sketch_SFM_nested_slow_slow.png")
ax.imshow(im, aspect='auto')
make_ax_size_pixel_precise(ax, im, fig)
ax.set_frame_on(False)

# Panel G
ax = axes["sketch_nested_graph"]
im = pl.imread("./panel/sketch_SFM_nested_graph.png")
ax.imshow(im, aspect='auto')
make_ax_size_pixel_precise(ax, im, fig)
ax.set_frame_on(False)

# Panel I
ax = axes["sketch_nested_slow_fast"]
im = pl.imread("./panel/sketch_SFM_nested_slow_fast.png")
ax.imshow(im, aspect='auto')
make_ax_size_pixel_precise(ax, im, fig)
ax.set_frame_on(False)

# Panel J
ax = axes["sketch_nested_fast_slow"]
im = pl.imread("./panel/sketch_SFM_nested_fast_slow.png")
ax.imshow(im, aspect='auto')
make_ax_size_pixel_precise(ax, im, fig)
ax.set_frame_on(False)


def plot_sources(ax, DSL, midx, colors, tmax, thresh=None, ymax=None):
    # Load data
    ds, cfg = si.load_dataset(DSL, datadir=param["datadir"], F="adiab", R=0)
    S = ds.S_inf
    T = S.t
    # select time
    tidx = T <= tmax
    S = S[tidx]
    T = T[tidx]
    # # # Plot
    # Source
    lines = []
    for m,c in zip(midx, colors):
        lines += ax.plot(T, S[:,m[0], m[1]], c=c, lw=0.50, zorder=5)
    # Threshold
    if thresh is not None:
        ax.hlines((-thresh,thresh), 0., tmax, colors="0.4", linestyles="--", lw=0.5, zorder=10)
    ax.hlines((0.), 0., tmax, colors="0.7", linestyles="--", lw=0.5, zorder=1)
    # Beautify
    ax.set_xlim(0., tmax)
    if ymax is None:
        ymax = max(np.abs(ax.yaxis.get_data_interval()))
        ax.set_ylim(-1.05*ymax, 1.05*ymax)
    elif ymax == "auto":
        ymax = axes["source_one"].get_ylim()[1]
        ax.set_ylim(-ymax, ymax)
    else:
        ax.set_ylim(-ymax, ymax)
    ax.set_ylabel("Rotational source" + ("s" if len(lines) > 1 else ""), labelpad=1)
    return lines    


# # #  PANEL D (rotational source, single cylinder) # # #
if "D" in PANEL:
    lines = plot_sources(ax = axes["source_one"],
                         DSL = param["DSL_one_cylinder"],
                         midx = param["m_rot_one"],
                         colors = param["color_one"],
                         tmax = param["t_max_one"],
                         thresh = param["switching_threshold"],
                         ymax = None,
                         )


# # #  PANEL E (switching distribution, single cylinder) # # #
if "E" in PANEL:
    ax = axes["switch_distribution"]
    import os
    fname = os.path.join(param["datadir"], param["fname_switching_times"])
    import pickle
    with open(fname, "rb") as f:
        datadict = pickle.load(f)

    duration = np.array(datadict["duration"])
    # # #  Histogram  # # #
    tmean = duration.mean()
    w = 1.0 # tmean / 5.
    bins = np.concatenate(([0.], np.arange(w/2, 4.0*tmean, w)))
    kwargs = dict(color=param["color_one"][0], alpha=0.5)
    n,bins,patches = ax.hist(duration, bins, density=True, label="Simulation", **kwargs)
    # Fit (ML fitting will be available in scipy 1.9.0 ; for now, we use an approximation)
    # Approx ML solution from: https://en.wikipedia.org/wiki/Gamma_distribution#Maximum_likelihood_estimation
    from scipy import stats
    dist = stats.gamma
    s = np.log(duration.mean()) - np.log(duration).mean()
    P = {"a" : (3 - s + np.sqrt((s-3)**2 + 24*s)) / (12*s)}
    P["scale"] = duration.mean() / P["a"]
    pdf = dist(**P).pdf
    x = np.linspace(bins[0], bins[-1], 101)
    ax.plot(x, pdf(x), c="k", lw=1., label=f"Γ-dist.(fit)")
    # Beautify
    ax.set_xlim(bins[0], bins[-1]/2)
    # ax.set_ylim()
    ax.set_xlabel("Duration [s]", labelpad=1)
    ax.set_ylabel("Rel. frequency")
    # leg = ax.legend(loc="upper right", fontsize=5.)
    # Reorder legend labels:
    handles, labels = ax.get_legend_handles_labels()
    order = [1,0]
    leg = ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc="upper right", fontsize=5.)
    # Add mean value
    m = duration.mean()
    ax.vlines(m, 0., pdf(m), linestyles=(0,(2,1)), colors="k", lw=1.)
    ax.text(x=0.95*m, y=0.33*pdf(m), s=f"Mean: {m:.2f}s", ha="right", va="center", size=6.)
    # ax.set_title(f"Mean duration: {duration.mean():.2f}s", pad=2)


# # #  PANEL H (rotational source, two cylinders, slow + slow) # # #
if "H" in PANEL:
    ax = axes["source_nested_slow_slow"]
    if param["t_max_nested"] == "auto":
        w_one = axes['source_one'].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width
        w_here = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width
        t_max = param["t_max_one"] * w_here / w_one
    else:
        t_max = param["t_max_nested"]
    lines = plot_sources(ax = ax,
                         DSL = param["DSL_outer_slow_inner_slow"],
                         midx = param["m_rot_nested"],
                         colors = param["color_nested"],
                         tmax = t_max,
                         thresh = None,
                         ymax = param["ymax_nested"]
                         )

# # #  PANEL I (rotational source, two cylinders, slow + fast) # # #
if "I" in PANEL:
    ax = axes["source_nested_slow_fast"]
    if param["t_max_nested"] == "auto":
        w_one = axes['source_one'].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width
        w_here = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width
        t_max = param["t_max_one"] * w_here / w_one
    else:
        t_max = param["t_max_nested"]
    lines = plot_sources(ax = ax,
                         DSL = param["DSL_outer_slow_inner_fast"],
                         midx = param["m_rot_nested"],
                         colors = param["color_nested"],
                         tmax = t_max,
                         thresh = None,
                         ymax = param["ymax_nested"]
                         )

# # #  PANEL J (rotational source, two cylinders, fast + slow) # # #
if "J" in PANEL:
    ax = axes["source_nested_fast_slow"]
    if param["t_max_nested"] == "auto":
        w_one = axes['source_one'].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width
        w_here = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width
        t_max = 4 * param["t_max_one"] * w_here / w_one
    else:
        t_max = param["t_max_nested"]
    lines = plot_sources(ax = ax,
                         DSL = param["DSL_outer_fast_inner_slow"],
                         midx = param["m_rot_nested"],
                         colors = param["color_nested"],
                         tmax = t_max,
                         thresh = None,
                         ymax = param["ymax_nested"]
                         )




# # #  SAVE AND/OR SHOW  # # #
if SAVE:
    si.plot.savefig(fig, basename="Figure_6", path="./fig/", fmt=(".png", ".pdf"), dpi=600, logger=si.log)
else:
    si.log.warning("Figure not saved!")

if SHOW:
    pl.show()
    
