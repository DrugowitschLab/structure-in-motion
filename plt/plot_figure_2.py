"""
Plot Figure 2
"""
import numpy as np
import strinf as si

# # #  PARAMETERS  # # # 
ZOOM = 1.5
SAVE = True
SHOW = True
PANEL = ("C", "D", "F", "G")   # The sketches in A, B, E are always plotted.

johansson = {
    "DSL" : "2021-08-17-18-30-38-392007_020_3_dots_Johansson_1973",
    "lamTmax" : 20.,
    "lamfilterframes" : 1,
    "sfilterframes" : 1,
    "sTminmax" : (12.5, 17.5),
}

duncker = {
    "DSL" : "2021-08-18-11-25-56-034025_020_DunckerWheel",
    "lamTmax" : 10.,
    "lamfilterframes" : 1,
    "sfilterframes" : 3,
    "sTminmax" : (0., 10.),
}

# # #  END OF PARAMETERS  # # # 
# Choose backend
import pylab as pl
if SHOW:
    pl.matplotlib.use("TkAgg")
else:
    pl.matplotlib.use("Agg")
    assert SAVE, "Neither showing (SHOW) nor saving (SAVE) the figure. This makes no sense."

si.plot.set_zoom(ZOOM)

# # #  AXES DEFINITIONS IN CM  # # # 
axeslayout = {
    "bg"     :      dict(rect=(0.0, 0.0, 18.0, 8.), xticks=[], yticks=[], frame_on=False, zorder=-1),
    "objidx" :      dict(rect=(0.25, 0.25, 3.25, 5.5), xticks=[], yticks=[]),
    "joh_sketch" :  dict(rect=(4.25, 0.25, 2.75, 2.75), xticks=[], yticks=[]),
    "joh_lam" :     dict(rect=(8.25, 0.25, 4.0, 2.75), xlabel="Time [s]", ylabel="Motion strengths λ"),
    "joh_C"   :     dict(rect=(8.50, 0.30, 4*0.175, 3*0.175), xticks=[], yticks=[]),
    "joh_s" :       dict(rect=(13.75, 0.25, 4.0, 2.75), xlabel="Time [s]", ylabel="Motion sources s"),
    "dun_sketch" :  dict(rect=(4.25, 4.25, 3.25, 3.25), xticks=[], yticks=[]),
    "dun_lam" :     dict(rect=(8.75, 4.25, 4.0, 2.75), xlabel="Time [s]", ylabel="Motion strengths λ"),
    "dun_C"   :     dict(rect=(9.00, 4.30, 3*0.175, 2*0.175), xticks=[], yticks=[]),
    "dun_s" :       dict(rect=(14.50, 4.25, 3.25, 2.75), xlabel="Sources s, x-direction", ylabel="Sources s, y-direction", aspect="equal"),
    }

fig, axes = si.plot.init_figure_absolute_units(figsize=(18.,7.75), axes=axeslayout, units="cm")
fig.canvas.manager.set_window_title('Figure 2')

# # #  BACKGROUND COLORS AND PANEL LABELS  # # #

# good bg colors: papayawhip, lavenderblush, lavender, aliceblue, honeydew
# Background Johansson 
rect = (3.75, 0.05, 17.95-3.75, 3.65)
patch = si.plot.background_patch_absolute_units(axes["bg"], rect, color="papayawhip", units="cm")
# Background Duncker
rect = (3.75, 4.05, 17.95-3.75, 3.65)
patch = si.plot.background_patch_absolute_units(axes["bg"], rect, color="lavender", units="cm")

labels = {  # (label, x-pos, y-pos in absolute coords)
    "objidx" :      ("a", 0.15, 0.32),
    "joh_sketch" :  ("b", 3.90, 0.32),
    "joh_lam" :     ("c", 7.40, 0.32),
    "joh_s" :       ("d", 12.90, 0.32),
    "dun_sketch" :  ("e", 3.90, 4.32),
    "bg" :          ("f", 3.90, 5.82),
    "dun_lam" :     ("g", 7.90 , 4.32),
    "dun_s" :       ("h", 13.65, 4.32),
    }

for axname, (label, x, y) in labels.items():
    ax = axes[axname]
    si.plot.print_panel_label_abcolute_units(ax, label, x, y, units="cm", fontkwargs=dict(fontsize=6.))


# # #  AUXILIARY FUNCTIONS  # # #
def boxfilter(y, N):
    """
    Like a convolution with [1/N]*N, but with "shorter" kernel at the beginning to avoid boundary effects.
    """
    n = np.clip(np.arange(len(y)), 0, N-1) + 1
    return np.array([ y[i+1-ni:i+1].sum() / ni for i,ni in zip(range(len(y)), n) ])



# # #  PLOT THE SKETCHES  # # #
ax = axes["objidx"]
ax.imshow(pl.imread("./panel/sketch_object_indexed.png"))
ax.set_frame_on(False)

ax = axes["joh_sketch"]
ax.imshow(pl.imread("./panel/sketch_Johansson.png"))
ax.set_frame_on(False)
# for spine in ax.spines.values():
#     spine.set_edgecolor('0.75')
#     spine.set_lw(0.75)

ax = axes["dun_sketch"]
ax.imshow(pl.imread("./panel/sketch_Duncker.png"))
ax.set_frame_on(False)
# for spine in ax.spines.values():
#     spine.set_edgecolor('0.75')
#     spine.set_lw(0.75)


# # #  PANEL C and D (Johansson) # # #
if "C" in PANEL or "D" in PANEL:
    # Load data
    ds, cfg = si.load_dataset(johansson["DSL"], F="adiab", R=0)
    # Get colors
    col = si.colors
    color = [col.get_color(col.CMAP["glo"], 0, 1)] + [col.get_color(col.CMAP["ind"], k, 3) for k in range(3) ]


if "C" in PANEL:
    tidx = ds.t <= johansson['lamTmax']
    t = ds.t[tidx]
    lam = ds.Lam_inf[tidx]

    # Plot lambda
    ax = axes['joh_lam']
    for l,c in zip(lam.T, color):
        l = boxfilter(l, N=johansson["lamfilterframes"])
        ax.plot(t, l, c=c)
    si.plot.auto_axes_lim(ax, which=("x", "y"), ymin=0., ymargin=0.07)

    # Highligh region in D
    x0, x1 = johansson["sTminmax"]
    y0, y1 = ax.get_ylim()
    patch = pl.matplotlib.patches.Rectangle((x0,y0), x1-x0, y1-y0, lw=0., ec='0.9', fc='0.9', zorder=-1)
    ax.add_patch(patch)

    # Plot C matrix
    C = np.array(cfg['fil']['default_params']['C'])
    viscolors = si.colors.get_colors(si.colors.CMAP['velo'], N=C.shape[0])
    si.imshow_C_matrix(C, axes['joh_C'], color, addVisibleCircles=True, viscolors=viscolors)
    for spine in axes['joh_C'].spines.values():
        spine.set_edgecolor('0.75')
        spine.set_lw(0.5)


if "D" in PANEL:
    ax = axes["joh_s"]
    t0, t1 = johansson["sTminmax"]
    tidx =  (ds.t >= t0 ) * (ds.t <= t1)
    t = ds.t[tidx]

    # Plot mean
    import itertools
    # iterate over dimensions and sources
    for d,m in itertools.product( (0,1), (0,1,2,3) ):
        c = color[m]
        lw = 0.5 if (d,m) in ((0,0), (1,2)) else 0.35
        ls = '-' if (d,m) in ((0,0), (1,2)) else (0,(1,1))
        zorder = 5 if (d,m) in ((0,0), (1,2)) else 4
        mu = ds.S_inf[tidx].loc[dict(d=d, m=m)]
        mu = boxfilter(mu, N=johansson["sfilterframes"])
        ax.plot(t, mu, c=c, ls=ls, lw=lw, zorder=zorder)
        # error bars
        if (d,m) in ((0,0), (1,2)):
            sig = np.sqrt( ds.Sig_inf[tidx].loc[dict(d=d, m=m)] )
            sig = boxfilter(sig, N=johansson["sfilterframes"])
            y1, y2 = mu-sig, mu+sig
            ax.fill_between(t, y1, y2, lw=0., ec=None, fc=c, alpha=0.35, zorder=3)

    si.plot.auto_axes_lim(ax, which=("x", "y"), ymin=None, ymargin=-0.06)
    ymax = np.abs(ax.get_ylim()).max()
    ax.set_ylim(-ymax, ymax)
        
    # Annotations
    ax.annotate("Shared\nx-direction",
                xy=(14.13, 0.47), xycoords='data',
                xytext=(12.95, 0.75), textcoords='data',
                arrowprops=dict(arrowstyle="-", lw=0.75, color=color[0], connectionstyle="arc3,rad=0.4", shrinkA=0.),
                fontsize=5.5, color=color[0], zorder=10
                )

    ax.annotate("Center dot\ny-direction",
                xy=(16.40, 0.44), xycoords='data',
                xytext=(15.00, 0.78), textcoords='data',
                arrowprops=dict(arrowstyle="-", lw=0.75, color=color[2], connectionstyle="arc3,rad=0.4", shrinkA=0.),
                fontsize=5.5, color=color[2], zorder=10
                )
    

# # #  PANEL F and G (Duncker wheel) # # #
if "F" in PANEL or "G" in PANEL:
    # Load data
    ds, cfg = si.load_dataset(duncker["DSL"], F="adiab", R=0)
    # Get colors
    col = si.colors
    color = [col.get_color(col.CMAP["glo"], 0, 1)] + [col.get_color(col.CMAP["ind"], k, 3) for k in range(2) ]
    

if "F" in PANEL:
    tidx = ds.t <= duncker['lamTmax']
    t = ds.t[tidx]
    lam = ds.Lam_inf[tidx]

    # Plot lambda
    ax = axes['dun_lam']
    for l,c in zip(lam.T, color):
        l = boxfilter(l, N=duncker["lamfilterframes"])
        ax.plot(t, l, c=c)
    si.plot.auto_axes_lim(ax, which=("x", "y"), ymin=0., ymargin=0.07)

    ax.set_yticks([0,5,10])

    # Plot C matrix
    C = np.array(cfg['fil']['default_params']['C'])
    viscolors = si.colors.get_colors(si.colors.CMAP['velo'], N=3)[:2]
    si.imshow_C_matrix(C, axes['dun_C'], color, addVisibleCircles=True, viscolors=viscolors)
    for spine in axes['dun_C'].spines.values():
        spine.set_edgecolor('0.75')
        spine.set_lw(0.5)


if "G" in PANEL:
    ax = axes["dun_s"]
    ax.set_adjustable('datalim')  # force the box to be maintained --> adjust x/ylim to maintain aspect
    t0, t1 = duncker["sTminmax"]
    tidx =  (ds.t >= t0 ) * (ds.t <= t1)
    t = ds.t[tidx]

    for m in range(3):
        c = color[m]
        lw = 0.5 if m in (0,1) else 0.35
        # ls = '-' if m in (0,1) else (0,(1,1))
        ls = '-'
        mux = ds.S_inf[tidx].loc[dict(d=0, m=m)]
        mux = boxfilter(mux, N=duncker["sfilterframes"])
        muy = ds.S_inf[tidx].loc[dict(d=1, m=m)]
        muy = boxfilter(muy, N=duncker["sfilterframes"])

        cm = si.plot.cmap_white(color=c)
        n = len(t)
        ax.set_prop_cycle(color=[cm(1.*i/(n-2)) for i in range(n-1)])
        for i in range(n-1):
            ax.plot(mux[i:i+2], muy[i:i+2], ls=ls, lw=lw, zorder=i)

    ymax = np.max([abs(ax.dataLim.ymin), abs(ax.dataLim.ymax)])
    ax.set_ylim(-1.05 * ymax, 1.05 * ymax)
    ax.set_xticks([-5,0,5,10])
    ax.set_yticks([-5,0,5])


# # #  SAVE AND/OR SHOW  # # #
if SAVE:
    si.plot.savefig(fig, basename="Figure_2", path="./fig/", fmt=(".png", ".pdf"), dpi=600, logger=si.log)
else:
    si.log.warning("Figure not saved!")

if SHOW:
    pl.show()
    