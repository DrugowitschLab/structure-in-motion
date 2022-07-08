"""
Plot Figure 5 (Lorenceau)
"""
import numpy as np
import strinf as si

# # #  PARAMETERS  # # # 
ZOOM = 2.0
SAVE = True
SHOW = True
PANEL = ("C",)

lorenceau = {
    "DSL_low_noise" : "2021-08-26-12-09-10-696387_110_lorenceau_1996_low_noise",
    "DSL_high_noise" : "2021-08-26-12-07-25-644304_110_lorenceau_1996_high_noise",
    "vfilterframes" : 12,  # 12 = 200ms at 60 Hz
    "tminmax" : dict(low=(0., 2.0), high=(0., 5.0)),
    "repr_idx" : (0, 15),   # which indices k will serve as representatives of their group
    "vxylim" : 2.5
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
axeslayout = {
    # "bg"     :      dict(rect=(0.0, 0.0, 18.0, 10.25), xticks=[], yticks=[], frame_on=False, zorder=-1),
    "lor_stim"  :   dict(rect=(3.025,  0.25, 2.75, 2.75), xticks=[], yticks=[]),
    # "lor_split" :   dict(rect=(2.60,  4.35, 3.45, 10.1-4.35), xticks=[], yticks=[]),  # incl. human percept
    "lor_split" :   dict(rect=(0.25,  3.20, 8.30, 3.00), xticks=[], yticks=[]),  # incl. human percept
    "lor_vlow"  :   dict(rect=(1.00,  6.60, 2.50, 2.50), xlabel="Perceived $v_x$", ylabel="Perceived $v_y$", aspect='equal'),
    "lor_vhigh" :   dict(rect=(8.80-1.0-2.50, 6.60, 2.50, 2.50), xlabel="", ylabel="", aspect='equal'),
    "lor_strlow":   dict(rect=(1.25, 10.25, 2.00, 0.85), xticks=[], yticks=[]),
    "lor_strhigh":  dict(rect=(6.125,10.25, 0.85, 0.85), xticks=[], yticks=[]),
    }

fig, axes = si.plot.init_figure_absolute_units(figsize=(8.8, 11.35), axes=axeslayout, units="cm")
fig.canvas.manager.set_window_title('Figure 5')

labels = {  # (label, x-pos, y-pos in absolute coords)
   "lor_stim"   :  ("a", 2.75, 0.32),
   "lor_split"  :  ("b", 0.15, 3.27),
   "lor_vlow"   :  ("c", 0.15, 6.45),
   "lor_strlow" :  ("d", 0.15,10.10),
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

ax = axes["lor_stim"]
ax.imshow(pl.imread("./panel/sketch_lor_vertical_stim.png"))
ax.set_frame_on(False)

ax = axes["lor_split"]
ax.imshow(pl.imread("./panel/sketch_lor_vertical_split.png"))
ax.set_frame_on(False)
fontkwargs = dict(fontsize=6., fontweight="normal", ha="center")
si.plot.print_panel_label_abcolute_units(ax, "Human percept", x=4.40, y=4.90, units="cm", fontkwargs=fontkwargs)

si.plot.print_panel_label_abcolute_units(axes["lor_vlow"], "Model perceived velocities", x=4.40, y=6.45, units="cm", fontkwargs=fontkwargs)

ax = axes["lor_strlow"]
ax.imshow(pl.imread("./panel/sketch_lor_vertical_strlow.png"))
ax.set_frame_on(False)
si.plot.print_panel_label_abcolute_units(ax, "Model motion decomposition", x=4.40, y=10.10, units="cm", fontkwargs=fontkwargs)

ax = axes["lor_strhigh"]
ax.imshow(pl.imread("./panel/sketch_lor_vertical_strhigh.png"))
ax.set_frame_on(False)





# # #  PANEL C (Lorenceau perceived velocities) # # #
if "C" in PANEL:
    # for top and bottom panel
    for noise in ("low", "high"):
        ax = axes[f"lor_v{noise}"]
        # ax.set_adjustable('datalim')  # force the box to be maintained --> adjust x/ylim to maintain aspect
        DSL = lorenceau[f"DSL_{noise}_noise"]
        # Load data
        ds, cfg = si.load_dataset(DSL, F="adiab", R=0)
        # Load sources for desired time slice
        tmin, tmax = lorenceau['tminmax'][noise]
        tidx = (tmin <= ds.t) * (ds.t <= tmax)
        S = ds.S_inf[tidx]
        # load C matrix
        C = np.array(cfg['fil']['default_params']['C'])
        # Erase self-motion (since projection is linear, this is the same as ignoring its contribution)
        mself = 0                                          # Index of the self-motion component 
        S[:,:,mself] = 0.
        # Perceived velocities
        V = S.data @ C.T        # --> dims: (t, x/y, k) with k=-1 for direct self-motion input
        Kdot = C.shape[0] - 1
        kidx = lorenceau["repr_idx"]
        for k in kidx:
            x,y = V[:,:,k].T
            x = boxfilter(x, N=lorenceau["vfilterframes"])
            y = boxfilter(y, N=lorenceau["vfilterframes"])
            # Build the cmap for gradient plots
            color = si.colors.get_color(si.colors.CMAP['velo'], k, Kdot)
            cm = si.plot.cmap_white(color=color)
            n = len(x)
            ax.set_prop_cycle(color=[cm(1.*i/(n-2)) for i in range(n-1)])
            kwargs = dict(lw=0.75, ls='-', )
            for i in range(n-1):
                ax.plot(x[i:i+2], y[i:i+2], zorder=i + 10000 * (k == kidx[0]), **kwargs)
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        ax.set_xlim(-lorenceau["vxylim"], lorenceau["vxylim"])
        ax.set_ylim(-lorenceau["vxylim"], lorenceau["vxylim"])

    axes["lor_vlow"].set_xlabel("Perceived $v_x$", labelpad=+1)
    axes["lor_vlow"].set_ylabel("Perceived $v_y$", labelpad=-1)
    axes["lor_vhigh"].set_xlabel("Perceived $v_x$", labelpad=+1)
    axes["lor_vhigh"].set_ylabel("Perceived $v_y$", labelpad=-1)


# # #  SAVE AND/OR SHOW  # # #
if SAVE:
    si.plot.savefig(fig, basename="Figure_5", path="./fig/", fmt=(".png", ".pdf"), dpi=600, logger=si.log)
else:
    si.log.warning("Figure not saved!")

if SHOW:
    pl.show()
    
