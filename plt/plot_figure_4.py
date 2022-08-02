"""
Plot Figure 4
"""
import numpy as np
import strinf as si

# # #  PARAMETERS  # # # 
ZOOM = 1.5
SAVE = True
SHOW = True
PANEL = ("E", "F", "G", "H", "I", "J", "K", "L",)   # The sketches in A, B, C, D, are always plotted.

motion_repulsion = {
    # By angle
    "angle_model" : "./data/analysis_2021-08-19-12-14-27-196363_100_repulsion_by_angle_Braddick.pkl",
    "angle_braddick": "./data/data_Braddick_2002_Fig3C.txt",
    "angle_plot_kwargs" :  dict(ls='-',  lw=0.75, marker='o',  ms=1.5, c="slateblue"),
    # By contrast (2nd component)
    "contrast_model" : "./data/analysis_direction_repulsion_by_2nd_component_contrast.pkl",
    "contrast_plot_angles": ([45., dict(ls='-',  lw=0.75, marker='o',  ms=1.5, c='slateblue')],  # angle, plotKwargs
                             [20., dict(ls='-',  lw=0.75, marker=None, ms=1.5, c='cornflowerblue')]),
    "contrast_xmax" : 8.,
    # By speed (2nd component)
    "speed_model" : "./data/analysis_2021-08-19-15-18-47-392303_100_repulsion_by_2ndspeed_Braddick.pkl",
    "speed_plot_angles": ([90., dict(ls=(0.7,(3,2)),  lw=0.75, marker=None,  ms=1.5, c='royalblue')],  # angle, plotKwargs
                          [60., dict(ls='-',  lw=0.75, marker='o',  ms=1.5, c='slateblue')],
                          [30., dict(ls='-',  lw=0.75, marker=None, ms=1.5, c='cornflowerblue')]),
}

takemura = {
    "data" : "./data/analysis_direction_repulsion_Takemura.pkl",
    "order": ('multy_0_surround_bi', 'multy_0_surround_down', 'multy_1_surround_down', 'multy_1_surround_bi', 'multy_1_surround_up'),
    "bins" : np.linspace(-22.5,202.5,46)/180*np.pi,
    "kinner" : (0,1),                                       # Indices of the inner dots (the ones of interest)
    "kouter" : (2,3),                                       # Indices of the outer dots
    "count_max" : 60,
    "count_mark" : 50,
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
    "bg"     :      dict(rect=(0.0, 0.0, 18.0, 10.25), xticks=[], yticks=[], frame_on=False, zorder=-1),
    "locidx" :      dict(rect=(0.25, 0.25, 3.25, 5.0), xticks=[], yticks=[]),
    "tree"   :      dict(rect=(3.75, 0.25, 2.75, 3.25), xticks=[], yticks=[]),
    "v_perc" :      dict(rect=(6.75, 0.25, 2.50, 3.25), xticks=[], yticks=[]),
    "mdr_sketch":   dict(rect=(9.80, 0.25, 3.00, 3.25), xticks=[], yticks=[]),
    "mdr_angle" :   dict(rect=(14.00, 0.25, 3.75, 2.75), xlabel=r"Opening angle $\gamma$ [°]", ylabel="Bias of perceived angle [°]"),
    "mdr_in1"   :   dict(rect=(14.80, 2.05, 0.85, 0.85), xticks=[], yticks=[]),
    "mdr_in2"   :   dict(rect=(16.50, 0.35, 0.85, 0.85), xticks=[], yticks=[]),
    "mdr_contrast_sketch" : dict(rect=(12.25, 4.20, 1.80, 2.25), xticks=[], yticks=[]),
    "mdr_contrast": dict(rect=(15.00, 4.10, 2.75, 2.00), xlabel="1/$\sigma_\mathrm{obs}^2$ of 2nd group (factor)", ylabel="Bias of 1st group [°]"),
    "mdr_speed_sketch" : dict(rect=(12.25, 7.30, 1.80, 2.25), xticks=[], yticks=[]),
    "mdr_speed" :   dict(rect=(15.00, 7.20, 2.75, 2.00), xlabel="Speed of 2nd group (factor)", ylabel="Bias of 1st group [°]"),
    "tak_0_hist"   : dict(rect=(04.60, 4.70, 3.00, 2.50), polar=True),
    "tak_0_sketch" : dict(rect=(04.05, 4.35, 1.10, 1.10), xticks=[], yticks=[], aspect="equal"),
    "tak_1_hist"   : dict(rect=(08.50, 4.70, 3.00, 2.50), polar=True),
    "tak_1_sketch" : dict(rect=(07.95, 4.35, 1.10, 1.10), xticks=[], yticks=[], aspect="equal"),
    "tak_2_hist"   : dict(rect=(00.70, 7.60, 3.00, 2.50), polar=True),
    "tak_2_sketch" : dict(rect=(00.15, 7.25, 1.10, 1.10), xticks=[], yticks=[], aspect="equal"),
    "tak_3_hist"   : dict(rect=(04.60, 7.60, 3.00, 2.50), polar=True),
    "tak_3_sketch" : dict(rect=(04.05, 7.25, 1.10, 1.10), xticks=[], yticks=[], aspect="equal"),
    "tak_4_hist"   : dict(rect=(08.50, 7.60, 3.00, 2.50), polar=True),
    "tak_4_sketch" : dict(rect=(07.95, 7.25, 1.10, 1.10), xticks=[], yticks=[], aspect="equal"),
    }

fig, axes = si.plot.init_figure_absolute_units(figsize=(18., 10.), axes=axeslayout, units="cm")
fig.canvas.manager.set_window_title('Figure 4')

# # #  BACKGROUND COLORS AND PANEL LABELS  # # #

# good bg colors: papayawhip, lavenderblush, lavender, aliceblue, honeydew
# Background motion repulsion 
rect = (9.50, 0.05, 17.95-9.50, 3.75-0.05)
patch = si.plot.background_patch_absolute_units(axes["bg"], rect, color="aliceblue", units="cm")
rect = (12.05, 3.00, 17.95-12.05, 9.95-3.00)
patch = si.plot.background_patch_absolute_units(axes["bg"], rect, color="aliceblue", units="cm")
# Background Lorenceau
rect = (0.05, 6.95, 11.75-0.05, 9.95-5.55)
c = np.array(pl.matplotlib.colors.to_rgba("cornsilk"))
c = 1. - 0.9 * (1 - c)
patch = si.plot.background_patch_absolute_units(axes["bg"], rect, color=c, units="cm")
rect = (3.75, 4.05, 11.75-3.75, 9.95-4.05)
patch = si.plot.background_patch_absolute_units(axes["bg"], rect, color=c, units="cm")

labels = {  # (label, x-pos, y-pos in absolute coords)
   "locidx"     :  ("a", 0.15, 0.32),
   "tree"       :  ("b", 3.45, 0.32),
   "v_perc"     :  ("c", 6.65, 0.32),
   "mdr_sketch" :  ("d", 9.60, 0.32),
   "mdr_angle"  :  ("e",13.00, 0.32),
   "mdr_contrast": ("f",12.15, 4.12),
   "mdr_speed"  :  ("g",12.15, 7.17),
   "tak_0_sketch" :("h", 4.05, 4.37),
   "tak_1_sketch" :("i", 7.95, 4.37),
   "tak_2_sketch" :("j", 0.20, 7.27),
   "tak_3_sketch" :("k", 4.05, 7.27),
   "tak_4_sketch" :("l", 7.95, 7.27),
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
ax = axes["locidx"]
ax.imshow(pl.imread("./panel/sketch_location_indexed.png"))
ax.set_frame_on(False)

ax = axes["tree"]
ax.imshow(pl.imread("./panel/sketch_tree_with_selfmotion.png"))
ax.set_frame_on(False)

ax = axes["v_perc"]
ax.imshow(pl.imread("./panel/sketch_perceived_velocity.png"))
ax.set_frame_on(False)

ax = axes["mdr_sketch"]
ax.imshow(pl.imread("./panel/sketch_mdr_angle.png"))
ax.set_frame_on(False)

ax = axes["mdr_in1"]
ax.imshow(pl.imread("./panel/sketch_mdr_angle_in1.png"))
ax.set_frame_on(False)
# Annotation arrow
axes['mdr_angle'].annotate("",
            xy=(123., 17.), xycoords='data',
            xytext=(90., 9.), textcoords='data',
            arrowprops=dict(arrowstyle="-", lw=0.75, color='0.8', connectionstyle="arc3,rad=-0.4", shrinkA=0.),
            fontsize=5.5, color='0.8', zorder=-1
            )


ax = axes["mdr_in2"]
ax.imshow(pl.imread("./panel/sketch_mdr_angle_in2.png"))
ax.set_frame_on(False)
# Annotation arrow
axes['mdr_angle'].annotate("",
            xy=(39.5, -8.), xycoords='data',
            xytext=(20., -11.5), textcoords='data',
            arrowprops=dict(arrowstyle="-", lw=0.75, color='0.8', connectionstyle="arc3,rad=0.3", shrinkA=0.),
            fontsize=5.5, color='0.8', zorder=-1
            )


ax = axes["mdr_contrast_sketch"]
ax.imshow(pl.imread("./panel/sketch_mdr_contrast.png"))
ax.set_frame_on(False)

ax = axes["mdr_speed_sketch"]
ax.imshow(pl.imread("./panel/sketch_mdr_speed.png"))
ax.set_frame_on(False)


# # #  PANEL E (motion direction repulsion by opening angle) # # #
if "E" in PANEL:
    ax = axes['mdr_angle']
    # # # Model
    import pickle
    with open(motion_repulsion["angle_model"], "rb") as f:
        X = pickle.load(f)
    x = X["stimulus_angle"] / np.pi * 180
    y = X["perceived_opening_angle_mean"] / np.pi * 180 - x
    yerr = X["perceived_opening_angle_sem"] / np.pi * 180
    ax.plot(x, y, label="Model (avg.)", zorder=2, **motion_repulsion["angle_plot_kwargs"])
    # ax.fill_between(x, y-yerr, y+yerr, lw=0., ec=None, fc='b', alpha=0.35, zorder=0)  # too small
    # # # Experiment
    try:
        B = np.loadtxt(motion_repulsion["angle_braddick"])
        kwargs = dict(fmt='o', color='k', ms=2.5, capsize=1.25, elinewidth=0.5, markeredgewidth=0.5, zorder=2)
        ax.errorbar(B[:,0], B[:,1], yerr=B[:,2], label="Braddick (2002)", **kwargs)
    except:
        si.log.warning("Cannot load experimental data.")
    # # # Decoration
    ax.hlines(0., x[0], x[-1], lw=0.75, ls=':', colors='0.7', zorder=0)
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
    si.plot.auto_axes_lim(ax, which=("x", "y"), ymin=None, ymargin=0.02)
    # # # Legend (ordering from: https://stackoverflow.com/a/27512450)
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    leg = ax.legend(handles, labels, loc=(0.51, 0.04), fontsize=5, facecolor='0.95', handlelength=1)
    leg.get_frame().set_linewidth(0.65)
    # # #  INSETS  # # #
    # TODO


    
# # #  PANEL F (motion direction repulsion by 2nd contrast) # # #
if "F" in PANEL:
    ax = axes['mdr_contrast']
    # # # Model
    import pickle
    with open(motion_repulsion["contrast_model"], "rb") as f:
        X = pickle.load(f)
    x = X["noiseFactor"]
    ximax = None if (motion_repulsion['contrast_xmax'] > x).all() else np.argmax(x > motion_repulsion['contrast_xmax'])
    angle = X["stimulus_opening_angle"] / np.pi * 180
    Y = X["perceived_1st_angle_mean"] / np.pi * 180 - angle[:,None] / 2  # angle/2 --> relative to 1st angle
    Yerr = X["perceived_1st_angle_sem"] / np.pi * 180
    for a,kwargs in motion_repulsion["contrast_plot_angles"]:
        i = tuple(angle).index(a)
        y = Y[i]
        if ximax:
            ax.plot(x[:ximax], y[:ximax], label=r"$\gamma$ = %d°" % int(a), **kwargs)
        else:
            ax.plot(x, y, label=r"$\gamma$ = %d°" % int(a), **kwargs)
    # # # Decoration
    if ximax:
        ax.hlines(0., x[0], x[ximax-1], lw=0.75, ls=':', colors='0.7', zorder=0)
    else:
        ax.hlines(0., x[0], x[-1], lw=0.75, ls=':', colors='0.7', zorder=0)
    ax.set_xticks([0, 1, 2, 4, 6, 8, 10])
    ax.set_yticks([-5, 0, 5, 10, 15])
    si.plot.auto_axes_lim(ax, which=("x", "y"), ymin=None, ymargin=0.07)
    ax.set_xlim(0, None)  # fix xlim
    # # # Legend
    leg = ax.legend(loc=(0.04, 0.69), fontsize=5, facecolor='0.95', handlelength=1, borderpad=0.25, labelspacing=0.)
    leg.get_frame().set_linewidth(0.65)
    # # #  Arrow "Higher contrast"
    from matplotlib.patches import FancyArrowPatch
    kwargs = dict(arrowstyle="simple,head_length=4.0,head_width=2.5,tail_width=0.5",
                  capstyle="round", color="0.3", lw=0.5)
    p = FancyArrowPatch(posA=(2.0,-5.0), posB=(7.5,-5.0), **kwargs)
    ax.add_patch(p)
    ax.text(4.50, -4.25, "Increasing contrast", fontdict=dict(fontsize=5, color="0.3", va="baseline", ha="center"))
    



# # #  PANEL G (motion direction repulsion by 2nd speed) # # #
if "G" in PANEL:
    ax = axes['mdr_speed']
    # # # Model
    import pickle
    with open(motion_repulsion["speed_model"], "rb") as f:
        X = pickle.load(f)
        x = X["v_2nd_factor"]
        angle = X["stimulus_angle"] / np.pi * 180  # opening angle
        Y = X["perceived_1st_angle_mean"] / np.pi * 180 - angle[:,None] / 2  # angle/2 --> relative to 1st angle
        Yerr = X["perceived_1st_angle_sem"] / np.pi * 180
        for a,kwargs in motion_repulsion["speed_plot_angles"]:
            i = tuple(angle).index(a)
            y = Y[i]
            ax.plot(x, y, label=r"$\gamma$ = %d°" % int(a), **kwargs)
        # # # Decoration
        ax.hlines(0., x[0], x[-1], lw=0.75, ls=':', colors='0.7', zorder=0)
        ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
        ax.set_yticks([-5, 0, 5, 10, 15])
        si.plot.auto_axes_lim(ax, which=("x", "y"), ymin=None, ymargin=0.07)
        # # # Legend
        leg = ax.legend(loc=(0.60, 0.60), fontsize=5, facecolor='0.95', handlelength=1, borderpad=0.25, labelspacing=0.)
        leg.get_frame().set_linewidth(0.65)



# # # # # # # # # # # # # # # # # # # # 
# # #        T A K E M U R A      # # #  
# # # # # # # # # # # # # # # # # # # # 

def plot_takemura_sketch(key, ax):
    # Circles
    kwargs = dict(ec='0.5', lw=0.5)
    ax.add_patch( pl.Circle(xy=(0.,0.), radius=0.95, fc='0.90', zorder=0., **kwargs) )
    ax.add_patch( pl.Circle(xy=(0.,0.), radius=0.55, fc='0.95', zorder=1., **kwargs) )
    # Inner arrows
    kwargs = dict(x=0., y=0., lw=0., width=0.08, length_includes_head=True, head_width=0.22, head_length=0.22)
    import strinf as si
    Arrow = pl.matplotlib.patches.FancyArrow
    K = 1 + max(takemura["kouter"])
    for k in takemura["kinner"]:
        c = si.colors.get_color(si.colors.CMAP['velo'], k, K)
        dy = 0.35 if "multy_1" in key else 0.
        dx = 0.35 if k==1 else -0.35
        ax.add_patch( Arrow(dx=dx, dy=dy, fc=c, ec=c, **kwargs) )
    # Outer arrows
    c = si.colors.get_color(si.colors.CMAP['velo'], min(takemura["kouter"]), K)
    kwargs = dict(dx=0., lw=0., width=0.08, fc=c, ec=c, length_includes_head=True, head_width=0.22, head_length=0.22)
    for x in (-0.72, 0.72):
        if "up" in key:
            ax.add_patch( Arrow(x=x, y=-0.30, dy=0.60, **kwargs) )
        elif "down" in key:
            ax.add_patch( Arrow(x=x, y=0.30, dy=-0.60, **kwargs) )
        elif "bi" in key:
            ax.add_patch( Arrow(x=x, y=0., dy=+0.40, **kwargs) )
            ax.add_patch( Arrow(x=x, y=0., dy=-0.40, **kwargs) )
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_frame_on(False)


def plot_takemura_inner_angles(inner_angles, bins, ax): 
    ax.set_thetamin(bins.min()*180/np.pi)   # Interestingly, in degree
    ax.set_thetamax(bins.max()*180/np.pi)
    theta = ( bins[1:] + bins[:-1] ) / 2
    for k,angle in zip(takemura["kinner"],inner_angles):
        # Bring angles into the range [-1/2 pi, +3/2 pi]
        angle += np.pi/2
        angle %= 2 * np.pi
        angle -= np.pi/2
        # count
        count,_ = np.histogram(angle, bins=bins)
        print(np.sum(count), count)
        # plot hist
        import strinf as si
        K = 1 + max(takemura["kouter"])
        color = si.colors.get_color(si.colors.CMAP['velo'], k, K)
        kwargs = dict(bottom=0., width=np.diff(bins), zorder=5)
        ax.bar(theta, count, color=color, **kwargs)
    ax.set_xticks(np.linspace(0, 180, 7)/180*np.pi)
    ax.xaxis.set_tick_params(pad=-2.)
    ax.xaxis.set_tick_params(grid_linewidth=0.5)
    ax.set_ylim(0, takemura["count_max"])
    ax.set_yticks([])
    mark = takemura["count_mark"]
    ax.plot([np.pi/2-0.05, np.pi/2+0.05], [mark]*2, '0.2', lw=0.5)
    ax.text(np.pi/2+0.05, mark-1, f"{mark} \nresp.", size=5, ha="right", va="center", linespacing=0.75)


# # #  PANEL H -- L (Takemura: direction repulsion with outer annulus) # # #
takemura_panels = ("H","I","J","K","L")
if np.intersect1d( takemura_panels, PANEL ).size > 0:
    idx_to_plot = [ i for i,l in enumerate(takemura_panels) if l in PANEL ]
    import pickle
    fname = takemura["data"]
    with open(fname, "rb") as f:
        perceived_angles = pickle.load(f)
    si.log.info(f"Takemura data read to file '{fname}'.")
    for idx in idx_to_plot:
        ax_hist   = axes[f"tak_{idx}_hist"] 
        ax_sketch = axes[f"tak_{idx}_sketch"]
        key = takemura["order"][idx]
        inner_angles = perceived_angles[key]
        plot_takemura_sketch(key, ax_sketch)
        plot_takemura_inner_angles(inner_angles, takemura["bins"], ax_hist)
    si.plot.print_panel_label_abcolute_units(ax=axes["tak_0_hist"], \
         label="Model\nperceived velocities", \
         x=6.1, y=7.0, \
         units="cm", \
         fontkwargs={'ha':'center', 'size' : 6, 'weight' : 'normal'})
    



# # #  SAVE AND/OR SHOW  # # #
if SAVE:
    si.plot.savefig(fig, basename="Figure_4", path="./fig/", fmt=(".png", ".pdf"), dpi=600, logger=si.log)
else:
    si.log.warning("Figure not saved!")

if SHOW:
    pl.show()
    
