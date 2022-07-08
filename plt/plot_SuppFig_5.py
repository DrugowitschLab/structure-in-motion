"""
Plot Supplemental Figure S5
"""
import numpy as np
import pylab as pl
import strinf as si


# # #  PARAMETERS  # # #
ZOOM = 1.5
SAVE = True                   # Save figure file
SHOW = True

# The trials from all conditons
datadir = "./data/"
DSL = {"multy_0_surround_bi" : "2022-05-12-16-03-09-435421_107_direction_repulsion_Takemura_inner_y_0_outer_bi",
       "multy_0_surround_down" : "2022-05-12-16-03-25-555141_107_direction_repulsion_Takemura_inner_y_0_outer_down",
       "multy_1_surround_down" : "2022-05-12-16-03-46-053483_107_direction_repulsion_Takemura_inner_y_1_outer_down",
       "multy_1_surround_bi" : "2022-05-12-16-45-40-330033_107_direction_repulsion_Takemura_inner_y_1_outer_bi",
       "multy_1_surround_up" : "2022-05-12-16-46-03-495979_107_direction_repulsion_Takemura_inner_y_1_outer_up",
       }

order = ("multy_0_surround_bi", "multy_0_surround_down", "multy_1_surround_down", "multy_1_surround_bi", "multy_1_surround_up")
fil = "adiab"                                      # used filter
Rep = [0]*5                                        # which trial to use for each condition
Mplot = (0,1,2,4,5)                                # Which m's to show (for clarity)
smoothframes = 30                                   # Smooth curves for visual clarity (60 fps)

# Colors of all lines
colors = [si.colors.SINGLE["self"]]
colors += [ si.colors.get_color(si.colors.CMAP['glo'], 0, 1) ]
colors += list(si.colors.get_colors(si.colors.CMAP['clu'], 2))
colors += list(si.colors.get_colors(si.colors.CMAP['ind'], 4))


tavg = 0.                                         # Average percept over the last tavg seconds
mself = 0                                          # Index of the self-motion component 
kinner = 0,1                                       # Indices of the inner dots (the ones of interest)


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
    "tree" :    dict(rect=(1.00, 0.25, 3.50, 3.50),  xticks=[], yticks=[]),
    "source_multy_0_surround_bi"  : dict(rect=(10.0, 0.25, 7.5, 3.25), xlabel="Time [s]", ylabel="Sources, $s(t)$"),
    "source_multy_0_surround_down": dict(rect=(01.0, 4.75, 7.5, 3.25), xlabel="Time [s]", ylabel="Sources, $s(t)$"),
    "source_multy_1_surround_down": dict(rect=(10.0, 4.75, 7.5, 3.25), xlabel="Time [s]", ylabel="Sources, $s(t)$"),
    "source_multy_1_surround_bi"  : dict(rect=(01.0, 9.25, 7.5, 3.25), xlabel="Time [s]", ylabel="Sources, $s(t)$"),
    "source_multy_1_surround_up"  : dict(rect=(10.0, 9.25, 7.5, 3.25), xlabel="Time [s]", ylabel="Sources, $s(t)$"),
    # sketches
    "sketch_multy_0_surround_bi"  : dict(rect=(10.05, 0.30, 1.25, 1.25), xticks=[], yticks=[]),
    "sketch_multy_0_surround_down": dict(rect=(01.05, 4.80, 1.25, 1.25), xticks=[], yticks=[]),
    "sketch_multy_1_surround_down": dict(rect=(10.05, 4.80, 1.25, 1.25), xticks=[], yticks=[]),
    "sketch_multy_1_surround_bi"  : dict(rect=(01.05, 9.30, 1.25, 1.25), xticks=[], yticks=[]),
    "sketch_multy_1_surround_up"  : dict(rect=(10.05, 9.30, 1.25, 1.25), xticks=[], yticks=[]),
    }

fig, axes = si.plot.init_figure_absolute_units(figsize=(18.,13.25), axes=axeslayout, units="cm")
fig.canvas.manager.set_window_title('Supplemental Figure Takemura')

# # #  PANEL LABELS  # # #

labels = {  # (label, x-pos, y-pos in absolute coords)
    "tree" : ("a", 0.15, 0.32),
    "source_multy_0_surround_bi"  : ("b", 9.15, 0.32),
    "source_multy_0_surround_down": ("c", 0.15, 4.82),
    "source_multy_1_surround_down": ("d", 9.15, 4.82),
    "source_multy_1_surround_bi"  : ("e", 0.15, 9.32),
    "source_multy_1_surround_up"  : ("f", 9.15, 9.32),
    }

for axname, (label, x, y) in labels.items():
    ax = axes[axname]
    si.plot.print_panel_label_abcolute_units(ax, label, x, y, units="cm", fontkwargs=dict(fontsize=6.))


# # #  AUX FUNC  # # #

def plot_setup_sketch(key, ax):
    # Circles
    kwargs = dict(ec='0.5', lw=0.5)
    ax.add_patch( pl.Circle(xy=(0.,0.), radius=0.95, fc='0.90', zorder=0., **kwargs) )
    ax.add_patch( pl.Circle(xy=(0.,0.), radius=0.55, fc='1.00', zorder=1., **kwargs) )
    # Inner arrows
    kwargs = dict(x=0., y=0., lw=0., width=0.08, length_includes_head=True, head_width=0.2, head_length=0.2)
    import strinf as si
    Arrow = pl.matplotlib.patches.FancyArrow
    for k in kinner:
        c = si.colors.get_color(si.colors.CMAP['velo'], k, 4)
        dy = 0.35 if "multy_1" in key else 0.
        dx = 0.35 if k==1 else -0.35
        ax.add_patch( Arrow(dx=dx, dy=dy, fc=c, ec=c, **kwargs) )
    # Outer arrows
    c = si.colors.get_color(si.colors.CMAP['velo'], 3, 4)
    kwargs = dict(dx=0., lw=0., width=0.08, fc=c, ec=c, length_includes_head=True, head_width=0.2, head_length=0.2)
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



def boxfilter(y, N):
    """
    Like a convolution with [1/N]*N, but with "shorter" kernel at the beginning to avoid boundary effects.
    """
    n = np.clip(np.arange(len(y)), 0, N-1) + 1
    return np.array([ y[i+1-ni:i+1].sum() / ni for i,ni in zip(range(len(y)), n) ])


    
# # #  PLOT  # # #  


# # #  PLOT THE TREE  # # #
ax = axes["tree"]
ax.imshow(pl.imread("./panel/sketch_tree_takemura.png"))
ax.set_frame_on(False)

# # #  PLOT THE SKETCHES  # # #
for key in order:
    ax = axes["sketch_" + key]
    plot_setup_sketch(key, ax)




# # #  PLOT THE SOURCES  # # #
# i = 0
# key = "multy_0_surround_bi"

for i,key in enumerate(order):
    ax = axes["source_" + key]
    ds, cfg = si.load_dataset(DSL[key], datadir=datadir, R=Rep[0], F=fil)
    t = ds.t
    for m in reversed(Mplot):
        for d in ds.d:
            kwargs = dict(color=colors[m], ls = "-" if d==0 else (0, (5,5)), lw=0.5)
            sm = ds.S_inf[:,d,m]
            if m == 0:              # In the simulation code, we have no minu sign for self-motion.
                sm = -sm            # Thus, we add the minus sign for plotting (both forumations are perfectly equivalent).
            ax.plot(t, boxfilter(sm, smoothframes), **kwargs)
    ax.set_xlim(t[0], t[-1])
    ymax = 1.10 * max(ax.dataLim.ymax, abs(ax.dataLim.ymin))
    ax.set_ylim(-ymax, ymax)
    ax.set_yticks([-0.5, 0.0, 0.5])
    ax.set_ylabel("Sources, $s(t)$", labelpad=-2)
    



# # #  SAVE AND/OR SHOW  # # #
if SAVE:
    si.plot.savefig(fig, basename="SuppFig_S5", path="./fig/", fmt=(".png", ".pdf"), dpi=600, logger=si.log)
else:
    si.log.warning("Figure not saved!")

if SHOW:
    pl.show()
