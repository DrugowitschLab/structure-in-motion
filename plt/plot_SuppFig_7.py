"""
Plot Supplemental Figure S7
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
DSL = "2022-05-31-17-04-42-775422_120_SFM_C_control" 

fil = "adiab"                                      # used filter
rep = 0
Mplot = (0, 1, 2, 3, 4, 5, 18)                     # Show only two of the individual motions
Mrot = [1]                                            # index of rotational sources
tmax = 42

Cmrot = array([-0.9       , -1.26885775, -1.44568323, -1.5       , -1.44568323,
       -1.26885775, -0.9       ,  0.9       ,  1.26885775,  1.44568323,
        1.5       ,  1.44568323,  1.26885775,  0.9       , -0.        ])



# Colors of all lines
colors = [si.colors.SINGLE["self"]]
colors += list(si.colors.get_colors(si.colors.CMAP['glo'], 2))
colors += list(si.colors.get_colors(si.colors.CMAP['clu'], 2))
colors += list(si.colors.get_colors(si.colors.CMAP['ind'], 14))


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
    "C" :     dict(rect=( 1.50, 0.25, 19*0.2, 15*0.2), xticks=[], yticks=[]),
    "cb":     dict(rect=( 2.75, 3.45, 2.55, 0.20), xlabel="C$_{km}$ for rotational source"),
    "lambda": dict(rect=( 7.0, 0.25, 4.5, 3.25), xlabel="Time [s]", ylabel="Strengths, $λ(t)$"),
    "source": dict(rect=(13.0, 0.25, 4.5, 3.25), xlabel="Time [s]", ylabel="Sources, x-dir. & rotation, $s(t)$"),
    }

fig, axes = si.plot.init_figure_absolute_units(figsize=(18.,4.50), axes=axeslayout, units="cm")
fig.canvas.manager.set_window_title('Supplemental Figure SfM more components')

# # #  PANEL LABELS  # # #

labels = {  # (label, x-pos, y-pos in absolute coords)
    "C": ("a", 0.25, 0.32),
    "lambda": ("b", 6.0, 0.32),
    "source": ("c", 12.0, 0.32),
    }

for axname, (label, x, y) in labels.items():
    ax = axes[axname]
    si.plot.print_panel_label_abcolute_units(ax, label, x, y, units="cm", fontkwargs=dict(fontsize=6.))


# # #  AUX FUNC  # # #

def boxfilter(y, N):
    """
    Like a convolution with [1/N]*N, but with "shorter" kernel at the beginning to avoid boundary effects.
    """
    n = np.clip(np.arange(len(y)), 0, N-1) + 1
    return np.array([ y[i+1-ni:i+1].sum() / ni for i,ni in zip(range(len(y)), n) ])


    
# # #  PLOT  # # #  

ds, cfg = si.load_dataset(DSL, datadir=datadir, R=rep, F=fil)
t = ds.t
Lam = ds.Lam_inf
S = ds.S_inf

# C matrix
ax = axes['C']
viscolors = np.vstack(( si.colors.get_colors(si.colors.CMAP['velo'], 14),
                       [matplotlib.colors.to_rgba(si.colors.SINGLE['vestibular'])]))

si.plotting.imshow_C_matrix(
    C = np.array(cfg['fil']['default_params']['C']),
    ax = ax,
    colors=colors,
    addVisibleCircles=True,
    viscolors=viscolors
)

# Colorbar for rotational source
mappable = ax.imshow(Cmrot.reshape(15,1), extent=(0.5,1.5,14.5,-0.5), cmap=pl.cm.RdBu_r)
cb = pl.colorbar(mappable, cax=axes["cb"], orientation="horizontal")
cb.set_label("C$_{km}$ for rotational source", labelpad=0)

# Annotations
ax.text(-2, 6.45, "|--- Obs. velocities ---|",rotation="vertical", ha="right", va="center")
ax.text(-2, 14, "Vest. →", ha="right", va="center")
ax.text(1, 14.75, "↑\nRotational\nsource", ha="center", va="top", color=colors[1])

# Structure
ax = axes["lambda"]
for m in reversed(Mplot):
    kwargs = dict(color=colors[m], ls="-", lw=0.5)
    lm = Lam[:,m]
    ax.plot(t, lm, **kwargs)
ax.set_xlim(t[0], t[-1] if tmax is None else tmax)

ax = axes["source"]
for m in reversed(Mplot):
    d = 1 if m in Mrot else 0 
    kwargs = dict(color=colors[m], ls="-", lw=0.5)
    sm = S[:,d,m]
    if m == 0:              # In the simulation code, we have no minus sign for self-motion.
        sm = -sm            # Thus, we add the minus sign for plotting (both forumations are perfectly equivalent).
    ax.plot(t, sm, **kwargs)
ax.set_xlim(t[0], t[-1] if tmax is None else tmax)


# # #  SAVE AND/OR SHOW  # # #
if SAVE:
    si.plot.savefig(fig, basename="SuppFig_S7", path="./fig/", fmt=(".png", ".pdf"), dpi=600, logger=si.log)
else:
    si.log.warning("Figure not saved!")

if SHOW:
    pl.show()
