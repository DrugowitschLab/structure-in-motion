"""
Plot Supplemental Figure S6
"""
import numpy as np
import strinf as si

# # #  PARAMETERS  # # # 
ZOOM = 1.5
SAVE = True
SHOW = True

fil = "adiab"

low_noise = {
    "DSL" : "2021-08-26-12-09-10-696387_110_lorenceau_1996_low_noise",
    "sTminmax" : (0., 8.),
    "suffix" : "low",
}

high_noise = {
    "DSL" : "2021-08-26-12-07-25-644304_110_lorenceau_1996_high_noise",
    "sTminmax" : (0., 8.),
    "suffix" : "high",
}


# Colors of all lines
colors = [si.colors.SINGLE["self"]]
colors += [ si.colors.get_color(si.colors.CMAP['glo'], 0, 1) ]
colors += list(si.colors.get_colors(si.colors.CMAP['clu'], 2))
colors += list(si.colors.get_colors(si.colors.CMAP['ind'], 20))


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
    "lam_low" : dict(rect=(1.00, 0.25, 16.5, 3.25), xlabel="Time [s]", ylabel="Motion strengths λ"),
    "C_low"   : dict(rect=(17.45-24*0.10, 0.30, 24*0.10, 21*0.10), xticks=[], yticks=[]),
    "x_low" :   dict(rect=(1.00, 4.50,  7.5, 3.25), xlabel="Time [s]", ylabel="Sources x-direction, $s_x(t)$"),
    "y_low" :   dict(rect=(10.0, 4.50,  7.5, 3.25), xlabel="Time [s]", ylabel="Sources y-direction, $s_y(t)$"),
    "lam_high" : dict(rect=(1.00, 9.00, 16.5, 3.25), xlabel="Time [s]", ylabel="Motion strengths λ"),
    "C_high"   : dict(rect=(17.45-24*0.10, 9.05, 24*0.10, 21*0.10), xticks=[], yticks=[]),
    "x_high" :   dict(rect=(1.00, 13.25,  7.5, 3.25), xlabel="Time [s]", ylabel="Sources x-direction, $s_x(t)$"),
    "y_high" :   dict(rect=(10.0, 13.25,  7.5, 3.25), xlabel="Time [s]", ylabel="Sources y-direction, $s_y(t)$"),
    }

fig, axes = si.plot.init_figure_absolute_units(figsize=(18.,17.25), axes=axeslayout, units="cm")
fig.canvas.manager.set_window_title('Supplemental Figure S6')


# # #  PANEL LABELS  # # #

labels = {  # (label, x-pos, y-pos in absolute coords)
    "lam_low" : ("a", 0.15, 0.32),
    "x_low" :   ("b", 0.15, 4.57),
    "y_low" :   ("c", 9.15, 4.57),
    "lam_high" : ("d", 0.15, 9.07),
    "x_high" :   ("e", 0.15, 13.32),
    "y_high" :   ("f", 9.15, 13.32),
    }

for axname, (label, x, y) in labels.items():
    ax = axes[axname]
    si.plot.print_panel_label_abcolute_units(ax, label, x, y, units="cm", fontkwargs=dict(fontsize=6.))


# # #  PLOT  # # #  

for D in (low_noise, high_noise):
    ds, cfg = si.load_dataset(D["DSL"], R=0)
    # # #  Plot Lambda  # # #
    ax = axes[f"lam_{D['suffix']}"]
    axC = axes[f"C_{D['suffix']}"]
    Lam = { str(fil) : ds.Lam_inf.loc[fil] }
    C = np.array(cfg['fil']['default_params']['C'])
    ret = si.plot_inferred_lambda(ds.t, Lam, C, colors=colors, ax=ax, axC=axC)
    # Remove legend
    ret['leg'].set_visible(False)
    # Fix linestyle
    for l in ax.get_children()[:len(colors)]:
        l.set_ls('-')
    # Highligh region in b and c
    x0, x1 = D["sTminmax"]
    y0, y1 = ax.get_ylim()
    patch = pl.matplotlib.patches.Rectangle((x0,y0), x1-x0, y1-y0, lw=0., ec='0.9', fc='0.9', zorder=-1)
    ax.add_patch(patch)

    # # #  Plot sources  # # #  
    t0, t1 = D["sTminmax"]
    tidx =  (ds.t >= t0 ) * (ds.t <= t1)
    t = ds.t[tidx]
    for pre,idx in zip(["x_", "y_"], [0, 1]):
        ax = axes[pre + D['suffix']]
        S = { fil : ds.S_inf.loc[fil][tidx,idx] }
        ret = si.plot_inferred_source(t=t, S=S, C=C, colors=colors, ax=ax)
        ax.set_ylabel(axeslayout[pre + D['suffix']]['ylabel'])
        # Remove legend
        ret['leg'].set_visible(False)
        # Fix linestyle
        for l in ax.get_children()[:len(colors)]:
            l.set_ls('-')


# # #  SAVE AND/OR SHOW  # # #
if SAVE:
    si.plot.savefig(fig, basename="SuppFig_S6", path="./fig/", fmt=(".png", ".pdf"), dpi=600, logger=si.log)
else:
    si.log.warning("Figure not saved!")

if SHOW:
    pl.show()
