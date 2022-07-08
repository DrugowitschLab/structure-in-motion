"""
Plot Supplemental Figure S1
"""
import numpy as np
import strinf as si

# # #  PARAMETERS  # # # 
ZOOM = 1.0
SAVE = True
SHOW = True

filnames = {"exact" : "Online EM", "adiab" : "Approx. algorithm"}

default = {
    "DSL" : "2021-10-03-15-54-20-561901_001_nested_model_recovery",
    "sTminmax" : (13.5, 16.5),
}

ML = {    # This means: tau_lam *= 10; nu = -2/D = -2
    "DSL" : "2021-10-03-15-57-25-927721_002_nested_model_recovery_ML",
}

lamList = {
    "DSL" : "2021-10-03-17-50-46-057828_003_nested_model_recovery_change_lam",
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
    "lam_def" : dict(rect=(1.00, 0.25, 16.5, 3.25), xlabel="Time [s]", ylabel="Motion strengths λ"),
    "C_def"   : dict(rect=(1.05, 0.30, 11*0.135, 8*0.135), xticks=[], yticks=[]),
    "x_def" :   dict(rect=(1.00, 4.25,  7.5, 3.25), xlabel="Time [s]", ylabel="Sources x-direction, $s_x(t)$"),
    "y_def" :   dict(rect=(10.0, 4.25,  7.5, 3.25), xlabel="Time [s]", ylabel="Sources y-direction, $s_y(t)$"),
    "lam_ML":   dict(rect=(1.00, 8.50, 16.5, 3.25), xlabel="Time [s]", ylabel="Motion strengths λ"),
    "lam_list":   dict(rect=(1.00,12.75, 16.5, 3.25), xlabel="Time [s]", ylabel="Motion strengths λ"),
    }

fig, axes = si.plot.init_figure_absolute_units(figsize=(18.,16.75), axes=axeslayout, units="cm")
fig.canvas.manager.set_window_title('Supplemental Figure 1')

# # #  PANEL LABELS  # # #

labels = {  # (label, x-pos, y-pos in absolute coords)
    "lam_def" : ("a", 0.15, 0.32),
    "x_def" :   ("b", 0.15, 4.32),
    "y_def" :   ("c", 9.15, 4.32),
    "lam_ML" :  ("d", 0.15, 8.57),
    "lam_list": ("e", 0.15,12.83),
    }

for axname, (label, x, y) in labels.items():
    ax = axes[axname]
    si.plot.print_panel_label_abcolute_units(ax, label, x, y, units="cm", fontkwargs=dict(fontsize=6.))


# # #  PLOT FOR DEFAULT PARAMETERS  # # #  

ax = axes["lam_def"]
axC = axes["C_def"]
ds, cfg = si.load_dataset(default["DSL"], R=0)

# # #  Plot Lambda  # # #
lam_star = cfg["wld"].get("lamList", cfg["wld"]["lam"])
Lam = { str(fil.data) : ds.Lam_inf.loc[fil] for fil in ds.F}
C = np.array(cfg['fil']['default_params']['C'])
ret = si.plot_inferred_lambda(ds.t, Lam, C, lam_star=lam_star, ax=ax, axC=axC)

# fix labels
for text in ret['leg'].get_texts():
    if text.get_text().lower() in filnames:
        text.set_text(filnames[text.get_text().lower()])

# Highligh region in b and c
x0, x1 = default["sTminmax"]
y0, y1 = ax.get_ylim()
patch = pl.matplotlib.patches.Rectangle((x0,y0), x1-x0, y1-y0, lw=0., ec='0.9', fc='0.9', zorder=-1)
ax.add_patch(patch)


# # #  Plot sources  # # #  
t0, t1 = default["sTminmax"]
tidx =  (ds.t >= t0 ) * (ds.t <= t1)
t = ds.t[tidx]

for l,idx in zip(["x_def", "y_def"], [0, 1]):
    ax = axes[l]
    S = { str(fil.data) : ds.S_inf.loc[fil][tidx,idx] for fil in ds.F}
    S_star = ds.S_wld[0][tidx,idx]
    ret = si.plot_inferred_source(t=t, S=S, C=C, S_star=S_star, ax=ax) # S_star)
    ret['leg'].set_visible(False)
    ax.set_ylabel(axeslayout[l]['ylabel'])



# # #  PLOT FOR ML PARAMETERS  # # #  

ax = axes["lam_ML"]
ds, cfg = si.load_dataset(ML["DSL"], R=0)

# # #  Plot Lambda  # # #
lam_star = cfg["wld"].get("lamList", cfg["wld"]["lam"])
Lam = { str(fil.data) : ds.Lam_inf.loc[fil] for fil in ds.F}
C = np.array(cfg['fil']['default_params']['C'])
ret = si.plot_inferred_lambda(ds.t, Lam, C, lam_star=lam_star, ax=ax, axC=axC)

# fix labels
for text in ret['leg'].get_texts():
    if text.get_text().lower() in filnames:
        text.set_text(filnames[text.get_text().lower()])


# # #  PLOT FOR CHANING LAMBDA (default parameters)  # # #  

ax = axes["lam_list"]
ds, cfg = si.load_dataset(lamList["DSL"], R=0)

# # #  Plot Lambda  # # #
lam_star = cfg["wld"].get("lamList")
Lam = { str(fil.data) : ds.Lam_inf.loc[fil] for fil in ds.F}
C = np.array(cfg['fil']['default_params']['C'])
ret = si.plot_inferred_lambda(ds.t, Lam, C, lam_star=lam_star, ax=ax, axC=axC)

# fix labels
for text in ret['leg'].get_texts():
    if text.get_text().lower() in filnames:
        text.set_text(filnames[text.get_text().lower()])

# # #  SAVE AND/OR SHOW  # # #
if SAVE:
    si.plot.savefig(fig, basename="SuppFig_S1", path="./fig/", fmt=(".png", ".pdf"), dpi=600, logger=si.log)
else:
    si.log.warning("Figure not saved!")

if SHOW:
    pl.show()
