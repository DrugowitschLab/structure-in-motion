"""
Plot Supplemental Figure S2
"""
import numpy as np
import strinf as si

# # #  PARAMETERS  # # # 
ZOOM = 1.0
SAVE = True
SHOW = True

filnames = {"exact" : "Online EM", "adiab" : "Approx. algorithm"}

DSL = "2021-10-06-16-59-47-161227_021_3_dots_Johansson_1973_sparsity"

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
    "lam" : dict(rect=(1.00, 0.25, 7.75, 3.25), xlabel="Time [s]", ylabel="Motion strengths λ"),
    "C"   : dict(rect=(1.05, 0.30, 5*0.17, 3*0.17), xticks=[], yticks=[]),
    }

fig, axes = si.plot.init_figure_absolute_units(figsize=(9.,4.25), axes=axeslayout, units="cm")
fig.canvas.manager.set_window_title('Supplemental Figure 1')


# # #  PLOT FOR DEFAULT PARAMETERS  # # #  

ax = axes["lam"]
axC = axes["C"]
ds, cfg = si.load_dataset(DSL, R=0)

# # #  Plot Lambda  # # #
Lam = { str(fil.data) : ds.Lam_inf.loc[fil] for fil in ds.F}
C = np.array(cfg['fil']['default_params']['C'])
ret = si.plot_inferred_lambda(ds.t, Lam, C, lam_star=None, ax=ax, axC=axC)

# Highlight the relevant lines
ax.get_children()[5].set_zorder(10)
ax.get_children()[5].set_lw(0.6)
ax.get_children()[6].set_zorder(10)
ax.get_children()[6].set_lw(0.6)

# fix labels
for text in ret['leg'].get_texts():
    if text.get_text().lower() in filnames:
        text.set_text(filnames[text.get_text().lower()])

# Fix legend location
ret["leg"].set_bbox_to_anchor((1.0, 0.40))

# # #  SAVE AND/OR SHOW  # # #
if SAVE:
    si.plot.savefig(fig, basename="SuppFig_S2", path="./fig/", fmt=(".png", ".pdf"), dpi=600, logger=si.log)
else:
    si.log.warning("Figure not saved!")

if SHOW:
    pl.show()
