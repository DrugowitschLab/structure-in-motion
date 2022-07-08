"""
Plot Supplemental Figure S3
"""
import numpy as np
import strinf as si

# # #  PARAMETERS  # # # 
ZOOM = 1.0
SAVE = True
SHOW = True

fil = "adiab_0.040"
# One trial of each type for participant 1
DSL = "2021-08-18-14-23-15-071947_040_Yang_2021_01"
trials = (6, 87, 12, 56, 5, 42, 0, 104)    # Examples where humans chose correctly (left) or incorrectly (right)

# Preditions
pid = 1
predfname = "./data/data_yang2021_experiment_1/yang_predicted_choices_by_motion_structure_inference_algorithm_CV.pkl"


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
axeslayout = {}

for i in range(len(trials)):
    xi, yi = i % 2, i // 2
    l,t,w,h = 1.0 + 9.0 * xi, 0.5 + 4.50 * yi, 7.75, 3.25
    axeslayout[f"{i:d}_lam"] = dict(rect=(l, t, w, h), xlabel="Time [s]", ylabel="Motion strengths λ")
    axeslayout[f"{i:d}_C"] = dict(rect=(l + 0.05, t+0.05, 7*0.17, 3*0.17), xticks=[], yticks=[])

fig, axes = si.plot.init_figure_absolute_units(figsize=(18., (yi+1) * 4.50), axes=axeslayout, units="cm")
fig.canvas.manager.set_window_title('Supplemental Figure 3')

# # #  LOAD DATA   # # #
ds, cfg = si.load_dataset(DSL, F=fil)

import pickle
with open(predfname, "rb") as f:
    preddata = pickle.load(f)[fil]

# # #  PLOT  # # #  


for i,r in enumerate(trials):
    ax = axes[f"{i:d}_lam"]
    axC = axes[f"{i:d}_C"]
    # # #  Plot Lambda  # # #
    Lam = {"Approx." : ds.Lam_inf.loc[r]}
    C = np.array(cfg['fil']['default_params']['C'])
    ret = si.plot_inferred_lambda(ds.t, Lam, C, lam_star=None, ax=ax, axC=axC)
    # Title
    ps = ", ".join([ f"{p:.2f}" for p in preddata[pid]['P_pred'][r] ])
    ax.set_title(f"Stimulus: {cfg['label']['ground_truth'][r]},  \
Human choice: {cfg['label']['choice'][r]},  \
Model pred. prob.: (I,G,C,H) = {ps}", pad=3)
    # Remove legend
    ret['leg'].set_visible(False)
    # Fix linestyle
    for l in ax.get_children()[:7]:
        l.set_ls('-')

# # #  SAVE AND/OR SHOW  # # #
if SAVE:
    si.plot.savefig(fig, basename="SuppFig_S3", path="./fig/", fmt=(".png", ".pdf"), dpi=600, logger=si.log)
else:
    si.log.warning("Figure not saved!")

if SHOW:
    pl.show()
