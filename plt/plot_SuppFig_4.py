"""
Plot Supplemental Figure S4
"""
import numpy as np
import strinf as si

# # #  PARAMETERS  # # # 
ZOOM = 1.0
SAVE = True
SHOW = True

datafname = "./data/data_yang2021_experiment_1/yang_predicted_choices_by_motion_structure_inference_algorithm_CV.pkl"
fil = "adiab_0.040"
Struct = ('I','G','C','H')

nP = 12

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
for pid in range(nP):
    xi, yi = pid % 6, pid//6
    rect=(1.50 + 2.75*xi, 0.50 + 6.25*yi, 2.5, 2.5)
    axeslayout[f"{pid+1:02d}_human"] = dict(rect=rect, xticks=[], yticks=[], frame_on=False)
    rect=(1.50 + 2.75*xi, 3.25 + 6.25*yi, 2.5, 2.5)
    axeslayout[f"{pid+1:02d}_model"] = dict(rect=rect, xticks=[], yticks=[], frame_on=False)

fig, axes = si.plot.init_figure_absolute_units(figsize=(18.,13.), axes=axeslayout, units="cm")
fig.canvas.manager.set_window_title('Supplemental Figure S4')

# Titles
for pid in range(nP):
    pid += 1
    ax = axes[f"{pid:02d}_human"]
    ax.set_title(f"# {pid}", fontdict=dict(size=8, weight='bold'), pad=2.)

# Text
kwargs = dict(fontsize=12, fontweight='normal', rotation='vertical', ha='center', va='center', clip_on=False)
axes['01_human'].text(-1.5, 1.5, "Human", **kwargs)
axes['01_model'].text(-1.5, 1.5, "Model", **kwargs)
axes['07_human'].text(-1.5, 1.5, "Human", **kwargs)
axes['07_model'].text(-1.5, 1.5, "Model", **kwargs)

axes['01_model'].text(-2.40, 4.5, "True structure", **kwargs)
kwargs['rotation'] = 'horizontal'
axes['09_model'].text( 3.70, 4.60, "Human choice / model prediction", **kwargs)



# # #  LOAD # # #  

import pickle
with open(datafname, "rb") as f:
    data = pickle.load(f)[fil]

Human = dict()
Model = dict()
for pid in range(nP):
    pid += 1 
    ground = data[pid]['ground_truth']
    choice = data[pid]['choice']
    # Human confusion matrix
    M = np.zeros((4,4))
    for g,c in zip(ground, choice):
        M[Struct.index(g), Struct.index(c)] += 1
    M /= M.sum(1)[:,None]
    Human[pid] = M
    # Model confusion matrix
    pred = data[pid]['P_pred']
    M = np.zeros((4,4))
    for g,p in zip(ground, pred):
        M[Struct.index(g)] += p
    M /= M.sum(1)[:,None]
    Model[pid] = M
    
    

# # #  PLOT # # #  

# Human choices
for pid in range(nP):
    pid += 1 
    ax = axes[f"{pid:02d}_human"]
    kwargs = dict(vmin=0, vmax=1, cmap=pl.cm.Blues)
    M = Human[pid]
    ax.imshow(M, **kwargs)
    for y,row in enumerate(M):
        for x,p in enumerate(row):
            c = "w" if p > 0.5 else 'k'
            t = "1.00" if p==1 else f"{p:.2f}"[1:]
            ax.text(x, y, t, weight="bold", ha='center', va='center', color=c)
    # Labels
    if pid in(1,7):
        ax.set_yticks([0,1,2,3])
        ax.set_yticklabels(Struct, fontdict=dict(fontsize=8, ha='center'))
        ax.tick_params(length=0., pad=6.)
        
    
# Model predictions
for pid in range(nP):
    pid += 1 
    ax = axes[f"{pid:02d}_model"]
    kwargs = dict(vmin=0, vmax=1, cmap=pl.cm.Blues)
    M = Model[pid]
    ax.imshow(M, **kwargs)
    for y,row in enumerate(M):
        for x,p in enumerate(row):
            c = "w" if p > 0.5 else 'k'
            t = "1.00" if p==1 else f"{p:.2f}"[1:]
            ax.text(x, y, t, weight="bold", ha='center', va='center', color=c)
    # Labels
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(Struct, fontdict=dict(fontsize=8, va='center'))
    ax.tick_params(length=0., pad=6.)
    if pid in(1,7):
        ax.set_yticks([0,1,2,3])
        ax.set_yticklabels(Struct, fontdict=dict(fontsize=8, ha='center'))
        ax.tick_params(length=0., pad=6.)
    

# # #  SAVE AND/OR SHOW  # # #
if SAVE:
    si.plot.savefig(fig, basename="SuppFig_S4", path="./fig/", fmt=(".png", ".pdf"), dpi=600, logger=si.log)
else:
    si.log.warning("Figure not saved!")

if SHOW:
    pl.show()
