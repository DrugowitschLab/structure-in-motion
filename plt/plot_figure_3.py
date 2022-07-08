"""
Plot Figure 3
"""
import numpy as np
import strinf as si

# # #  PARAMETERS  # # # 
ZOOM = 1.5
SAVE = False
SHOW = True
PANEL = ("C", "E", "F")   # The sketches in A, B, D are always plotted.

# data analysis obtained from '../ana/Yang_regress_and_choice.py'
fnameData = "./data/data_yang2021_experiment_1/yang_predicted_choices_by_motion_structure_inference_algorithm_CV.pkl"
fil = 'adiab_0.040'     # use this filter
# Log-likelihood from Yang et al. (2021) before subtracting chance level
yangLL = "./data/data_yang2021_experiment_1/yang_logL_exp1_4paramModel.txt"
llchance = np.log(1/4) * 200



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
    "stim"    :  dict(rect=(0.50, 0.25, 3.25, 3.25), xticks=[], yticks=[]),
    "structs" :  dict(rect=(4.50, 0.175, 3.25, 3.50), xticks=[], yticks=[]),
    "cmhuman" :  dict(rect=(1.00, 4.00, 2.50, 2.50), xlabel="Human choice", ylabel="True structure"),
    "models"  :  dict(rect=(4.00, 4.00, 4.25, 3.50), xticks=[], yticks=[]),
    "cmmodel" :  dict(rect=(1.00, 7.75, 2.50, 2.50), xlabel="Model prediction", ylabel="True structure"),
    "loglike" :  dict(rect=(4.75, 8.00, 3.25, 2.25), xlabel="Participant", ylabel="Log-likelihood vs. chance"),
    }

fig, axes = si.plot.init_figure_absolute_units(figsize=(8.50,11.0), axes=axeslayout, units="cm")
fig.canvas.manager.set_window_title('Figure 3')

# # #  PANEL LABELS  # # #
labels = {  # (label, x-pos, y-pos in absolute coords)
    "stim"    :  ("a", 0.15, 0.30),
    "structs" :  ("b", 4.20, 0.30),
    "cmhuman" :  ("c", 0.15, 4.05),
    "models"  :  ("d", 3.80, 4.05),
    "cmmodel" :  ("e", 0.15, 7.80),
    "loglike" :  ("f", 3.80, 7.80),
    }

for axname, (label, x, y) in labels.items():
    ax = axes[axname]
    si.plot.print_panel_label_abcolute_units(ax, label, x, y, units="cm", fontkwargs=dict(fontsize=6.))

# # #   LOAD DATA  # # #
if ("C" in PANEL) or ("E" in PANEL) or ("F" in PANEL):
    import pickle
    with open(fnameData, "rb") as f:
        data = pickle.load(f)[fil]

Pid = np.array([k for k in data.keys()])
Pid.sort()


# # #  AUXILIARY FUNCTIONS  # # # 
def plot_confusion_matrix(M, ax):
    labels = "I", "G", "C", "H"
    assert M.shape == (4,4) 
    kwargs = dict(vmin=0, vmax=1, cmap=pl.cm.Blues)
    ax.imshow(M, **kwargs)
    for y,row in enumerate(M):
        for x,p in enumerate(row):
            c = "w" if p > 0.5 else 'k'
            ax.text(x, y, f"{p:.2f}"[1:], weight="bold", ha='center', va='center', color=c, fontsize=6.)
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(labels)
    ax.set_yticks([0,1,2,3])
    ax.set_yticklabels(labels)


# # #  PLOT THE SKETCHES  # # #
ax = axes["stim"]
ax.imshow(pl.imread("./panel/sketch_yang_stimulus.png"))
ax.set_frame_on(False)

ax = axes["structs"]
ax.imshow(pl.imread("./panel/sketch_yang_structures.png"))
ax.set_frame_on(False)

ax = axes["models"]
ax.imshow(pl.imread("./panel/sketch_yang_models.png"))
ax.set_frame_on(False)



# # #  PANEL C (Human confusion matrix)  # # #
if "C" in PANEL:
    ax = axes["cmhuman"]
    # Concatenate all participants' data
    Gt = np.concatenate([ data[pid]['ground_truth'] for pid in Pid ])
    Ch =  np.concatenate([ data[pid]['choice'] for pid in Pid ], axis=0)
    # Calculate the confusion matrix
    M = np.zeros((4,4))
    labels = "I", "G", "C", "H"
    for i,gt in enumerate(labels):
        idx = Gt == gt
        P = np.zeros(4)
        for ch in Ch[idx]:
            P[labels.index(ch)] += 1.
        P /= P.sum()
        M[i] = P
    # plot it (okay, checked that identical to Yang's evaluation)
    plot_confusion_matrix(M, ax)


# # #  PANEL E (Model confusion matrix)  # # #
if "C" in PANEL:
    ax = axes["cmmodel"]
    # Concatenate all participants' data
    Gt = np.concatenate([ data[pid]['ground_truth'] for pid in Pid ])
    P_pred =  np.concatenate([ data[pid]['P_pred'] for pid in Pid ], axis=0)
    # Calculate the confusion matrix
    M = np.zeros((4,4))
    labels = "I", "G", "C", "H"
    for i,gt in enumerate(labels):
        idx = Gt == gt
        M[i] = P_pred[idx].mean(0)
    # plot it
    plot_confusion_matrix(M, ax)


# # #  PANEL F (Log-likelihood)  # # #
if "F" in PANEL:
    ax = axes["loglike"]
    x = Pid
    y = np.array([data[pid]['ll2chance'] for pid in Pid])
    ax.plot(x, y, 'xr', ms=3.5, mew=0.85, alpha=0.8, label="This work", zorder=5)
    ymax = y.max()
    try:
        llyang = np.loadtxt(yangLL)
        llyang -= llchance  # subtract chance level
        ax.plot(x, llyang, '_b', ms=4.5, mew=1.0, label="Yang et al. (2021)")
        ymax = max(ymax, llyang.max())
    except Exception:
        print(" > Cannot plot reference data from Yang et al. (2021).")
    ax.hlines(0, Pid[0]-1, Pid[-1]+1, colors='0.35', lw=0.75)
    leg = ax.legend(loc=(0.03, 0.09), fontsize=5, facecolor='0.95')
    leg.get_frame().set_linewidth(0.65)

    ax.set_xticks(Pid)
    ax.set_xlim(Pid[0]-0.5, Pid[-1]+0.5)
    ax.set_ylim(-0.05*ymax, 1.07*ymax)
    # ax.set_ylabel("$\mathcal{L}$(model) $-$ $\mathcal{L}$(chance)")

    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    # Paired rank test
    from scipy.stats import wilcoxon
    stat, pval = wilcoxon(x=y, y=llyang, zero_method='wilcox', correction=False, \
                          alternative='two-sided', mode='auto')
    print(f"Two-sided paired Wilcoxon signed-rank test. P-value = {pval:.5f}")


# # #  SAVE AND/OR SHOW  # # #
if SAVE:
    si.plot.savefig(fig, basename="Figure_3", path="./fig/", fmt=(".png", ".pdf"), dpi=600, logger=si.log)
else:
    si.log.warning("Figure not saved!")

if SHOW:
    pl.show()
    
