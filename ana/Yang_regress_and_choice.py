# # # # # # # # # # # # # # # # # # # # # # # #
# # # #
# # # #   Modeling human choices on motion structure classification.
# # # # 
# # # #   Steps:
# # # #   1) Define a set of features on lambda
# # # #   2) Train a Logistic Regression model (LRM) on these features to predict the ground truth structure
# # # #   3) Put a decision model onto the LRM log-likelihoods like (Yang et al., 2021)
# # # #
# # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
import pylab as pl
from collections import defaultdict

# # #  PARAMETERS  # # #
PLOT = True                   # Plot some results right away? Make sure that "/ana/fig/" exists
CROSSVALIDATED = True         # Use leave-one-out cross-validation (slower) or direct max. likelihood (faster)
SAVE = True                   # Save results to file?

Pid = range(1,12+1)            # Participant IDs; range(1,12+1) for all
Fil = ('adiab_0.040', )        # Which filters to include (None: use all)

reference = "ground_truth"     # Labels for the Logistic Regression Classifier. ("choice" or "ground_truth")
timeslice = slice(-1,None)   # Which part of lambda(t) to average over? only endpoint: slice(-1,None) ; all lambda's: slice(None, None); 2nd half: slice(100,None)
piL = 0.04                   # Lapse probability in the decision model  (Yang et al: 0.14)
param0 = (0.5, 2., 0., 0.)   # inital fit parameters (beta, bg, bc, bh)

bLamSquared = False          # Use lam**2 instead of lam?
bNormalizedRange = False     # Scale each trial to range [0,1]?

# Data set labels
# sig_obs = 0.01-0.06 (0 presented), fps = 50, lam0 = marginal,
# tau_s = 0.3, tau_lam = 1., nu = kappa = 0.
from YangETAL_dsl import DSL
dataDir = "./data/"            # Location of the data from the motion structure algo.

# # # #
# # # # # #
# # FEATURES
# # # # # # # # # # #
# # # # # # # # # # # # # # #

def calc_Tg(lam):
    "Does global motion stand out?"
    assert lam.ndim == 3, "lambda must be [pid, trial, sources]"
    Tg = lam[:,:,0] / lam.sum(2)
    return Tg


def calc_Tcmax(lam):
    "Does one cluster dominate the others?"
    assert lam.ndim == 3, "lambda must be [pid, trial, sources]"
    Tc = lam[:,:,1:4].max(2) / lam[:,:,1:4].sum(2)
    return Tc


def calc_Ti(lam):
    "Does one individual component stand out?"
    assert lam.ndim == 3, "lambda must be [pid, trial, sources]"
    Ti = lam[:,:,4:].max(2) / lam[:,:,4:].sum(2)
    return Ti


def calc_Tcdom(lam):
    "Does the strongest cluster dominate its children?"
    assert lam.ndim == 3, "lambda must be [pid, trial, sources]"
    # the cluster must 1) stand out, and 2) dominate its ind. dots
    kclu = lam[:,:,1:4].argmax(2)
    kind = np.array([ (4 + k%3, 4 + (k+1)%3) for k in kclu ]).transpose((1,0,2)) # a bit clumsy construction, but correct
    P,R,M = lam.shape
    all1,all0 = np.meshgrid(np.arange(R), np.arange(P))
    Tc = lam[:,:,1:4].max(2)**2 / (lam[:,:,1:4].max(2)**2 + (lam[all0,all1,kind[0]]**2 + lam[all0,all1,kind[1]]**2) )
    return Tc


def calc_Tcnondom(lam):
    "Does the strongest cluster dominate the maverick?"
    assert lam.ndim == 3, "lambda must be [pid, trial, sources]"
    kclu = lam[:,:,1:4].argmax(2)
    kind = np.array([ 4 + (k+2)%3 for k in kclu ])
    P,R,M = lam.shape
    all1,all0 = np.meshgrid(np.arange(R), np.arange(P))
    Tc = lam[:,:,1:4].max(2)**2 / (lam[:,:,1:4].max(2)**2 + lam[all0,all1,kind]**2 )
    return Tc



def build_feature_dataset(lam):
    _,_,M = lam.shape
    X = [ ]
    # Statistics
    X.append( calc_Tg(lam) )
    X.append( calc_Tcmax(lam) )
    X.append( calc_Ti(lam) )
    X.append( calc_Tcdom(lam) )
    X.append( calc_Tcnondom(lam) )
    # so far X has order (F,P,R) with Feature, Participants, Repetitions
    return np.array(X).transpose([1,2,0]) # change to: shape = ( Participants, Repetitions, Feature )


# # # #
# # # # # #
# # LOAD DATA
# # # # # # # # # # #
# # # # # # # # # # # # # # #


def load_lambda_and_labels(pid):
    print(f" > Loading data for participant {pid:02d}.")
    from os import path
    fname = path.join(dataDir, DSL[pid], "simdata.nc")
    print(f"   > From file: '{fname}'")
    import xarray as xr
    ds = xr.open_dataset(fname)
    import json
    cfg = json.loads(ds._metadict_str)
    # Filters
    fils = ds.F.data.tolist()
    # Trials
    Rs = ds.R.data
    print(f"   > Found {len(Rs)} trials for filters: {', '.join(fils)}")
    if Fil is not None:
        print(f"   > Using data of filters: {', '.join(Fil)}")
    else:
        print(f"   > Using data of all filters.")
    # Return data
    out = cfg['label']
    out["Rs"] = Rs
    out["fil"] = fils
    out['raw'] = {}
    for fn,fil in enumerate(fils):
        if Fil is not None and fil not in Fil:
            continue
        lam = ds.Lam_inf[:,fn,timeslice].data.mean(1)  # lambda at last time point for all trials
        if bLamSquared:
            lam = lam**2
        if bNormalizedRange:
            lam = (lam - lam.min(1)[:,None]) / (lam.max(1) - lam.min(1))[:,None]
        out['raw'][fil] = lam
    return out


def load_all_data():
    from collections import defaultdict
    data = defaultdict(list)
    data['raw'] = defaultdict(list)
    for pid in Pid:
        out = load_lambda_and_labels(pid)
        for key in out:
            if key in ("fil",):
                continue
            elif key == 'raw':
                for fil in out['raw']:
                    data['raw'][fil].append(out['raw'][fil])
            else:
                data[key].append(out[key])
    for key in data:
        if key == 'raw':
            for fil in data['raw']:
                data['raw'][fil] = np.array(data['raw'][fil])
        else:
            data[key] = np.array(data[key])
    return data


# # # #
# # # # # #
# # TRAIN LOGISTIC REGRESSION CLASSIFIER
# # # # # # # # # # #
# # # # # # # # # # # # # # #

def train_LogisticRegressionClassifier(data, fil):
    from sklearn.linear_model import LogisticRegression
    P,R,F = data['X'][fil].shape
    X = data['X'][fil].reshape(P*R,F)
    y = data[reference].reshape(P*R)
    # l1 and l2 regularization give almost indistinguishable fit quality
    kwargs = dict(multi_class='multinomial', max_iter=10000, solver='saga', penalty='l1', random_state=12345, fit_intercept=True)
    clf = LogisticRegression(**kwargs).fit(X, y)
    # P = clf.predict_proba(X)
    print(f"   > Trained classifier on labels '{reference}'.\n     > Accuracy: ", end="", flush=False)
    for s in ('I','G','C','H'):
        idx = y==s
        print(f"{s}:{clf.score(X[idx],y[idx]):.2f}, ", end="", flush=False)
    # reordered log likelihood (clf orders the classes in order of first appearance)
    logP = np.array( [ clf.predict_log_proba(X)[:,np.where(clf.classes_ == s)[0][0]] for s in ('I','G','C','H') ] ).T
    choiceidx = [ ('I','G','C','H').index(c) for c in y ]
    ll = -logP[np.arange(P*R),choiceidx].sum()/P - np.log(1/4)*R
    print("\b"*2, f"\n     > Log-likelihood (per-participant equivalent): {ll:.3f}")
    return clf


# # # #
# # # # # #
# # CHOICE MODEL AND FITTING
# # # # # # # # # # #
# # # # # # # # # # # # # # #

def log_likelihood(param, logP, choiceidx, piL, yield_probs=False):
    """param = (beta, bg, bc, bh)"""
    beta, bg, bc, bh = param
    b = (0., bg, bc, bh)
    exponent = beta * (logP + b)
    l1 = np.exp(exponent)[np.arange(len(choiceidx)),choiceidx] / np.exp(exponent).sum(-1)
    l0 = 1./4.
    p = piL * l0 + (1-piL) * l1
    if yield_probs:
        if logP.ndim == 1:
            l1 = np.exp(exponent) / np.exp(exponent).sum(-1)
        elif logP.ndim == 2:
            l1 = np.exp(exponent) / np.exp(exponent).sum(-1)[:,None]
        probs = piL * l0 + (1-piL) * l1
        return np.log(p).sum(), probs
    else:
        return np.log(p).sum()


def fit_choice_model_for_participant(clf, pid, fil, piL):
    X = data['X'][fil][pid-1]
    logPred = clf.predict_log_proba(X)
    # reorder
    logP = np.array( [ logPred[:,np.where(clf.classes_ == s)[0][0]] for s in ('I','G','C','H') ] ).T
    # index of the choices in IGCH one-hot coding
    choices = data['choice'][pid-1]
    Sidx = dict(I=0, G=1, C=2, H=3)
    choiceidx = np.array([Sidx[c] for c in choices])
    f = lambda param: -log_likelihood(param, logP, choiceidx, piL)
    from scipy.optimize import minimize
    res = minimize(f, param0, method="BFGS", options={'gtol': 1e-04}) 
    if not res.success:
        print(f"     > Optimization did not converge for participant {pid}! Message: " + res.message)
    _, P_pred = log_likelihood(res.x, logP, choiceidx, piL, yield_probs=True)
    return res, P_pred


def fit_choice_model_for_participant_CV(clf, pid, fil, piL):
    "Leave-one-out cross-validated log-likelihood."
    # Technically, we will simply mask one value per iteration
    from numpy.ma import masked_array as marray
    # Get all data and reorder to IGCH
    X = data['X'][fil][pid-1]
    R = X.shape[0]
    logPred = clf.predict_log_proba(X)
    logP = marray( [ logPred[:,np.where(clf.classes_ == s)[0][0]] for s in ('I','G','C','H') ] ).T
    logP.mask = np.array( [False]*R )  # The first dim is enough and broadcasts "downwards"
    # index of the choices in IGCH one-hot coding
    choices = data['choice'][pid-1]
    Sidx = dict(I=0, G=1, C=2, H=3)
    choiceidx = np.array([Sidx[c] for c in choices])
    # Data storage
    P_pred = np.zeros((R,4))
    ll = np.zeros(R)
    # We can use the same function for all trials
    from scipy.optimize import minimize
    f = lambda param: -log_likelihood(param, logP, choiceidx, piL, yield_probs=False)
    par0 = np.array(param0) # 1st time, the global param0 is used, then the latest fit
    # Central loop
    print("     > Fitting trial " + " "*3, flush=True, end="")
    for r in range(R):
        print("\b"*3 + f"{r+1:3d}", flush=True, end="")
        # Fit on all other trials
        logP.mask[r] = True             # mask current trial
        res = minimize(f, par0, method="BFGS", options={'gtol': 1e-04})
        if not res.success:
            print(f"     > Optimization did not converge for participant {pid} trial {r+1}! Message: " + res.message)
        par0 = np.array(res.x)        # this makes a copy which can serve as the next initializer
        # Calc logL of the trial
        logP.mask[r] = False   # unmask trial
        ll[r], P_pred[r] = log_likelihood(res.x, logP[[r]], choiceidx[[r]], piL, yield_probs=True)
    print("\b"*100, flush=True, end="")
    res = dict(ll=ll.sum(), P_pred=P_pred)
    return res
    
    




# # # #
# # # # # #
# # PLOTTING
# # # # # # # # # # #
# # # # # # # # # # # # # # #

def plot_used_features(clf, label=None):
    """Plot which features are used."""
    fig = pl.figure(figsize=(3.,1.75))
    coef = np.array( [ clf.coef_[np.where(clf.classes_ == s)[0][0]] for s in ('I','G','C','H') ] )
    pl.imshow(coef, vmin=-np.abs(clf.coef_).max(), vmax=np.abs(clf.coef_).max())
    cb = pl.colorbar()
    cb.set_label("Feature strength")
    pl.yticks(range(4), ('I','G','C','H'))
    pl.ylabel("Structure", labelpad=3)
    pl.xticks(range(coef.shape[1]), 1+np.arange(coef.shape[1]))
    pl.xlabel("Feature number")
    pl.subplots_adjust(0.12,0.18,0.96,0.90)
    pl.title(label)
    fname = "./fig/fig_features_" + "".join(x if (x.isalnum() or x in ".-") else "_" for x in label) + ".pdf"
    fig.savefig(fname)
    return fig

def predict_confusion_matrix(pid):
    # Predict response distribution per trial
    X = data['X'][pid-1]
    logPred = clf.predict_log_proba(X)
    # reorder
    logP = np.array( [ logPred[:,np.where(clf.classes_ == s)[0][0]] for s in ('I','G','C','H') ] ).T
    beta, bg, bc, bh = fitchoice[pid]['param']
    B = np.array([0.,bg,bc,bh])
    pChoice = np.exp( beta * (logP + B) )
    pChoice /= np.exp( beta * (logP + B) ).sum(-1)[:,None]
    pChoice = piL/4 + (1-piL)*pChoice
    # Ground truth per trial
    gt = data['ground_truth'][pid-1]
    # Assemble confusion matrix
    M = np.zeros((4,4))  # gt,choice
    labels = ("I", "G", "C", "H")
    for l,p in zip(gt, pChoice):
        M[labels.index(l)] += p[0], p[1], p[2], p[3]
    M /= M.sum(1)[:,None]  # Normalize per row
    return M


def plot_confusion_matrix(pid, P_pred, ax):
    labels = "I", "G", "C", "H"
    M = np.zeros((4,4))
    for ci,gt in enumerate(labels):
        idx = data['ground_truth'][pid-1] == gt
        M[ci] = P_pred[idx].mean(0)
    kwargs = dict(vmin=0, vmax=1, cmap=pl.cm.Blues)
    ax.imshow(M, **kwargs)
    for y,row in enumerate(M):
        for x,p in enumerate(row):
            c = "w" if p > 0.5 else 'k'
            ax.text(x, y, f"{p:.2f}"[1:], weight="bold", ha='center', va='center', color=c)
    ax.set_title(f"#{pid}")
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(labels)
    ax.set_yticks([0,1,2,3])
    ax.set_yticklabels(labels)
    if pid == 1:
        ax.set_xlabel("Predicted choice")
        ax.set_ylabel("True structure")


def plot_log_likelihoods_to_chance(y, label=None):
    fig = pl.figure(figsize=(3.5,2.25))
    x = Pid
    pl.plot(x, y, 'vr', ms=6, label="This work", zorder=5)
    ymax = y.max()
    try:
        fname = "./data/data_yang2021_experiment_1/yang_logL_exp1_4paramModel.txt"
        llyang = np.loadtxt(fname)
        llyang -= 200 * np.log(1/4)  # subtract chance level
        pl.plot(x, llyang, 'bo', label="Yang et al. (2021)")
        ymax = max(ymax, llyang.max())
    except Exception:
        print(" > Cannot plot reference data from Yang et al. (2021).")
    pl.hlines(0, Pid[0]-1, Pid[-1]+1, colors='0.25', lw=1)
    leg = pl.legend(loc=(0.02, 0.11))
    pl.xticks(Pid)
    pl.xlim(Pid[0]-0.5, Pid[-1]+0.5)
    pl.ylim(-0.07*ymax, 1.1*ymax)
    pl.xlabel("Participant")
    pl.ylabel("$\mathcal{L}$(model) $-$ $\mathcal{L}$(chance)")
    pl.title(label, pad=4)
    pl.subplots_adjust(0.15, 0.15, 0.98, 0.93)
    fname = "./fig/fig_ll2chance_" + "".join(x if (x.isalnum() or x in ".-") else "_" for x in label) + ".pdf"
    fig.savefig(fname)
    return fig

# # # #      # # # # # # # # # # # # # # #
# # # # # #          # # # # # # # # # # #
# #               M A I N              # #
# # # # # # # # # # #          # # # # # #
# # # # # # # # # # # # # # #      # # # #

np.set_printoptions(precision=2)

# # #  Load data and calculate features from lambda  # # #
print(" > Loading data...")
# Load all participants
data = load_all_data()
if Fil is None:
    Fil = tuple(data['raw'].keys())
# 1) Calculate the features
data['X'] = {}
for filname in Fil:
    data['X'][filname] = build_feature_dataset(data['raw'][filname])

print(f" > Loaded data of {len(data['X'].keys()):d} filters from {data['Rs'].shape[0]:d} participants.")

# # # Central for-loop over the filters  # # #
fitchoice = defaultdict(list)
# fil = 'adiab_0.020'
for fil in Fil:
    print(f"\n > Now processing filter '{fil}'.")
    # 2) Train the logistic regression classifier on the ground truth
    clf = train_LogisticRegressionClassifier(data, fil)
    if PLOT:
        plot_used_features(clf, label=f"Filter: {fil}")
        # Preprare figure for confusion matrix
        figCM = figure(figsize=(0.5+1.5*6, 0.5+1.5*2))
        axes = figCM.subplots(2, 6)
    if not CROSSVALIDATED:
        # 3a) Fit the choice model w/o cross validation (for quickly comparing filters and piL)
        print("   > Fitting choice model (w/o cross-validation):")
        fitchoice[fil] = defaultdict(list)
        # w/o cross-validation, fitchoice[fil][pid] has keys 'param' and 'll2chance'
        summed = 0
        for pid in Pid:
            res, P_pred = fit_choice_model_for_participant(clf, pid, fil, piL)
            lchance = len(data['Rs'][pid-1]) * np.log(1/4)
            fitchoice[fil][pid] = dict(param=res.x, ll2chance=-res.fun - lchance, P_pred=P_pred)
            print(f"     > Pid:{pid:2d}, logL-to-chance={fitchoice[fil][pid]['ll2chance']:6.2f}\t  (pi_lapse={piL:.2f})" )
            summed += fitchoice[fil][pid]['ll2chance']
        print(f"     > Summed logL: {summed:.2f}")
    else:
        # 3a) Fit the choice model with cross validation (slower, but no risk of over-fitting)
        print("   > Fitting choice model (cross-validated):")
        fitchoice[fil] = defaultdict(list)
        # with cross-validation, fitchoice[fil][pid] has keys 'll2chance' and 'P_pred', which is a Rx4 array
        summed = 0
        for pid in Pid:
            res = fit_choice_model_for_participant_CV(clf, pid, fil, piL)
            lchance = len(data['Rs'][pid-1]) * np.log(1/4)
            fitchoice[fil][pid] = dict(ll2chance=res['ll'] - lchance, P_pred=res['P_pred'])
            print(f"     > Pid:{pid:2d}, logL-to-chance={fitchoice[fil][pid]['ll2chance']:6.2f}\t  (pi_lapse={piL:.2f})" )
            summed += fitchoice[fil][pid]['ll2chance']
        print(f"     > Summed logL: {summed:.2f}")
    if PLOT:
        # Plot confusion matrix
        for pid,ax in zip(Pid,axes.flatten()):
            plot_confusion_matrix(pid, fitchoice[fil][pid]['P_pred'], ax)
        figCM.subplots_adjust(0.05,0.05, 0.98,0.90,0.3,0.3)
        fname = "./fig/fig_confusionmatrix_" + "".join(x if (x.isalnum() or x in ".-") else "_" for x in f"Filter: {fil}") + ".pdf"
        figCM.savefig(fname)
        # Title
        ax = figCM.add_axes([0.1,0.94,0.8,0.04], frame_on=False, xticks=[], yticks=[])
        ax.text(0.5, 0.5, f"Confusion matrices for filter '{fil}'", ha='center', va='center')
        # Plot the logL in comparison to Yang et al.
        y = np.array([ fitchoice[fil][pid]['ll2chance'] for pid in Pid ])
        figLL = plot_log_likelihoods_to_chance(y, label=f"Filter: {fil}")
    # Include also the ground truth and choice for further analysis and plotting
    for pid in Pid:
        fitchoice[fil][pid]['ground_truth'] = data['ground_truth'][pid-1]
        fitchoice[fil][pid]['choice'] = data['choice'][pid-1]


if SAVE:
    fname = "./data/data_yang2021_experiment_1/yang_predicted_choices_by_motion_structure_inference_algorithm"
    if CROSSVALIDATED:
        fname += "_CV"
    fname += ".pkl"
    import os
    if os.path.exists(fname):
        raise Exception(f"File '{fname}' exists. New data NOT saved.")
    import pickle
    with open(fname, "wb") as f:
        pickle.dump(fitchoice, f)


