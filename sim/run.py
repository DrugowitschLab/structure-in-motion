"""Main script for running simulations (algorithmic level and network)."""

import numpy as np
import pylab as pl
import strinf as si

# Get config
cfgfname = si.import_config(globals())

# Setup logger
si.init_logging(loglevel=cfg['glo']['loglevel'], dryrun=cfg['glo']['DRYRUN'], outdir=cfg['glo']['outdir'])
si.log.info(f"Config imported from '{cfgfname}'.")
si.log.info(f"DATASET LABEL is '{cfg['glo']['dsl']}'.")
si.copy_cfg(cfgfname=cfgfname, dryrun=cfg['glo']['DRYRUN'], outdir=cfg['glo']['outdir'])

# Create wld
wldCls = cfg["wld"].pop("cls")
wld = wldCls(**cfg["wld"])

# Create obs
obsCls = cfg["obs"].pop("cls")
obs = obsCls(**cfg["obs"])

# Is a neural network involved (if so, we will store firing rates)
bNETWORK = []     # will be updated if network is detected

# Create filters (observer model or network)
filters = {}
fil_defparams = cfg["fil"].pop("default_params")
default_inf_method_kwargs = fil_defparams.pop("inf_method_kwargs", dict())
default_feature_learning_kwargs = fil_defparams.pop("feature_learning_kwargs", dict())
for filname in cfg["fil"]:
    d = cfg["fil"][filname]
    filCls = d.pop("cls")
    pdict = dict()
    pdict.update(fil_defparams, **d)
    fil = filCls(**pdict)
    filters[filname] = fil
    if hasattr(fil, "bNETWORK") and (fil.bNETWORK is True):
        bNETWORK.append(filname)

if bNETWORK != []:
    si.log.info(f"Networks detected. Storing firing rates for the following filters: {', '.join(bNETWORK)}")

# # #  Setup storage  # # #
# General variables
vars = {'S_wld' : ('t', 'd', 'm'), 'V_wld' : ('t', 'd', 'k'), 'V_obs' : ('t', 'd', 'k'), 'S_inf' : ('t', 'd', 'm'), 'Sig_inf' : ('t', 'd', 'm'), 'Lam_inf' : ('t', 'm'),}
# Feature learning active (assumes that all or none filters do feature learning)?
if fil.feature_learning_method is not None:
    vars['C_learn'] = ('t', 'k', 'm')
# Neural networks involved?
netvars, netcoords = {}, {}
for filname in bNETWORK:
    for rname,r in filters[filname].r.items():
        varname = "r_" + rname
        idxname = "i_" + rname
        if varname in netvars:
            assert (netcoords[idxname] == np.arange(len(r))).all(), "Populations of same name must have identical size across filters."
        netvars[varname] = ('t', idxname)
        netcoords[idxname] = np.arange(len(r))
vars.update(netvars)
storage = si.TrialStorage(vars=vars, logger=si.log)

# # #  Main loop  # # #
si.log.debug(f"Entering main loop.")
# For every trial repetition r:
for r in range(cfg['glo']['R']):
    si.log.info(f"Start of trial repetition {r+1} of {cfg['glo']['R']}.")
    si.log.info(f"  Generating input of duration {cfg['glo']['T']}s.")
    # Create trial input
    V = obs.run_sim_and_generate_observations(T=cfg['glo']['T'], wld=wld)   # noisy velocity observations
    T = obs.get_times()                                                     # observation times
    S = wld.S[wld.index_of_t(T)]                      # ground truth sources are only for later data storage
    # For every filter
    for filname in filters:
        fil = filters[filname]
        fil.init_filter()   # resets the filter state for the trial
        si.log.info(f"  Inferring motion strengths for filter '{filname}'.")
        # Little iterator for printing progress to the console
        pripro = si.print_progress(0, obs.dt, cfg['glo']['PRINTEVERY'])
        # Central loop: present stimulus, infer motion sources s and motion strenths lam
        for t,v in zip(T,V):
            next(pripro)         # Print progress to console, only
            fil.propagate_and_integrate_observation(v, t)
            fil.infer_strengths(**default_inf_method_kwargs)  # default kwargs are superseded by individual kwargs if present
            # Optional learning of C matrix
            if fil.feature_learning_method is not None:
                fil.learn_motion_features(v=v, **default_feature_learning_kwargs)
        next(pripro); print("")
        # # #  Store data  # # #
        # First call? -> initialize (only after running at least one trial, we know the coordinates t,k,m,...)
        if not storage.isInitialized:
            coords = {"t" : T, "k" : np.arange(wld.K), "d" : np.arange(wld.D), "m" : np.arange(wld.M) }
            coords.update(netcoords)
            # indices = trial repetition and filters; coordinates = t, k, m,...
            storage.init_indices_and_coords(namesOfIndices=("R", "F"),
                                            listOfIndices=[np.arange(cfg['glo']['R']), list(filters.keys())],
                                            coords=coords)
        # Store data of any type of filter
        storage.dump_trial(idx={"R" : r, "F" : filname}, S_wld=S.reshape(-1,fil.D,fil.M),
                           V_wld=wld.get_v(S).reshape(-1,fil.D,fil.K),  # noise-free velocities
                           V_obs=V.reshape(-1,fil.D,fil.K),             # noisy velocities
                           S_inf=np.array(fil.archive["mu"])[1:],       # mean motion source
                           Sig_inf=np.array(fil.archive["Sig"])[1:].diagonal(axis1=1,axis2=2).reshape(-1,fil.D,fil.M),    # Var motion source
                           Lam_inf=np.array(fil.archive["lam"])[1:]     # motion strength
                           )
        # Only if feature learning was active
        if fil.feature_learning_method is not None:
            storage.dump_trial(idx={"R" : r, "F" : filname}, C_learn=np.array(fil.archive["C"])[1:])
        # Store firing rates (only if the filter is a neural network)
        if filname in bNETWORK:
            rdict = { "r_"+rname : np.array(fil.archive["r_"+rname])[1:] for rname in fil.r }
            storage.dump_trial(idx={"R" : r, "F" : filname}, **rdict)

si.log.debug(f"Exiting main loop.")

# Save the config as metadata (will be serialized via JSON)
fil_defparams["inf_method_kwargs"] = default_inf_method_kwargs                # we had pop'ed these earlier; now: re-add
fil_defparams["feature_learning_kwargs"] = default_feature_learning_kwargs    # we had pop'ed these earlier; now: re-add
cfg["fil"]["default_params"] = fil_defparams                                  # we had pop'ed these earlier; now: re-add
storage.set_metadata(cfg)

# Also store the population codes (if present)
if bNETWORK != []:
    popCode = { filname : { pop : pc.get_storable_dict() for pop,pc in filters[filname].pc.items() } for filname in bNETWORK }
    storage.set_metadata(popCode=popCode)

# Save data
if not cfg['glo']['DRYRUN']:
    from os import path
    fname = path.join(cfg['glo']['outdir'], "simdata.nc")
    storage.save_to_disk(fname)

# # # # # #   That's it :)  # # # # # 
si.log.info(f"Simulation completed.")
# # # # # # # # # # # # # # # # # # #

# # # What follows are some plots for immediate inspection  # # #
if ("PLOTAFTER" not in cfg['glo']) or (cfg['glo']["PLOTAFTER"] is False):
    import sys
    si.log.info("No request for plots. Exiting.")
    sys.exit(0)

si.log.info("Plotting some simulation results.")


# Create ./fig directory, if not yet present
import pathlib
pathlib.Path("fig").mkdir(exist_ok=True)

# Plot lambdas of first trial
Lam = { fil : storage.ds.Lam_inf.loc[0,fil] for fil in filters}
lam_star = wld.lamList if hasattr(wld, "lamList") else wld.lam
figdict = si.plot_inferred_lambda(t=T, Lam=Lam, C=fil.C0, lam_star=lam_star, cfg=cfg)
fig = figdict["fig"]
fig.savefig("./fig/fig_lambda.pdf")
fig.savefig("./fig/fig_lambda.png")

# Plot sources
# Only the x-component of first trial
S = { fil : storage.ds.S_inf.loc[0,fil][:,0] for fil in filters}
S_star = storage.ds.S_wld[0,0][:,0]
figdict = si.plot_inferred_source(t=T, S=S, C=fil.C0, S_star=None) # S_star)
fig = figdict["fig"]
fig.savefig("./fig/fig_source_x.pdf")
fig.savefig("./fig/fig_source_x.png")

# Plot sources
# Only the y-component  of first trial
if wld.D > 1:
    S = { fil : storage.ds.S_inf.loc[0,fil][:,1] for fil in filters}
    S_star = storage.ds.S_wld[0,0][:,1]
    figdict = si.plot_inferred_source(t=T, S=S, C=fil.C0, S_star=None) #S_star)
    fig = figdict["fig"]
    fig.savefig("./fig/fig_source_y.pdf")
    fig.savefig("./fig/fig_source_y.png")

# Print some Yang trials (if present)
if "label" in cfg:
    np.set_printoptions(precision=2)
    for i,tr in enumerate(storage.ds.Lam_inf):
        print(i, cfg['label']['ground_truth'][i], cfg['label']['choice'][i], ":", tr[0,-1].data, "\t", tr[1,-1].data)

# Show the plot results
pl.show()
