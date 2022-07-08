# # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # 
# # # 
# # #   The network model supports a linear readout of lambda**2 from the linear population.
# # # 
# # #   Here we ask: 1) With A_readout unknown, can we nonetheless train a linear regression model 
# # #                   to predict lam2 from neural activity r, based on some extreme values of lam2?
# # #                2) How would the predicted lam**2 look at intermediate values (when predicted from r)?
# # # 
# # #   For this, we consider a setup with lam_glo**2 + lam_ind**2 := lam_tot**2 == const. 
# # #   We vary the fraction q = lam_glo**2 / lam_tot**2 and test for the readout of q
# # # 
# # #   
# # # 
# # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import numpy as np
import pylab as pl
import strinf as si

# # #  PARAMS  # # #

DSL = "2021-08-30-14-28-45-031631_network_global_sweep_lamG"
fil = "net"              # filter to use
pop = "r_lin"            # neural population
duration = 5.            # length of the segement (t_end - duration, t_end) of neural activity to consider
idx_train = (0, 6)       # which indices of lam_list are taken for training the regressor (remainder: prediction)

# # #  END OF PARAMS  # # #

# # #  Load data  # # #
ds, cfg = si.load_dataset(DSL, F=fil)
lam_list = cfg["wld"]["lamList"]
reps = cfg["glo"]["R"]
Lam = np.array([ lam for _,lam in lam_list ])
T_end = np.array([ ti + lam_list[1][0] - lam_list[0][0] for ti,_ in lam_list ])
lam2_tot = Lam[0,0]**2 + Lam[0,1]**2
Q = Lam[:,0]**2 / lam2_tot
nQ = len(T_end)
assert np.allclose(Lam[0,0]**2 + Lam[0,1:]**2, lam2_tot)

# Reshape the data array and select relevant times
R = ds[pop]         # Firing rates; shape (reps, t_tot, N)
N = R.shape[-1]     # num Neurons
nTperlam = int((ds.t < T_end[0]).sum())  # num time steps per lambda value
R = R.data[:,:-1].reshape(reps, nQ, nTperlam, N)  # drop the very last time step, and reshape
tidx = (T_end[0] - duration <= ds.t) * (ds.t <= T_end[0]) # relevant duration (assuming convergence of lambda)
R = R[:,:,tidx[:nTperlam],:]  # ...and slice..
R = R.swapaxes(0,1)   # --> THIS IS OUR DATA: shape = (nQ, reps, duration, neurons)

# # #  Build the training set  # # # 
 
y = np.hstack([ [Q[i]]*(R.shape[1] * R.shape[2]) for i in idx_train ])
X = np.vstack([ R[i,r] for i in idx_train for r in range(reps)] )

# # #  Fit the linear regression model  # # #

from sklearn import linear_model
clf = linear_model.LinearRegression()
clf.fit(X, y)

fig = pl.figure(figsize=(2.5,2))
y_pred =  clf.coef_ @ X.T + clf.intercept_
kwargs = dict(marker='.', ms=0.5, mew=0., lw=0.)
pl.plot(y + np.random.normal(0,0.01,len(y)), y_pred, mfc='b', label="Training set", **kwargs)

# # #  Build the test set  # # #

idx_test = set(range(nQ)).difference(idx_train)

y_true = np.hstack([ [Q[i]]*(R.shape[1] * R.shape[2]) for i in idx_test ])
X = np.vstack([ R[i,r] for i in idx_test for r in range(reps)] )

y_pred = clf.coef_ @ X.T + clf.intercept_
pl.plot(y_true + np.random.normal(0,0.01,len(y_true)), y_pred, mfc='r', label="Test set", **kwargs)

pl.plot([0,1], [0,1], '0.5', lw=1., zorder=10)
pl.xlabel("True fraction of global motion (plus noise for visibility)")
pl.ylabel("Predicted fraction of global motion")
pl.xlim(0, 1)
pl.ylim(0, 1)

pl.xticks(Q)
pl.yticks(Q)

leg = pl.legend(markerscale=10)
pl.subplots_adjust(0.17, 0.14, 0.99, 0.99)

fig.savefig("fig/fig_network_linear_readout_of_lam2.png", dpi=600)