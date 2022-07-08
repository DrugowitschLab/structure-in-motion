import numpy as np
import pickle
from . import trialNumberAware

def load_data(file, key: str = None) -> list:
    data = []
    with open(file, 'rb') as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break
    if key:
        vals = []
        for trial in data:
            vals.append(trial[key])
        return vals
    return data


def dist_mod2pi(x, y=None):
        from numpy import pi
        if y is not None:
            d = x - y
        else:
            d = x.copy()
        d += pi
        d %= (2*pi)
        d -= pi
        return d


class PresentYangTrials(object):
    def __init__(self, pid, K=3, datadir="./data/data_yang2021_experiment_1/"):
        self.pid = pid
        self.K = K
        from os import path
        datafname = path.join(datadir, str(pid), f"{pid}_exp1.dat")
        self.data = load_data(datafname)
        self.R = len(self.data)
        self.V = []
        for trial in self.data:
            X = np.array(trial['φ'])[:,:K]
            T = np.array(trial['t'])
            self.V.append( dist_mod2pi(X[1:], X[:-1]) / (T[1:] - T[:-1])[:,None] )
        self.ground_truth = [ trial['ground_truth']  for trial in self.data ]
        self.choice = [ trial['choice']  for trial in self.data ]
        self.confidence = [ trial['confidence']  for trial in self.data ]


    def __call__(self, t, trialNumber):
        V = self.V[trialNumber]
        T = np.array(self.data[trialNumber]['t'][:-1])
        tn = np.argmin(np.abs(T-t))
        return [V[tn]]



