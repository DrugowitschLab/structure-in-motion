"""Class for data storage based in the xarray package. Meta information is serialized via JSON."""

import numpy as np
import json
from . import log

pi = np.pi
numerical_infty = 1.e10                   # covariance values to mimic infinity (for velocity observations)

class TrialStorage(object):
    def __init__(self, vars={}, missingData=np.NaN, logger=None):
        """"
        vars = { 'x' : ('t', 'n'), }
        """
        self.vars = vars
        self.missingData = missingData
        self.logger = logger
        self.isInitialized = False
        if self.logger:
            self.logger.info(f"Create storage with variables {tuple(vars.keys())}.")

    def init_indices_and_coords(self, namesOfIndices, listOfIndices, coords):
        assert not self.isInitialized, "Already initialized!"
        assert len(namesOfIndices) == len(listOfIndices), "Length of names and indices does not match!"
        import pandas as pd
        import xarray as xr
        self.namesOfIndices = tuple(namesOfIndices)
        self.indices = {k:v for k,v in zip(namesOfIndices,listOfIndices)}
        # self.midx = pd.MultiIndex.from_product(listOfIndices, names=namesOfIndices)
        self.coords = coords
        self.allCoords = self.indices.copy()
        self.allCoords.update(self.coords)
        # self.coords["midx"] = self.midx
        self.ds = xr.Dataset(coords=self.allCoords)
        # self.df = pd.DataFrame(index=self.midx)
        indicesDims = tuple([ len(self.indices[i]) for i in namesOfIndices ])
        for var,co in self.vars.items():
            dims = tuple([ len(self.coords[c]) for c in co ])
            self.ds[var] = self.namesOfIndices + tuple(co), self.missingData * np.ones(indicesDims + tuple(dims))
        self.isInitialized = True

    def dump_trial(self, idx, **kwargs):
        assert self.isInitialized, "Not initialized!"
        for v,x in kwargs.items():
            assert v in self.vars, f"Unknown variable {v}."
            self.ds[v].loc[idx] = x

    def save_to_disk(self, filename):
        from os import path
        if path.exists(filename):
            log.error(f"File '{filename}' exists! Aborting storage (data has not been saved).")
            return
        # We have to serialize the meta dict as it may contain non-standard dtypes
        import json
        metadictstr = json.dumps(self.ds.attrs, cls=MyJSONEncoder, indent = 4)
        self.ds.attrs = {"_metadict_str" : metadictstr}
        # Load via: json.loads(ds.attrs["_metadict_str"])
        # Save to disk
        self.ds.to_netcdf(filename)
        # Load via: ds = xr.open_dataset(filename)
        log.info(f"TrialStorage saved to file '{filename}'.")

    def set_metadata(self, metadict={}, **kwargs):
        assert self.isInitialized, "Initialize first."
        self.ds.attrs.update(metadict)
        self.ds.attrs.update(kwargs)


class MyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        import types
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, types.FunctionType):
            return str(obj)
        # Let the base class default method raise the TypeError
        try:
            return json.JSONEncoder.default(self, obj)
        except:
            return str(obj)


def load_dataset(dsl, datadir="./data", F=None, R=None):
    """
    Load data for dataset label 'dsl' located in 'datadir'.
    
    Optional:
      * Only load data for filter 'F' (str, or list of str)
      * Only load data for trial repetition 'R' (int, or list of int)
      
    Returns:
      (ds, cfg) with
        ds  : xarray dataset
        cfg : config dict
    """
    import xarray as xr
    import json
    from pathlib import Path
    filename = Path(datadir, dsl, 'simdata.nc')
    log.info(f"Load dataset from file '{filename}'.")
    ds = xr.open_dataset(filename)
    cfg = json.loads(ds.attrs["_metadict_str"])
    locdict = dict()
    if F is not None:
        locdict["F"] = F
        log.info(f"Subselect F={F}")
    if R is not None:
        locdict["R"] = R
        log.info(f"Subselect R={R}")
    ds = ds.loc[locdict]
    return ds, cfg

