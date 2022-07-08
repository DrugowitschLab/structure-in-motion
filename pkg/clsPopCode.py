"""Population code classes, mostly for convenient bookkeeping."""

import numpy as np
from . import log as logger

def turn_keys_into_str(d):
    """Replaces python-hashable keys by their string representation. Required for dict keys in JSON."""
    return { str(k) : v for k,v in d.items() }


class LinearPopulationCodeBase (object):
    """
    Base class for linear population codes. Purpose:
      * One instantiation per neural population
      * Manage linear readout of variables
      * Manage access to the submatrices for individual variable groups
      * Help create the coding matrices, such that the adjoint matrix AT fulfills A * AT = I.

    Usage:
      Initialization:
      * Upon creation, only the number of neurons, N, is needed. Optional: rngseed.
      * Then, register all variables, x, with their 'name' and dimensions dim(x) as tuple:
        self.register(name1=(dims1, readoutarray1), name2=(dims2, f_readout ), ...)
      * On registration, specify the readout vector:
        * provide an explicit readout matrix, ndarray with shape = dim(x) + (N,), or
        * pass a function for generating a matrix, call signature: f_readout( dim(x), N, rng ).
        * The type of available options may depend on the subclass of LinearPopulationCodeBase.
      * Call self.finalize() to calculate the adjoint matrix AT.
      After initialization:
      * Access the full matrices via self.A() and self.AT()
      * Access submatrices via self.A('name') and self.AT('name'). Add flat=False to get matrix in shape=dim(x) + (N,)
      * Readout via self.dec(r) with rate vector r of len N or matrix of shape (T,N) --> dict of all vars.
      * self.dec(r, name='name') to only decode variable 'name'
      * self.pseudoencode(**kwargs={'name1' : x1, 'name2' : x2}) --> use AT to encode some vars; all other vars assumed zero.
    """
    def __init__(self, N, rngseed=None):
        self.N = N
        self.rng = np.random.RandomState(rngseed)
        logger.debug(f"Create {self.__class__.__name__} with size {N}.")
        self._bInitialized = False
        # Coding matrices
        self._A = []
        self._AT = []
        # Registered variables name = dict(dim=x.shape, idx=(1st idx, last+1 idx))
        self._X = dict()
        self._lenA = 0

    def register(self, _kwargsdict=None, **kwargs):
        # _kwargsdict is a compatibility function to support non-string keys in kwargs
        if _kwargsdict is not None:
            assert kwargs == dict(), "If using _kwargsdict (Warning: private keyword!), kwargs must be empty."
            kwargs = _kwargsdict
        vardict = self._register_checktypes(kwargs)
        # Subclass specific code here
        raise NotImplementedError("Must be implemented by subclasses.")

    def _register_checktypes(self, kwargs):
        assert not self._bInitialized, "Cannot register new variables after finalize() was called."
        for name in kwargs:
            assert len(kwargs[name]) == 2, f"Must have form {name}=(dim, array OR f_readout), but received {kwargs[name]}."
            dim = kwargs[name][0]
            if np.iterable(dim):
                dim = tuple(dim)
            elif isinstance(dim, (int, np.integer)):
                dim = (dim,)
            else:
                logger.error(f"Invalid dimensions {dim} for variable '{name}'.")
                raise Exception(f"Invalid dimensions {dim} for variable '{name}'.")
            Af = kwargs[name][1]
            if callable(Af):
                from inspect import signature
                assert len(signature(Af).parameters) == 3, f"Passed function for {name} must be called as f(dim(x), N, rng)."
            else:
                Af = np.array(Af)
                assert Af.shape == dim + (self.N,), f"Readout weights for {name} must have shape {dim + (self.N,)}, but received {Af.shape}."
            kwargs[name] = (dim, Af)
        return kwargs

    def finalize(self):
        self._subclass_finalize()
        self._A = np.array(self._A)
        logger.debug(f"Calculating AT for {self.__class__.__name__} with size {self.N}")
        from numpy.linalg import lstsq
        K,N = self._A.shape
        AT = lstsq(self._A, np.eye(K), rcond=None)[0]
        self._AT = AT
        self._bInitialized = True

    def _subclass_finalize(self):
        "Optional subclass-specific finalization"
        pass

    def A(self, name=None, flat=True):
        return self._A_or_AT(self._A, name, flat)

    def AT(self, name=None, flat=True):
        return self._A_or_AT(self._AT.T, name, flat).T

    def _A_or_AT(self, M, name, flat):
        if not self._bInitialized:
            logger.error("Cannot access coding matrices before calling finalize().")
            raise Exception("Cannot access coding matrices before calling finalize().")
        if name is None:
            return M
        assert name in self._X, f"Unknown variable name '{name}'"
        idx = self._X[name]['idx']
        dim = self._X[name]['dim']
        Mx = M[slice(*idx)]
        if flat:
            return Mx
        else:
            return Mx.reshape( dim + (self.N,) )


    def dec(self, r, name=None, flat=False):
        if name is not None:
            x = self._dec_variable(r, name, flat)
            return x
        else:
            X = dict()
            for name in self._X.keys():
                X[name] = self._dec_variable(r, name, flat)
            return X


    def _dec_variable(self, r, name, flat):
        assert name in self._X.keys(), f"Variable '{name}' not registered."
        if isinstance(name, tuple) and len(name) == 2:  # compatibility with (group, name) decoding
            A = self.A(name[0], name[1], flat=True)
        else:
            A = self.A(name, flat=True)
        assert A.shape[1] == r.shape[-1], f"Rate vector/matrix has {r.shape[-1]} neurons, but expected {A.shape[1]}."
        if r.ndim == 1:
            x = A @ r
        elif r.ndim == 2:
            x = (A.T @ r).T
        else:
            raise Exception("Rate r must be 1 (N) or 2 (T,N) dimensional.")
        if flat:
            return x
        else:
            return x.reshape(self._X[name]['dim'])


    def pseudoencode(self, **kwargs):
        AT = self.AT()
        N,K = AT.shape
        X = np.zeros(K)
        for name, x in kwargs.items():
            assert name in self._X.keys(), f"Variable '{name}' not registered."
            idx = self._X[name]['idx']
            dim = self._X[name]['dim']
            if x.ndim > 1:
                assert x.shape == dim
                x = x.flatten()
            X[slice(*idx)] = x
        r = AT @ X
        return r

    def get_storable_dict(self):
        """Return a dict-represenation that can be serialized via JSON."""
        assert self._bInitialized is True
        return dict(X=turn_keys_into_str(self._X), A=self.A(), AT=self.AT())


# For LinearPopulationCode: maybe add avgrate option. I think that Beck's all-1 approach is not correct because 1 is not
# orthogonal to all readouts, though.
class LinearPopulationCode (LinearPopulationCodeBase):
    """
    The most common subtype of LinearPopulationCodeBase. Use it for latent variables.
    Functions pretty much like described for the base class.
    """
    def __init__(self, N, rngseed=None):
        super().__init__(N=N, rngseed=rngseed)

    def register(self, _kwargsdict=None, **kwargs):
        # _kwargsdict is a compatibility function to support non-string keys in kwargs
        if _kwargsdict is not None:
            assert kwargs == dict(), "If using _kwargsdict (Warning: private keyword!), kwargs must be empty."
            kwargs = _kwargsdict
        vardict = self._register_checktypes(kwargs)
        # Subclass specific code here
        for name, (dim, Af) in vardict.items():
            assert name not in self._X, f"Error: Variable {name} already registered."
            size = np.prod(dim)
            if callable(Af):
                Ax = Af(dim, self.N, self.rng).reshape(size, self.N)
            else:
                Ax = Af.reshape(size, self.N)
            for a in Ax:
                self._A.append(list(a))
            self._X[name] = dict(dim=dim, idx=(self._lenA, self._lenA+size))
            self._lenA += size
            logger.debug(f"Registered variable '{name}' of dim={dim} for population of size {self.N}.")


class OneToOnePopulationCode (LinearPopulationCodeBase):
    """
    A subtype of LinearPopulationCodeBase which assigns a local identity matrix to the next set of prod(dim(x)) neurons,
    optionally with provided readout factors (weights).
    """
    def __init__(self, rngseed=None):
        super().__init__(N=0, rngseed=rngseed)

    def register(self, **kwargs):
        """
        Changed syntax for 1-to-1 populations:
        register(x1=(dim, ax)) with ax either an array of shape==dim (one value per variable), or a function(dim, rng).
        """
        assert not self._bInitialized, "Cannot register new variables after finalize() was called."
        for name in kwargs:
            assert name not in self._X, f"Error: Variable {name} already registered."
            assert len(kwargs[name]) == 2, f"Must have form {name}=(dim, array OR int), but received {kwargs[name]}."
            dim = kwargs[name][0]
            if np.iterable(dim):
                dim = tuple(dim)
            elif isinstance(dim, (int, np.integer)):
                dim = (dim,)
            else:
                logger.error(f"Invalid dimensions {dim} for variable '{name}'.")
                raise Exception(f"Invalid dimensions {dim} for variable '{name}'.")
            size = np.prod(dim)
            Af = kwargs[name][1]
            if callable(Af):
                from inspect import signature
                assert len(signature(Af).parameters) == 2, f"Passed function for {name} must be called as f(dim(x), rng)."
                Ax = np.diag(Af(dim, self.rng).flatten())
                assert Ax.shape == (size,size)
            else:
                Af = np.array(Af)
                assert Af.shape == dim, f"Readout weights for {name} must have shape {dim}, but received {Af.shape}."
                Ax = np.diag(Af.flatten())
            self._X[name] = dict(dim=dim, idx=(self._lenA, self._lenA+size))
            self._A.append(Ax)
            self._lenA += size
            logger.debug(f"Registered variable '{name}' of dim={dim} for 1-to-1 population.")

    def _subclass_finalize(self):
        # Build A and set N
        from scipy.linalg import block_diag
        self._A = block_diag(*self._A)
        assert self._A.shape == (self._lenA, self._lenA)
        self.N = self._lenA


class LinearInputPopulationCode (LinearPopulationCode):
    """
    A subtype of LinearPopulationCodeBase tailored for input neurons.
    The main difference is in the registration of variable groups:
     self.register(group, f_encode, name1=(dims1, A1), ...)
     1) Each call of self.register treats the passed variables as belonging together (a variable group).
        Any hashable group identifier is allowed.
     2) Each group will maintain its own population of size N.
     3) An encoding function is assigned to the group via f_encode, which encodes the variables
        received as observation: f_encode(*observation) --> array of len N
     4) self.finalize() will tack the group matrices together to form one big matrix over numGroups * N neurons.
    In the simulation:
    * Call self.enc(obsdict) with obsdict = { group1 : obs1, group2 : obs2} will call all groups'
      f_encode(*obsN) and return the concatenation of their response.
    Technical remark:
    * Internally, objects will be managed non-hierarchically, but use a tuple as key, (group, name).
    """
    def __init__(self, N, rngseed=None):
        super().__init__(N=N, rngseed=rngseed)
        self._grp = dict()  # groupname : [list of keys in self._X]
        self._grp_order = [] # keeping track of which group goes where
        self._f_enc =  dict()
        self._A_grp = []  # while registering groups, we collect their _A's here (ordered)

    def register(self, group, f_encode, **kwargs):
        assert group not in self._grp, f"Error: {group} already registered."
        self._grp[group] = tuple(kwargs.keys())
        self._grp_order.append(group)
        assert callable(f_encode)
        self._f_enc[group] = f_encode
        augmentedKwargs = dict()
        for name, var in kwargs.items():
            grpname = (group, name)
            augmentedKwargs[grpname] = var
        self._A = []  # a new _A for the group
        super().register(_kwargsdict=augmentedKwargs)
        self._A_grp.append( np.array(self._A) )

    def _group_name_iter_wrapper(self, group, name, func):
        if group is not None:
            assert group in self._grp, f"Unknown group: {group}."
        if (group is None) and (name is None):
            ret = dict()
            for grp in self._grp:
                ret[grp] = dict()
                for name in self._grp[grp]:
                    grpname = (grp, name)
                    ret[grp][name] = func(grpname)
        elif (group is None) and (name is not None):
            ret = dict()
            for grp in self._grp:
                grpname = (grp, name)
                ret[grp] = func(grpname)
        elif (group is not None) and (name is None):
            ret = dict()
            for name in self._grp[group]:
                grpname = (group, name)
                ret[name] = func(grpname)
        elif (group is not None) and (name is not None):
            grpname = (group, name)
            ret = func(grpname)
        return ret

    def A(self, group=None, name=None, flat=True):
        func = lambda grpname: self._A_or_AT(self._A, grpname, flat)
        if (group is None) and (name is None):
            return func(None)
        return self._group_name_iter_wrapper(group, name, func)

    def AT(self, group=None, name=None, flat=True):
        func = lambda grpname: self._A_or_AT(self._AT.T, grpname, flat).T
        if (group is None) and (name is None):
            return func(None)
        return self._group_name_iter_wrapper(group, name, func)

    def _subclass_finalize(self):
        # build large A from block diagonals
        from scipy.linalg import block_diag
        self._A = block_diag(*self._A_grp)
        self.numGroups = len(self._grp)
        assert self._A.shape == (self._lenA, self.numGroups * self.N)

    def dec(self, r, group=None, name=None, flat=False):
        func = lambda grpname: self._dec_variable(r, grpname, flat)
        return self._group_name_iter_wrapper(group, name, func)

    def pseudoencode(self, **kwargs):
        raise NotImplementedError("Use encode method of LinearInputPopulationCode.")

    def enc(self, obsdict):
        r = np.zeros((self.numGroups, self.N))
        for grp, obs in obsdict.items():
            assert grp in self._grp, f"Unknown group: {grp}."
            gn = self._grp_order.index(grp)
            f_encode = self._f_enc[grp]
            # We refrain from checking the correct dimensions of obs; ensuring this will be left to the calling routine.
            r[gn] = f_encode(*obs)
        return r.flatten()

    def get_storable_dict(self):
        """Return a dict-represenation that can be serialized via JSON."""
        d = super().get_storable_dict()
        d.update(grp=turn_keys_into_str(self._grp), grp_order=self._grp_order)
        return d


if __name__ == "__main__":
    # Select one of the DEMOs by commenting out the other block
    # DEMO: LinearPopulationCode
    N = 100
    x1 = (3,4), lambda dim, N, rng: rng.rand(*(np.prod(dim),N))
    x2 = (4,5), np.random.normal(0, 1, (4,5,N))
    pc = LinearPopulationCode(N, rngseed=123)

    # DEMO: OneToOnePopulationCode
    # x1 = (3,4), lambda dim, rng: rng.rand(*dim)
    # x2 = (4,5), np.random.normal(0, 1, (4,5))
    # pc = OneToOnePopulationCode(rngseed=123)

    # Register and finalize the code
    pc.register(x1=x1, x2=x2)
    pc.finalize()

    # Try out en- and decoding
    x1example = np.linspace(-1, +1, np.prod(x1[0]))
    r = pc.pseudoencode(x1=x1example)
    x1dec = pc.dec(r, 'x1', flat=True)
    x2dec = pc.dec(r, 'x2', flat=True)
    assert np.allclose(x1dec, x1example)
    assert np.allclose(x2dec, 0.)


    # DEMO: LinearInputPopulationCode
    N = 100
    x1 = (3,4), lambda dim, N, rng: rng.rand(*(np.prod(dim),N))
    x1 = (3,4), np.random.normal(0, 1, (3,4,N))
    x2 = (4,5), np.random.normal(0, 1, (4,5,N))
    pc = LinearInputPopulationCode(N, rngseed=123)
    pc.register("grp_1", f_encode=lambda: None, x1=x1, x2=x2)
    pc.register( 2,      f_encode=lambda: None, x1=x1, x2=x2)
    pc.finalize()


    # Plot the matrices
    import pylab as pl
    fig = pl.figure(figsize=(10,4))
    axs = fig.subplots(1,3)
    # A
    m = axs[0].imshow(pc.A())
    pl.colorbar(m, ax=axs[0])
    # AT
    m = axs[1].imshow(pc.AT())
    pl.colorbar(m, ax=axs[1])
    # Identity
    m = axs[2].imshow(pc.A() @ pc.AT())
    pl.colorbar(m, ax=axs[2])



