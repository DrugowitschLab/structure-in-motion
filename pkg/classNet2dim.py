"""Class for the network model and for the input tuning functions. Assumes D=2 spatial dims."""

import numpy as np
from . import log, asciiL
from functools import lru_cache

from . import FilterStrinfAdiabaticDiagonalSigma
from . import LinearPopulationCode, OneToOnePopulationCode, LinearInputPopulationCode

from scipy.integrate import solve_ivp
default_integrator = lambda *args, **kwargs: solve_ivp(*args, method='RK45', **kwargs)

def make_MT_tuning_function(Na, Nr, psi, rhomin, rhomax, exponent, kappaa, sig2rho, returnFsingle=False):
    """
    Factory for creating MT-like neural responses which encode (v, sig2) with v=(vx,vy) and sig2 = sig_obs^2.
    Returns the tuning function and the corresponding readout weights in dict w/ keys 'vOverSig2' and 'oneOverSig2'.
    
    Parameters:
    
        Na : int                    num directional tuning centers in the polar grid (ranging from 0 to 2 pi) 
        Nr : int                    num radial tuning centers in the polar grid
        psi : float                 scaling of overall firing rate
        rhomin : float              min speed of radial tuning centers
        rhomax : float              max speed of radial tuning centers
        exponent : float            density non-linearity for radial tuning centers
        kappaa : float              width of angular tuning function (the kappa in von Mises)
        sig2rho: float              width of radial tuning (variance of the log-normal)
        
        returnFsingle : bool        Return f(alpha, rho, sig2, na, nr) for single-neurons instead (e.g., for plotting)

    Comments:
        * One such population will be created for each receptive field center.
    """
    sqrt, exp, log, cos, sin, pi = np.sqrt, np.exp, np.log, np.cos, np.sin, np.pi
    from scipy.special import iv
    I0 = lambda kappa: iv(0, kappa)
    # # #  Tuning centers; we describe the funcitons in polar coordinates  # # #
    da, dr = 2*pi/Na, (rhomax-rhomin)/(Nr-1)**exponent # distance of tuning between neighboring neurons (directional and radial respectively)
    mua = np.linspace(0, 2*pi, Na+1)[:-1]
    f_mur = lambda nr: rhomin + dr * nr**exponent
    f_mur_prime = lambda nr: dr * exponent * nr**(exponent-1)
    mur = f_mur(np.arange(Nr))
    # Two convenient meshgrids for elementwise functions
    Mua, Mur = np.meshgrid(mua, mur)
    NA, NR = np.meshgrid(np.arange(Na), np.arange(0,Nr))
    # # #  Neural response function with input in polar coordinates  # # #
    f = lambda alpha, rho, sig2, na, nr: f_sig(sig2) * f_alpha(alpha, na) * f_rho(rho, nr)
    f_sig = lambda sig2: psi / sig2
    f_alpha = lambda alpha, na: da/(2*pi*I0(kappaa)) * exp(cos(alpha - da*na) * kappaa)
    f_rho = lambda rho, nr: f_mur_prime(nr)/sqrt(2*pi*sig2rho)/f_mur(nr) * exp( - (log(rho) - log(f_mur(nr)))**2 / (2*sig2rho) )
    # Do we only want a basic function?
    if returnFsingle:
        return f
    # If not: build the function for full population
    # Coordinate transformation
    def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2)
        alpha = np.arctan2(y, x)
        return(alpha, rho)
    # # #  The actual neural response function  # # #
    def f_encode(v, sig2):
        alpha, rho = cart2pol(*v)
        return f(alpha, rho, sig2, NA, NR).flatten()
    # # #  Readout weights # # #
    N = Na * Nr
    W_oneOverSig2 = 1 / psi * np.ones((2,N))                  # readout for 1/sig^2  (formally for both dimesions x and y)
    W_vxOverSig2 = (cos(Mua) * Mur / psi).reshape((1,N))      # readout for v_x / sig^2
    W_vyOverSig2 = (sin(Mua) * Mur / psi).reshape((1,N))      # readout for v_y / sig^2
    W_vOverSig2 = np.vstack([W_vxOverSig2, W_vyOverSig2])
    return f_encode, dict(vOverSig2=W_vOverSig2, oneOverSig2=W_oneOverSig2)


def make_A_input_reorder(D, K):
    """
    Our canonical index ordering is shape = (D, K) --> flatten. However, for technical convenience,
    the neural input is encoded in order (K, D) --> flatten. This can be corrected by a permutation
    matrix P of shape (D*K, D*K). Then, in the neural interactions, we replace A by P@A and AT by AT@P.T.

    This function calculates P.
    """
    from scipy.linalg import block_diag
    P = block_diag(*[np.eye(D)]*K)
    return P.reshape(K,D,K,D).swapaxes(0,1).reshape(D*K,D*K)


def neural_variable_property(population, name, func=None, docstr=None):
    """
    A factory for neural activity-based properties.
    Example usage: mu = neural_variable_property('lin', 'mu', "The motion sources' estimated mean value.")
    """
    if func is None:
        func = lambda x: x

    def getter(self):
        if hasattr(self, "pc") and hasattr(self, "r"):
            return func(self.pc[population].dec(self.r[population], name=name, flat=True))
        else:
            log.debug(f"Ignoring attempt to access {name} in network before creation of population codes. Returned 'None'.")
            return None

    def setter(self, value):
        log.debug(f"Ignoring attempt to set {name} for network: use method 'init_filter' instead.")

    def deleter(self):
        log.debug(f"Ignoring attempt to delete {name} for network.")

    return property(getter, setter, deleter, doc=docstr)



class RateNetworkMTInputStrinfAdiabaticDiagonalSigma(FilterStrinfAdiabaticDiagonalSigma):
    """
    For this network implementation, we use the following variables.
    Distributed input code:
     * Velocity: v/sig^2 ((x,y) for k=1..K)
     * Uncertainty: 1/sig^2
    Distributed latent code:
     * Task variables: mu ((x,y) for m = 1..M), lam^2 (for m=1..M)
     * Auxiliary variables: predErr = (v/sig^2 - C 1/sig^2 mu) ((x,y) for k=1..K)
    One-to-one latent code:
     * Uncertainty: f_Sigma(lam^2)
    """
    bNETWORK = True   # indicate that this is a network-based filter
    # Some config in class variables (consider them immutable)
    feature_learning_method = None # No learning of features for now
    feature_learning_kwargs = None # No learning of features for now
    bOminvAssumeAllVisible = True  # No formal masking of invisible inputs (will be accommodated implicitly by 1/sig^2)

    def __init__(self, D, C, lam0, tau_s, sig_obs, inf_method, inf_method_kwargs=None, integrator=default_integrator, \
                polarIdx=None, RFCoord=None, \
                # New parameters, added for the network class
                inputNeuronKwargs=None,    # dict(Na, Nr, psi, rhomin, rhomax, exponent, kappaa, sig2rho)
                latentNeuronKwargs=None,   # dict(N, f_readout, rngseed, afSig, tau_pe)
                ):
        """Most arguments are inherited from FilterStrinfAdiabaticDiagonalSigma --> see doc there.
        
        Additional parameters:
        
            inputNeuronKwargs : dict            params for the MT tuning functions (see there)
            latentNeuronKwargs : dict           params for the population codes of latent vars (see comments)
            
        Comments:
            latentNeuronKwargs
                N : int                         number of neurons, make sure that there are more neurons than variables.
                f_readout : callable            function Af to generate linear readout matrices
                rngseed : int                   random seed for f_readout
                afSig : float or M array        readout weight for the one-to-one population
                tau_pe : float or D*K array     time constant for the prediction error (tau_epsilon in the theory)
        """
        assert D == 2, "Rate-based network implemenation assumes two spatial dimensions (D=2)."
        # TODO: set inf_method to inf_method_none (will be done automatically when propagating the neurons)
        #       but keep the inf_method_kwargs for use in  define_latent_computations()
        super().__init__(D=D, C=C, lam0=lam0, tau_s=tau_s, sig_obs=sig_obs, inf_method=inf_method, inf_method_kwargs=inf_method_kwargs, feature_learning_method=self.feature_learning_method, feature_learning_kwargs=self.feature_learning_kwargs, integrator=integrator, bOminvAssumeAllVisible=self.bOminvAssumeAllVisible, polarIdx=polarIdx, RFCoord=RFCoord)
        # remaining init
        # TODO: make this a configurable parameter?
        self.tau_pe = latentNeuronKwargs['tau_pe'] * np.ones(self.D*self.K)  # integration time constant for prediction error (should be at least as a fast as tau_s); len=D*K
        # Population codes
        self.pc = dict()
        # # #  Create input population code
        f_encode, W_input = make_MT_tuning_function(**inputNeuronKwargs)
        self.pc['inp'] = LinearInputPopulationCode(N=inputNeuronKwargs["Na"]*inputNeuronKwargs["Nr"])
        kwargs = { key : ((val.shape[0],), val) for key,val in W_input.items() }
        for k in range(self.K):
            self.pc['inp'].register(group=k, f_encode=f_encode, **kwargs)
        self.pc['inp'].finalize()
        # # #   Create latent population code
        # linear part
        self.pc['lin'] = LinearPopulationCode(N=latentNeuronKwargs['N'], rngseed=latentNeuronKwargs['rngseed'])
        self.pc['lin'].register( mu=((self.D,self.M), latentNeuronKwargs['f_readout']) )
        self.pc['lin'].register( lam2=((self.M), latentNeuronKwargs['f_readout']) )
        self.pc['lin'].register( predErr=((self.D,self.K), latentNeuronKwargs['f_readout']) )
        self.pc['lin'].finalize()
        # one to one part
        self.pc['one'] = OneToOnePopulationCode(rngseed=latentNeuronKwargs['rngseed'])
        self.pc['one'].register( fSig=((self.M,), latentNeuronKwargs['afSig'] * np.ones(self.M)) )
        self.pc['one'].finalize()
        # # #  Neural populations  # # #
        self.r = {'inp' : np.zeros(self.pc['inp'].A().shape[1]),          # Input neurons
                  'lin' : np.zeros(self.pc['lin'].A().shape[1]),          # Population encoding distributed variables
                  'one' : np.zeros(self.pc['one'].A().shape[1])}          # Population encoding 1-to-1 variables
        # # #  Assemble dynamics and neural computations  # # #
        # compose the dynamics of the latent variables
        self.latentComputations = self.define_latent_computations()
        # compose the dynamics in the neural domain
        self.latentNeuralComputations, self.neuralDynamics = self.assemble_neuralDynamics()
        # # #  M I S C  # # #
        # fixed values for |c_m|/sig^2 in f_Sig (Warning: here we assume that the observation noise in both spatial dims is equal.)
        self._c0norm = ( self.C0.T @ np.diag( 1/self.sig_obs[:self.K]**2 ) @ self.C0 ).diagonal()
        # Remove some variables of the parent class that are not needed for the network
        del self._lastVmask
        self._bVariableDomainWarningPrinted = False   # Will print a warning (once) if the system is propagated in the variable domain.

    def neural_response_fSig(self, r_in):
        """Returns the activity of the 1-to-1 population."""
        lam2 = self.pc['lin'].dec(r_in, name='lam2')  # linear readout
        taus = self.tau_s[:self.M]
        fSig = 1 / (taus * self._c0norm) * ( -1 + np.sqrt( 1 + taus**2 * self._c0norm * lam2 ) )
        AT = self.pc['one'].AT(name='fSig')
        r_out = AT @ fSig
        return r_out


    # # #  Make state variables neuron-based  # # #
    mu = neural_variable_property('lin', 'mu', docstr="The motion sources' estimated mean value.")
    Sig = neural_variable_property('one', 'fSig', docstr="The motion sources' estimated covariance.")
    lam2 = neural_variable_property('lin', 'lam2', docstr="Inferred motion strengths (squared).")
    lam = neural_variable_property('lin', 'lam2', func=lambda x: np.sqrt(np.maximum(0, x)), docstr="Inferred motion strengths.")
    # Compatibility function for base class. Could be removed in neural implementation.
    Lam = neural_variable_property('lin', 'lam2', func=lambda x: np.diag(np.tile(np.sqrt(np.maximum(0, x)), 2)), docstr="Compatibility function.")

    # Criple the method for inferring strengths: it becomes part of the neural dynamics
    def infer_strengths(self, *args, **kwargs):
        assert kwargs == {}, "The network does not support *default* parameters 'inf_method_kwargs'. Provide them as parameters during initialization!"
        # It's only remaining contribution is to read out and store the strengths.
        self.archive["lam"].append(self.lam.copy())
        return self.lam

    # Override init_filter method
    def init_filter(self, s=None):
        if s is None:
            s = np.zeros(self.D * self.M)
        assert len(s) == self.D * self.M, "ERROR: Init state s must be of length D*M !"
        mu = s.copy()
        lam2 = self.lam0**2
        # Set a compatible firing rate
        r = self.pc['lin'].pseudoencode(mu=mu, lam2=lam2)
        self.r['lin'][:] = r
        self.r['one'][:] = self.neural_response_fSig(r)
        # time is zero
        self.t_last = 0.
        # Start archive
        self.archive = dict(t=[self.t_last], lam=[self.lam])
        self.archive["mu"] = [self.mu.copy().reshape(self.D,self.M)]
        self.archive["Sig"] = [ np.diag(self.Sig) if len(self.Sig) == self.D*self.M else np.diag(np.tile(self.Sig, self.D)) ]     # Reset lambda
        if self.feature_learning_method is not None:
            self.archive["C"] = [self.C.copy()]
        # Network activity
        self.archive["r_inp"] = [self.r["inp"].copy()]
        self.archive["r_lin"] = [self.r["lin"].copy()]
        self.archive["r_one"] = [self.r["one"].copy()]
        return s


    def _get_val(self, pop, name, Z, v):
        """
        For development and debugging only.

        Helper function for _dZdt_variable_domain_wrapper
        """
        if pop == 'inp' and name == 'vOverSig2':
            return v/self.sig_obs**2
        elif pop == 'inp' and name == 'oneOverSig2':
            return 1/self.sig_obs**2
        elif pop == 'lin':
            return Z[name]
        elif pop == 'one' and name == 'fSig':
            taus = self.tau_s[:self.M]
            fSig = 1 / (taus * self._c0norm) * ( -1 + np.sqrt( 1 + taus**2 * self._c0norm * Z['lam2'] ) )
            return fSig
        else:
            raise Exception("Unknown variable.")


    def _dZdt_neural_domain_wrapper(self, t, Z):
        """
        Wrapper function for solve_ivp from scipy.integrate.

        We only integrate self.r['lin'].
        (self.r['inp'] has already been set, self.r['one'] is instantaneous and will be calculated on the fly.)
        """
        # 1) fetch r_inp and calculate r_one from r_lin (we use a local rate-dict to account for the hopping of the solver)
        r = { 'inp' : self.r['inp'],
              'lin' : Z,
              'one' : self.neural_response_fSig(Z),
            }
        # 2) iterate over the terms in self.neuralDynamics and sum them up
        # We only integrate the 'linear' population (this should be the only key in neuralDynamics)
        assert len(self.neuralDynamics.keys()) == 1
        dydt = np.zeros(Z.shape)
        for ctype,complist in self.neuralDynamics['lin'].items():
            if ctype == "constant":
                for c, in complist:
                    dydt += c
            elif ctype == "linear":
                for M,pop in complist:
                    dydt += M @ r[pop]
            elif ctype == "quadratic":
                for S,pop1,pop2 in complist:
                    r1, r2 = r[pop1], r[pop2]
                    dydt += np.dot( np.dot(S,r2) , r1 )
            else:
                raise Exception(f"Unknown computation type {ctype}.")
        # 3) return dr_lin / dt
        return dydt


    def _dZdt_variable_domain_wrapper(self, t, Z, v):
        """
        For development and debugging only.

        Z = concatenate( mu.flatten(), lam2.flatten(), predErr.flatten() )
        for the numerical integrator.
        """
        D, M, K = self.D, self.M, self.K
        Z = dict(mu = Z[:D*M],
                 lam2 = Z[D*M:D*M+M],
                 predErr = Z[-D*K:])
        dZdt = { k : np.zeros(v.shape) for k,v in Z.items()}
        for (_,target),complist in self.latentComputations.items():
            for comp in complist:
                ctype = comp[0]
                if ctype == "constant":
                    dZdt[target] += comp[1]
                elif ctype == "linear":
                    M = comp[1]
                    pop, name = comp[2]
                    x = self._get_val(pop, name, Z, v)
                    dZdt[target] += M @ x
                elif ctype == "quadratic":
                    S = comp[1]
                    pop, name = comp[2]
                    x = self._get_val(pop, name, Z, v)
                    pop, name = comp[3]
                    y = self._get_val(pop, name, Z, v)
                    dZdt[target] += np.dot( np.dot(S,y) , x )
        out = np.concatenate(( dZdt['mu'], dZdt['lam2'], dZdt['predErr'] ))
        return out


    def propagate_and_integrate_observation_NETWORK_DOMAIN(self, v, t):
        """
        This is the central propagator.
        """
        # 1) Some bureaucracy
        assert self.t_last is not None, " > ERROR: Initialize filter state first!"
        assert t >= self.t_last, " > ERROR: Negative interval since last observation!"
        # 2) Turn the observation v and sig2 into an MT response
        if isinstance(v, np.ma.core.MaskedArray):
            vmask = v.mask
        else:   # all umasked
            vmask = np.zeros(v.shape, dtype=bool)
        sig_multiplier = 1. + vmask * 1.e8        # 1 if unmasked, very large if masked
        sig2 = sig_multiplier * self.sig_obs**2
        obsdict = { k : (vk, sig2[k]) for k,vk in enumerate(v.reshape(self.D, self.K).T)}
        self.r['inp'] = self.pc['inp'].enc(obsdict)  # Here we set the MT activity
        # 3) Prepare and call the wrapper for numerical integration
        kwargs = dict(fun=self._dZdt_neural_domain_wrapper, args=None, y0=self.r['lin'].copy(), t_span=(self.t_last, t) )
        sol = self.integrator(**kwargs)
        if not sol.success:
            log.error("Numerical integration failed with message: " + sol.message)
        assert np.isclose(sol.t[-1], t), "Numerical integration did not reach requested time."
        self.r['lin'][:] = sol.y[:,-1]
        self.t_last = t
        # 4) Calculate the activity of the 1-to-1 population
        self.r['one'] = self.neural_response_fSig(self.r['lin'])
        # 5) Store state to archive
        self.archive["t"].append(t)
        # The following are read out from the neural activity
        self.archive["mu"].append( self.mu.copy().reshape(self.D,self.M) )
        self.archive["Sig"].append( np.diag(self.Sig) if len(self.Sig) == self.D*self.M else np.diag(np.tile(self.Sig, self.D)) )
        # We do not store lam here: it is done when infer_strengths() is called.
        # Store network activity
        self.archive["r_inp"].append( self.r["inp"].copy() )
        self.archive["r_lin"].append( self.r["lin"].copy() )
        self.archive["r_one"].append( self.r["one"].copy() )



    def propagate_and_integrate_observation_VARIABLE_DOMAIN(self, v, t):
        """
        For development and debugging only.

        We decode all variables from the neurons, then perform the integration in the variable domain
        based on self.latentComputations, and finally pseudoencode the result back into the neurons.
        """
        if not self._bVariableDomainWarningPrinted:
            log.warning("Working in variable domain! Not network domain!")
            self._bVariableDomainWarningPrinted = True
        assert self.t_last is not None, " > ERROR: Initialize filter state first!"
        assert t >= self.t_last, " > ERROR: Negative interval since last observation!"
        if isinstance(v, np.ma.core.MaskedArray):
            vmask = v.mask
        else:   # all umasked
            vmask = np.zeros(v.shape, dtype=bool)
        self._lastVmask = vmask  # Used for self.Sig
        # Call signature: self.MYDFDT(t, state, v, vmask)
        pc, r = self.pc['lin'], self.r['lin']
        y0 = np.concatenate([ pc.dec(r, name, flat=True) for name in ('mu', 'lam2', 'predErr') ])
        kwargs = dict(fun=self._dZdt_variable_domain_wrapper, args=(v,), y0=y0, t_span=(self.t_last, t) )
        sol = self.integrator(**kwargs)
        if not sol.success:
            log.error("Numerical integration failed with message: " + sol.message)
        assert np.isclose(sol.t[-1], t), "Numerical integration did not reach requested time."
        Z = sol.y[:,-1]
        D, M, K = self.D, self.M, self.K
        Z = dict(mu = Z[:D*M],
                 lam2 = Z[D*M:D*M+M],
                 predErr = Z[-D*K:])
        # Set a compatible firing rate
        r = self.pc['lin'].pseudoencode(mu=Z['mu'], lam2=Z['lam2'], predErr=Z['predErr'])
        self.r['lin'][:] = r
        self.t_last = t
        # store results
        self.archive["t"].append(t)
        self.archive["mu"].append( self.mu.copy().reshape(self.D,self.M) )
        self.archive["Sig"].append( np.diag(self.Sig) if len(self.Sig) == self.D*self.M else np.diag(np.tile(self.Sig, self.D)) )

    # # #  Selection of the integration method (VARIABLE_DOMAIN is for development only) # # #
    # propagate_and_integrate_observation = propagate_and_integrate_observation_VARIABLE_DOMAIN
    propagate_and_integrate_observation = propagate_and_integrate_observation_NETWORK_DOMAIN


    # # #  Here we define the computational dynamics of the system  # # #
    def define_latent_computations(self):
        """
        Return a dictionary of all latent dynamics in the domain of variables. It is implemented as
        a method to grant access to instance-specific parameters, e.g., time constants tau and the C-matrix.
        """
        # extract some params for easier access
        d = self.inf_method_kwargs
        tau_lam, nu, kappa = d['tau_lam'], d['nu'], d['kappa']
        # The tensor in d_t mu
        CTtensor = np.zeros((self.D*self.M, self.M, self.D*self.K))
        for i,j,k in np.ndindex(CTtensor.shape):
            # combines two mappings: C.T @ predErr, and broadcasting f_Sig from M --> D*M
            CTtensor[i,j,k] = (i % self.M == j) * self.C.T[i,k]
        # hyperprior on lam2
        taus = self.tau_s[:self.M]
        lam2Const = (self.D * taus/2)**(-1) * ( taus/2 * nu * np.sign(kappa) * kappa**2 ) / ( nu + tau_lam/taus + 2/self.D )
        # mu^2 and Sig^2 integration for d_t lam^2
        prefactor = (self.D * self.tau_s/2)**(-1) * ( tau_lam/self.tau_s ) / ( nu + tau_lam/self.tau_s + 2/self.D )
        mu2Tensor = np.zeros((self.M, self.D*self.M, self.D*self.M))
        for i,j,k in np.ndindex(mu2Tensor.shape):
            # multiplies mu_d,m * mu_d,m and sums over spatial dimensions d
            mu2Tensor[i,j,k] = prefactor[i] * (i == j % self.M) * (j == k)
        fSigMatrix = np.zeros((self.M, self.M))
        for i,j in np.ndindex(fSigMatrix.shape):
            # sums over spatial dimensions d simply by multiplying with self.D
            fSigMatrix[i,j] = prefactor[i] * self.D * (i == j)
        # predErrCtensor
        predErrCtensor = np.zeros((self.D*self.K, self.D*self.K, self.D*self.M))
        for i,j,k in np.ndindex(predErrCtensor.shape):
            # subtracts the expected v/sig^2
            predErrCtensor[i,j,k] = -1/self.tau_pe[i] * (i == j) * self.C[j,k]
        # # #  THE DEF OF LATENT DYNAMICS (variable domain)  # # #
        latentComputations = {
        # Format: (CodeKey, CodeName) : (term1 in d_t latVar, term2..., )
        # each term one of the following:
        #   * [ "constant", Vector ]
        #   * [ "linear", Matrix, Arg1 ]
        #   * [ "quadratic", Tensor, Arg1, Arg2 ]
        # Arg1/2: ( CodeKey eg 'inp', CodeName eg 'vOverSig2' )
        # The operations automatically broadcast across objects k=1..K, sources m=1..M,
        # and spatial dimensions d=1..D. For input vars, we assume (K,D)-ordering (see above),
        # for latent vars we assume (D,M)-ordering (or (D,K) for the prediction error).
        ('lin', 'mu')   : (["linear", np.diag(-1/self.tau_s), ('lin', 'mu')],
                           ["quadratic", CTtensor, ('one', 'fSig'), ('lin', 'predErr')]),
        ('lin', 'lam2') : (["linear", -1/tau_lam * np.eye(self.M), ('lin', 'lam2')],
                           ["constant",  lam2Const/tau_lam],
                           ["quadratic", mu2Tensor/tau_lam, ('lin', 'mu'), ('lin', 'mu')],
                           ["linear", fSigMatrix/tau_lam, ('one', 'fSig') ]),
        ('lin','predErr'):(["linear", np.diag(-1/self.tau_pe), ('lin', 'predErr')],
                           ["linear", np.eye(self.D*self.K) / self.tau_pe, ('inp', 'vOverSig2')],
                           ["quadratic", predErrCtensor, ('inp', 'oneOverSig2'), ('lin', 'mu')]),
        }
        return latentComputations


    def _fetch_Asrc(self, srcpop, srcvar):
        """
        Little helper function, taking care of input index-order and so on.
        """
        Asrc = self.pc[srcpop].A(name=srcvar)
        if srcpop == 'inp':
            Asrc = np.vstack([Asrc[k] for k in range(self.K)])  # Stitch all visibles together
            Asrc = make_A_input_reorder(self.D, self.K) @ Asrc  # correct for the ordering of input indices?
        return Asrc


    def assemble_neuralDynamics(self):
        """
        Take the decoding matrices A and AT from self.pc to lift self.latentComputations into the neural domain.
        """
        latentNeuralComputations = {}
        for (tarpop,tarvar),complist in self.latentComputations.items():
            if not tarpop in latentNeuralComputations:
                latentNeuralComputations[tarpop] = {"constant" : [], "linear" : [], "quadratic" : []}
            ATtar = self.pc[tarpop].AT(tarvar)   # target's adjoint matrix for re-encoding
            for comp in complist:
                ctype = comp[0]
                if ctype == "constant":
                    c = comp[1]
                    # Here's neural computation
                    cneural = ATtar @ c
                    latentNeuralComputations[tarpop]["constant"].append( (cneural, ) )
                elif ctype == "linear":
                    M = comp[1]
                    srcpop, srcvar = comp[2]
                    Asrc = self._fetch_Asrc(srcpop, srcvar)
                    # Here's neural computation
                    Mneural = ATtar @ M @ Asrc
                    latentNeuralComputations[tarpop]["linear"].append( (Mneural, srcpop) )
                elif ctype == "quadratic":
                    S = comp[1]
                    srcpop1, srcvar1 = comp[2]
                    Asrc1 = self._fetch_Asrc(srcpop1, srcvar1)
                    srcpop2, srcvar2 = comp[3]
                    Asrc2 = self._fetch_Asrc(srcpop2, srcvar2)
                    # Here's neural computation (unrolled)
                    Sneural = np.tensordot(ATtar, S, axes=([1],[0]))
                    Sneural = np.tensordot(Sneural, Asrc1, axes=([1],[0]))  # this appends an axis which we have to swap
                    Sneural = Sneural.swapaxes(1,2)
                    Sneural = np.tensordot(Sneural, Asrc2, axes=([2],[0])) # Here the appending is at the right spot
                    latentNeuralComputations[tarpop]["quadratic"].append( (Sneural, srcpop1, srcpop2) )

        # Conceptually, we are done. For beauty, we will sum the matrices which use the same arguments.
        neuralDynamics = {}
        for tarpop in latentNeuralComputations:
            neuralDynamics[tarpop] = {}
            for ctype in latentNeuralComputations[tarpop]:
                neuralDynamics[tarpop][ctype] = []
                complist = latentNeuralComputations[tarpop][ctype]
                for comp in complist:
                    M, args = comp[0], comp[1:]
                    ndel = 0
                    # For each candidate computation
                    for i in range(len(complist)):
                        candcomp = complist[i-ndel]
                        if (candcomp[1:] == args): # same args?
                            if np.all(M == candcomp[0]):
                                continue  # only different computations
                            M += candcomp[0]
                            # Because comp is of type list, we can list.pop() identical computations.
                            complist.pop(i-ndel)
                            ndel += 1  # correct listlength
                    neuralDynamics[tarpop][ctype].append( (M,) + args )

        return latentNeuralComputations, neuralDynamics


