"""
Classes for velocity-observing filters with D spatial dimentions.
    The "logical" format for indexing is X[t,d,m] for time 't', spatial dimension 'd',
    and motion source 'm'. However, for the implementation, we use X.reshape(T, D*M)
    to obtain efficient matrix equations.
"""

import numpy as np
from . import log, asciiL
from functools import lru_cache

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # #     MULTI-DIM. COMPONENT MATRIX   # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # #

def build_C_matrix(C0, D=1, polarIdx=None, RFCoord=None, bSFM=False):
    """
    Extend the C matrix to multiple dimensions, potentially
    with polar (radial + angular) motion sources. If polar
    sources are used, we require D==2 and the (K x 2) array
    RFCoord provides the receptive field locations [(R_k, vartheta_k),..]
    The indices of sources that are polar is provided in polarIdx
    (one idx per radial-angular pair).
    If bSFM==True, polar sources and their RFs describe a expansion or rotation
    around a vertical axis in 3D (for "structure-from-motion" experiments).
    """
    # Simple case (will be patched if polarIdx not None)
    from scipy.linalg import block_diag
    C = block_diag(*[C0 for d in range(D)]).astype(float)
    if polarIdx is None:
        return C
    # PolarIdx
    polarIdx = np.array(polarIdx)
    RFCoord = np.array(RFCoord)
    K,M = C0.shape
    assert D == 2
    assert (polarIdx < M).all()
    assert RFCoord.shape == (K,2)
    assert (0 < RFCoord[:,0]).all()
    assert (-np.pi <= RFCoord[:,1]).all() and (RFCoord[:,1] < 2*np.pi).all()
    for m in polarIdx:
        # clear the naively built components
        for d in range(D):
            C[:,d*M + m] = 0.
        # By convention, we put the radial source at index m and the angular source at M+m
        for k,(R,th) in enumerate(RFCoord):
            # 'normal' rotation in 2D
            if bSFM is False:
                # v_x of radial
                C[k,m] = C0[k,m] * np.cos(th)
                # v_x of angual
                C[k,M+m] = C0[k,m] * (- R * np.sin(th))
                # v_y of radial
                C[K+k,m] = C0[k,m] * np.sin(th)
                # v_y of angular
                C[K+k,M+m] = C0[k,m] * R * np.cos(th)
            # rotation around upward-directing y-axis (x points to the right; z towards the observer)
            # observed are only the (x,y) coordinates, no depth percpetion.
            elif bSFM is True:
                # v_x of radial
                C[k,m] = C0[k,m] * np.cos(th)
                # v_x of angual
                C[k,M+m] = C0[k,m] * (- R * np.sin(th))
                # v_y of radial
                C[K+k,m] = 0.
                # v_y of angular
                C[K+k,M+m] = 0.
                
                
    return C


# # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # #   W O R L D   S I M U L A T O R   # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # #

class WorldBase(object):
    """Base class for generating stimuli."""
    def __init__(self, D, C, lam, dt, seed, Mrot=None, polarIdx=None, RFCoord=None, bSFM=False):
        """Parameters:
            D : int                 spatial dimensions
            C : K x M matrix        1-dim C matrix (it will automatically be extended to all D dims)
            lam : M-array           ground truth lambda values
            dt : floating           integration time step (can be finer than observation times)
            seed : int              random seed
            Mrot : D x D matrix     (Optional) rotate the stimulus in the spatial dims
            polarIdx : tuple        (Optional) which indices m describe polar (= radial+angular) sources
            RFCoord : K x 2 array   (If polarIdx) receptive field centers [(R_k, vartheta_k),..]
        """
        self.D = D
        self.K, self.M = C.shape
        self.C0 = C                                 # store the 1-dim version
        self.C = build_C_matrix(self.C0, D=D, polarIdx=polarIdx, RFCoord=RFCoord, bSFM=bSFM)
        self.set_lam(lam)
        self.dt = dt
        self.rng = np.random.RandomState(seed)
        if Mrot is not None:     # optional rotation matrix
            assert Mrot.shape == (self.D, self.D), "Rotation matrix must operate on spatial dims."
            assert polarIdx is None, "Rotation currently does not support polar coordinates (Todo: swap order of C@M to M@C in get_v)"
            self.Mrot0 = Mrot
            from scipy.linalg import block_diag
            self.Mrot = block_diag(*[Mrot for m in range(self.M)]).reshape(self.M,-1).T.reshape(self.M*self.D,-1)   # fancy way of distributing Mrot0 to all sources m
        else:
            self.Mrot = np.eye(self.M * self.D)
        self.S = None
        log.info(f"Created {self.__class__} with lam={self.lam}")
        log.info(f"This leads to motion structure matrix\n{asciiL(self.get_L(), 4)}")

    def get_L(self):
        return self.C0 @ np.diag(self.lam)

    def get_v(self, s):
        if s.ndim == 1 and len(s) == self.M:
            return self.C0 @ s
        elif s.ndim == 1 and len(s) == self.D * self.M:
            return self.C @ self.Mrot @ s
        elif (s.ndim == 2) and (s.shape[0] == self.M):
            return self.C0 @ s
        elif (s.ndim == 2) and (s.shape[0] == self.D * self.M):
            return self.C @ self.Mrot @ s
        elif (s.ndim == 2) and (s.shape[1] == self.M):
            return (self.C0 @ s.T).T
        elif (s.ndim == 2) and (s.shape[1] == self.D * self.M):
            return (self.C @ self.Mrot @ s.T).T
        else:
            raise Exception(" > Shape of s does not match component matrix C.")

    def set_lam(self, lam):
        assert len(lam) == self.M
        self.lam = lam
        self.Lam = np.diag(np.tile(self.lam, self.D))

    def index_of_t(self, t):
        tn = np.round(t/self.dt).astype(int)
        return tn

    def get_times(self):
        assert self.S is not None, " > ERROR: Draw trajectory first!"
        Tn = self.S.shape[0]
        return self.dt * np.arange(Tn)

    def draw_trajectory(self, T):
        """To be implemented by subclasses."""
        raise NotImplementedError
        return self.S


class WorldOUSources(WorldBase):
    """Creates noise-free stimuli accoring to the ground truth of the generative model.
    
    Each motion sources s_d,m(t) follows Ornstein Uhlenbeck process
          ds/dt = -s/tau_s + lam_m * dW
    """
    def __init__(self, D, C, lam, tau_s, dt, seed, Mrot=None, polarIdx=None, RFCoord=None):
        """See class WorldBase for parameter description.
        
        Additional parameters:
        
            tau_s : float           time constant of s
        """
        super().__init__(D, C, lam, dt, seed, Mrot, polarIdx, RFCoord)
        if isinstance(tau_s, (float, np.floating)):
            tau_s = tau_s * np.ones(self.M * self.D)
        assert isinstance(tau_s, np.ndarray), " > ERROR: tau_s must be ndarray or float."
        self.tau_s = tau_s
        if np.any(self.tau_s < 5. * self.dt):
            log.warning(" > WARNING: Coarse world simulation time step in units of tau_s! Object trajectories may be inaccurate!")
        # Equations of motion
        self.F = -np.diag(1./self.tau_s)
        sqrtdt = np.sqrt(self.dt)
        dW = lambda: self.rng.normal(0., 1., self.D * self.M)
        self.ds = lambda s: self.F @ s * self.dt + sqrtdt * self.Lam @ dW()

    def draw_initial_state(self):
        # Independent motion sources
        s = self.rng.multivariate_normal([0]*(self.D*self.M), self.tau_s/2 * self.Lam**2 )
        return s

    def draw_trajectory(self, T):
        Tn = int(round(T / self.dt))
        S = np.zeros((Tn+1, self.D*self.M))
        S[0] = self.draw_initial_state()
        for tn in range(Tn):
            s = S[tn]
            S[tn+1] = s + self.ds(s)
        self.S = S
        return S


class WorldOUSourcesLamlist(WorldOUSources):
    """Extends class WorldOUSources with a list of lambdas:
       lamList = [ (t0, lam0), (t1, lam1), ... ]
       such that lam = lamN for tN < t < tN+1.
    """
    def __init__(self, D, C, lamList, tau_s, dt, seed, Mrot=None, polarIdx=None, RFCoord=None):
        for t,l in lamList:
            assert isinstance(t, (float, np.floating))
            assert len(l) == C.shape[1]
        self.lamList = lamList
        super().__init__(D, C, lamList[0][1], tau_s, dt, seed, Mrot, polarIdx, RFCoord)

    def draw_trajectory(self, T):
        Tn = int(round(T / self.dt))
        tnSwitch = [self.index_of_t(t) for t,l in self.lamList]
        S = np.zeros((Tn+1, self.D*self.M))
        S[0] = self.draw_initial_state()
        for tn in range(Tn):
            if tn in tnSwitch:
                idx = tnSwitch.index(tn)
                self.set_lam(self.lamList[idx][1])
            s = S[tn]
            S[tn+1] = s + self.ds(s)
        self.S = S
        return S


class WorldOscillator(WorldBase):
    """Each motion sources s_d,m(t) follows a cosine oscillation
          ds/dt = lam * (2 * pi * freq) * cos(2 * pi * freq * t + phase)
       accellerating the dots. This leads to amplitude max(s(t)) = lam.
       Optionally, an initial state s0 can be given.
    """
    def __init__(self, D, C, lam, freq, phase, dt, seed, s0=0., Mrot=None, polarIdx=None, RFCoord=None):
        super().__init__(D, C, lam, dt, seed, Mrot, polarIdx, RFCoord)
        if isinstance(freq, (float, np.floating)):
            freq = freq * np.ones(self.D * self.M)
        assert len(freq) == self.D * self.M
        self.freq = freq
        if isinstance(phase, (float, np.floating)):
            phase = phase * np.ones(self.D * self.M)
        assert len(phase) == self.D * self.M
        self.phase = phase
        if isinstance(s0, (float, np.floating)):
            s0 = s0 * np.ones(self.D * self.M)
        assert len(s0) == self.D * self.M
        self.s0 = s0
        # Equation of motion
        self.ds = lambda t: self.dt * self.Lam @ np.cos(2 * np.pi * self.freq * t + self.phase) * (2 * np.pi * self.freq)

    def draw_trajectory(self, T):
        Tn = int(round(T / self.dt))
        times = self.dt * np.arange(Tn)
        S = np.zeros((Tn+1, self.D*self.M))
        S[0] = self.s0
        for tn,t in enumerate(times):
            s = S[tn]
            S[tn+1] = s + self.ds(t)
        self.S = S
        return S


class WorldVeloFunction(WorldBase):
    """Generate the velocity trajectory v(t) directly from a function f_v -> shape (D,K).
    
    This is mostly a 'hack' to support manually defined trajectories within the established software stack.
    
    Inner working:
        Will set self.S[tn] = f_v(tn*dt).flatten() and return v(t)=S(t).
        M is given for creating states compatibile with remaining software package and should match the filter's M.
    """
    def __init__(self, M, dt, f_v, seed, Mrot=None, polarIdx=None, RFCoord=None):
        D, K = np.array(f_v(0.)).shape
        # The C matrix will be identity on m=1..K, and zero otherwise
        C = np.zeros((K,M))
        C[:K,:K] = np.eye(K)
        lam = np.zeros(M)
        # No polarIdx: This is taken care of by f_v
        polarIdx = None
        super().__init__(D, C, lam, dt, seed, Mrot, polarIdx, RFCoord)
        if seed is not None:
            # provide a private rng to f_v
            self.f_v = lambda *args, **kwargs: f_v(*args, rng=self.rng, **kwargs)
        else:
            self.f_v = f_v

    def draw_trajectory(self, T):
        Tn = int(round(T / self.dt))
        S = np.zeros((Tn+1, self.D*self.M))
        for tn in range(Tn+1):
            s = np.zeros((self.D, self.M))
            s[:,:self.K] = self.f_v(self.dt*tn)
            S[tn] = s.flatten()
        self.S = S
        return S


# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #   O B S E R V A T I O N   G E N E R A T O R   # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class ObservationGeneratorVelo(object):
    """Observation generator turning the noise-free trajectories from class World... into noisy observations.
    
    Suggested usage: for ObservationGeneratorVelo obs, call obs.run_sim_and_generate_observations(T, wld)
        to generate noisy observations from a trajectory of duration T from World wld. Use obs.get_times()
        to get the observation times.
    """
    
    def __init__(self, dt, seed, sig_obs, final_precise, tInvisible=None, idxInvisible=None):
        """Parameters:
        
            dt : float                          inter-observation interval (inverse frame rate)
            seed : into                         random seed for the noise
            sig_obs : float or D*K array        observation noise (see comment below)
            final_precise : bool                Do not apply noise to observation (see comment below)
            tInvisible : float                  (Optional) For t > tInvisible, the objects with 
            idxInvisible : tuple                index in idxInvisible will be masked (hidden) in all dims
            
        Comments:
            * Observation noise sig_obs is given in dt-independent manner. The actually applied noise
              will be sig_obs/sqrt(dt). Further, sig_obs can be either of type float, or a np.ndarray of shape (D*K, )
            * final_precise could be used in decision tasks to prevnet that errors in the decision are mainly
              driven by random noise at the trial's end.
        """
        self.rng = np.random.RandomState(seed)
        self.sig_obs = sig_obs
        self.dt = dt
        self.sig_obs_eff = sig_obs/np.sqrt(dt)          # Keep the information per time independent of dt
        self.gen_v = lambda v_noiseless: self.rng.normal(v_noiseless, self.sig_obs_eff)
        self.final_precise = final_precise
        self.gen_v_precise = lambda v_noiseless: v_noiseless
        self.tInvisible = tInvisible
        if tInvisible is not None:
            assert idxInvisible is not None, "Must give list of indices to make invisible."
            self.idxInvisible = np.array(idxInvisible)
        self.T = None
        self.V = None
        log.info(f"Created {self.__class__} with dt={self.dt}")

    def run_sim_and_generate_observations(self, T, wld):
        wld.draw_trajectory(T)
        V = self.generate_observations(T, wld)
        return V

    def get_times(self):
        assert self.T is not None, " > ERROR: Generate observations first!"
        eps = 1.e-10
        return np.arange(0., self.T+eps, self.dt)

    def generate_observations(self, T, wld):
        self.T = T
        t = self.get_times()
        tn = wld.index_of_t(t)
        V_noiseless = wld.get_v(wld.S[tn])
        V = np.array( [self.gen_v(v) for v in V_noiseless] )
        if self.final_precise:
            V[-1] = self.gen_v_precise(V_noiseless[-1])
        # masking invisible velocities
        if self.tInvisible is not None:
            log.info(f"  Masking {self.idxInvisible} invisible for t >= {self.tInvisible:.2f}s")
            from numpy import ma
            D, K = wld.D, wld.K
            mask = np.array([[False]*(D*K)]*len(tn), dtype=bool)
            idxInvisible = np.array([self.idxInvisible + K*d for d in range(D) ]).flatten()
            mask[t > self.tInvisible, idxInvisible[:,None]] = True
            V = ma.masked_array(V, mask)
        self.V = V
        return V

    def get_noiseless_locations(self, wld, x0=0.):
        """This is mostly for visualization."""
        t = self.get_times()
        tn = wld.index_of_t(t)
        V_noiseless = wld.get_v(wld.S[tn])
        X = x0 + self.dt * np.cumsum(V_noiseless, axis=0)
        return X

    def get_noiseless_velocities(self, wld, x0=0.):
        """This is mostly for visualization."""
        t = self.get_times()
        tn = wld.index_of_t(t)
        V_noiseless = wld.get_v(wld.S[tn])
        return V_noiseless



# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # #         F I L T E R S         # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from scipy.integrate import solve_ivp
default_integrator = lambda *args, **kwargs: solve_ivp(*args, method='RK45', **kwargs)

class FilterStrinfVeloBase(object):
    """Base class for filtering with structure inference on velocity observations.
    
    This is the base class for all observer models, including the neural network implementation.
    
    Suggested usage: for Filter fil
        1. fil.init_filter() at the beginning of the trial
        2a. feed with observations v at time t via fil.propagate_and_integrate_observation(v, t)
        2b. (optional) Update lambda via fil.infer_strengths(**default_inf_method_kwargs)
        
        In 2b., default_inf_method_kwargs is a default dict for all used filters. It's arguments are
        superseded by individual kwargs given to the filter (if present).
    """
    def __init__(self, D, C, lam0, tau_s, sig_obs, inf_method, inf_method_kwargs=None, feature_learning_method=None, feature_learning_kwargs=None, integrator=default_integrator, polarIdx=None, RFCoord=None, bSFM=False):
        """Parameters:
        
            D : int                         spatial dims
            C : K x M matrix                1-dim C matrix (it will automatically be extended to all D dims)  
            lam0 : M-array                  initial values for lambda
            tau_s : float or D*M array      time constant tau_s for motion source inference (E-Step)
            sig_obs : float or D*K array    observation noise, dt-independent (see comments below)
            inf_method : callable           method used for inferring lambda^2 (see functions at end of file)
            inf_method_kwargs : dict        (optional) kwargs used by this filter (overrides default_inf_method_kwargs
                                            given in the experiment description)
            feature_learning_method : callable      (experimental) method for learning the C matrix
            feature_learning_kwargs : dict          (experimental) kwargs for C learning
            integrator : callable           (optional) numerical integrator, default: scipy's Runge-Kutta implementation
            polarIdx : tuple                (optional) which indices m describe polar (= radial+angular) sources
            RFCoord : K x 2 array           (if polarIdx) receptive field centers [(R_k, vartheta_k),..]
            bSFM : bool                     (optional) polar C-matrix for structure-from-motion stimuli
            
        Comments:
            * The C-matrix will usually match the one given to the World class, but this is not necassary: The observer
              will perform inference within the feature space that it is endowed with.
            * Contrary to the theory, we treat sigma_obs as a constant in the observer models. In the network, however,
              it will be an input variable which is conveyed by the input population.
        """
        self.D = D
        self.K, self.M = C.shape
        self.C0 = C                                 # store the 1-dim version
        # from scipy.linalg import block_diag
        # self.C = block_diag(*[C for d in range(self.D)])
        self.C = build_C_matrix(self.C0, D=D, polarIdx=polarIdx, RFCoord=RFCoord, bSFM=bSFM)
        self.RFCoord = RFCoord
        self.bSFM = bSFM
        self.lam0 = lam0.copy()                         # keep for re-init
        self.set_lam(lam0)
        if isinstance(tau_s, (float, np.floating)):
            tau_s = tau_s * np.ones(self.D*self.M)
        assert isinstance(tau_s, np.ndarray), " > ERROR: tau_s must be ndarray or float."
        self.tau_s = tau_s
        assert isinstance(sig_obs, (float, np.floating, np.ndarray)), " > ERROR: sig_obs must be float or ndarray of shape (D*K)."
        if isinstance(sig_obs, np.ndarray):
            assert sig_obs.shape == (self.D*self.K,), " If sig_obs is ndarray, it must have shape (D*K). That is: shape (D,K).flatten()."
        self.sig_obs = sig_obs
        assert callable(inf_method), f"'{inf_method.__name__}' is not a function."
        self.inf_method = lambda *args, **kwargs: inf_method(self, *args, **kwargs)
        self.inf_method_kwargs = dict() if inf_method_kwargs is None else inf_method_kwargs
        if feature_learning_method is None:
            self.feature_learning_method = None
        else:
            assert callable(feature_learning_method),f"'{feature_learning_method.__name__}' is not a function."
            self.feature_learning_method = lambda *args, **kwargs: feature_learning_method(self, *args, **kwargs)
        self.feature_learning_kwargs = dict() if feature_learning_kwargs is None else feature_learning_kwargs
        self.integrator = integrator
        self.t_last = None

    def set_lam(self, lam):
        assert len(lam) == self.M
        self.lam = lam
        self.Lam = np.diag(np.tile(self.lam, self.D))
        if hasattr(self, "lam2"):   # if present, update the square-cache
            self.lam2 = self.lam**2

    def infer_strengths(self, *args, **kwargs):
        # 1) Call the inf_method --> lam_new
        kwargs = kwargs.copy()                      # Don't mess with the default kwargs
        kwargs.update(self.inf_method_kwargs)       # The individual "self.inf_method_kwargs" have higher priority
        lamInf = self.inf_method(*args, **kwargs)
        # 2) Save the new lam to all internal variables
        self.set_lam(lamInf)
        # 3) Save new lam to archive
        self.archive["lam"].append(self.lam.copy())
        return self.lam

    def learn_motion_features(self, v, *args, **kwargs):
        kwargs.update(self.feature_learning_kwargs)
        # We work on the extended C matrix directly, to see that coherent features can emerge,
        # and that even polar coordinate-features can be learned.
        C = self.feature_learning_method(v, *args, **kwargs)
        # Update all self.C and dependent functions, e.g., call f.cache_clear() for lru_cached functions.
        self._set_C_dependent_functions(C)
        # Archive
        self.archive["C"].append(self.C.copy())

    def _set_C_dependent_functions(self, C):
        """This is just a template."""
        raise NotImplementedError
        self.C[:] = C
        # Update other dependent functions,  e.g., call f.cache_clear() for lru_cached functions.
        if self.feature_learning_method is not None:
            self.f_helper.cache_clear()


    def _init_filter_base(self, s=None):
        # Use s in derived representation
        if s is None:
            s = np.zeros(self.D * self.M)
        assert len(s) == self.D * self.M, "ERROR: Init state s must be of length D*M !"
        # Reset lambda
        self.set_lam(self.lam0.copy())
        # Some inference methods operate on lam**2; delete their variable
        if hasattr(self, "lam2"):
            delattr(self, "lam2")
        # time is zero
        self.t_last = 0.
        self.archive = dict(t=[self.t_last], lam=[self.lam])
        if self.feature_learning_method is not None:
            self.archive["C"] = [self.C.copy()]
        return s

    def init_filter(self, s=None):
        """This is just a template."""
        raise NotImplementedError
        s = super()._init_filter_base(s)
        self.archive["MYSTATE"] = ["self.INITIAL STATE"]


    def propagate_and_integrate_observation(self, v, t):
        """This is just a template."""
        raise NotImplementedError
        assert self.t_last is not None, " > ERROR: Initialize filter state first!"
        assert t >= self.t_last, " > ERROR: Negative interval since last observation!"
        # Call signature: self.MYDFDT(t, state, v)
        kwargs = dict(fun="self.MYDFDT", args=(v,), y0="self.MYSTATE", t_span=(self.t_last, t) )
        sol = self.integrator(**kwargs)
        if not sol.success:
            log.error("Numerical integration failed with message: " + sol.message)
        assert np.isclose(sol.t[-1], t), "Numerical integration did not reach requested time."
        self.MYSTATE = sol.y[:,-1]
        self.t_last = t
        # store results
        self.archive["t"].append(t)
        self.archive["MYSTATE"].append( "self.MYSTATE".copy().reshape(self.D,self.M) )
        
        
    def _assign_best_matching_inputs(self, v, sticky=0.70, randomAssign=False):  # 0.7
        """
        Helper function for structure-from-motion experiments.
          (1) Find matching receptive fields (from self.RFCoord)
          (2) Find permutation P such that |C*mu - P*v| is minimal.
              But, with probability 'sticky', keep the previous assignment.
          (3) return P*v
        """
        if not( self.bSFM and self.RFCoord ):
            log.error("Function '_assign_best_matching_inputs' requires bSFM and RFCoord to be given!")
        # (1) find matchin RFs (leaving out the vestibular input)
        xprojection = np.array([ rk*np.cos(phik) for (rk, phik) in self.RFCoord[:-1] ])
        groups = []
        for k,xk in enumerate(xprojection):
            idx = np.isclose(xk, xprojection).nonzero()[0]
            if (idx >= k).all():  # no duplicate groups
                groups.append(idx)
        # (2) Assignment
        if not hasattr(self, "_SFM_last_assignment"):
            # assignments are index sets within the group; e.g. g = (2,5,9), assign = [1,0,2], g[assign] = (5,2,9)
            # initially: None = unassigned
            self._SFM_last_assignment = [ None for g in groups ]
        vexpect = (self.C @ self.mu).reshape(self.D, self.K)
        vin = v.reshape(self.D, self.K)
        vout = np.zeros((self.D, self.K))   # init empty
        vout[:,-1] = vin[:,-1]              # Copy vestibular input
        for gi,(g,la) in enumerate(zip(groups, self._SFM_last_assignment)):
            # Group-size == 1? --> only one assignment possible
            if len(g) == 1:
                vout[:,g[0]] = vin[:,g[0]]
                log.warning(f"Found singluar group {g} during SFM. Make sure that groups are correctly matched.")
                continue
            # Stickyness triggers? --> keep last assignment
            # Remark: we use numpy's general rng because the otherwise-deterministic algorithm has no private rng.
            if la is not None and np.random.rand() < sticky:
                vout[:,g] = vin[:,g[la]]
                continue
            # # # Group is not sticky or not yet assigned? --> Let's do the assignment  # # #
            from itertools import permutations
            Perms = [ list(pi) for pi in permutations(range(len(g))) ]  # We turn them to lists, so they work for numpy indexing
            D = np.zeros(len(Perms))   # init distances
            for pi,assign in enumerate(Perms):
                # We manually compute the (squared) norm because it respects masked arrays, while np.linalg.norm does not.
                D[pi] = ((vexpect[:,g] - vin[:,g[assign]])**2).sum()
            # Identify the winning assignment
            if randomAssign:
                beta = 1.
                probs = np.exp(- beta * D)
                probs /= probs.sum()
                pi = np.random.choice(range(len(Perms)), p=probs)
                assign = Perms[pi]
            else:  # This is the default case: deterministic assignment of the best-matching permutation
                pi = np.argmin(D)
                assign = Perms[pi]
            vout[:,g] = vin[:,g[assign]]
            self._SFM_last_assignment[gi] = assign
            if assign != la:
                log.debug(f"Group {g} gets new assignment: {g[la]} --> {g[assign]}")
                log.debug(f"Expected velocities: {vexpect[:,g][0]}")
                log.debug(f"Assigned velocities: {vout[:,g][0]}")
        return vout.flatten()
        


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #     FILTER: IDEAL USING NATURAL PARAMETERS      # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class FilterStrinfNaturalParameters(FilterStrinfVeloBase):
    """The exact observer model.
    
    The filter operates on the natural parameters Omega (inverse covariance) and Omega * mu.
    See theory for a detailed description.
    
    Supports masked (hidden) inputs, as can be provided by class ObservationGeneratorVelo.
    """
    def __init__(self, D, C, lam0, tau_s, sig_obs, inf_method, forceOmDiagonal=False, inf_method_kwargs=None, feature_learning_method=None, feature_learning_kwargs=None,  integrator=default_integrator, polarIdx=None, RFCoord=None, bSFM=False):
        """See FilterStrinfVeloBase for a description of most parameters.
        
        Additional parameter:
            forceOmDiagonal : bool      (optional) mostly a developer feature to test the effect
                                        of a diagonal posterior covariance matrix
        """
        super().__init__(D, C, lam0, tau_s, sig_obs, inf_method, inf_method_kwargs=inf_method_kwargs, feature_learning_method=feature_learning_method, feature_learning_kwargs=feature_learning_kwargs, integrator=integrator, polarIdx=polarIdx, RFCoord=RFCoord, bSFM=bSFM)
        self.forceOmDiagonal = forceOmDiagonal   # Bool: Make the precision matrix diagonal after each observation?
        # current state estimates
        self.Om = None
        self.Omu = None
        # Auxiliary quantities
        diagTauSInv = np.diag(1/self.tau_s)
        # CTC-term (we lru_cache the function for faster execution; it does not depend on lambda)
        self.f_CTCsobs2 = lru_cache()(lambda mask: self.C.T @ np.diag(np.logical_not(mask) / self.sig_obs**2) @ self.C)
        # CT-term
        self.f_CTsobs2 = lru_cache()(lambda mask: self.C.T * ( np.logical_not(mask) / self.sig_obs**2 ))
        # Dynamics of the natural parameters (with a steady stream of observations)
        self.dOmdt = lambda Om, vmask: Om @ diagTauSInv + diagTauSInv @ Om - Om @ (self.Lam**2) @ Om + self.f_CTCsobs2(vmask)
        self.dOmudt = lambda Om, Omu, v, vmask: diagTauSInv @ Omu - Om @ (self.Lam**2) @ Omu + np.ma.dot(self.f_CTsobs2(vmask), v)

    def _make_Om_diagonal(self):
        if self.forceOmDiagonal and (self.Om is not None) and (self.Omu is not None):
            Omdiag = np.diag(self.Om.diagonal())
            mu = self.mu
            self.Om[:] = Omdiag
            self.Omu[:] =  Omdiag @ mu

    # The mu property (for convenient interfacing)
    @property
    def mu(self):
        assert self.Om is not None, "No mu without Omega!"
        return np.linalg.inv(self.Om) @ self.Omu

    @mu.setter
    def mu(self, s):
        if s is None:
            self.Omu = None
            return
        assert self.Om is not None, "No mu without Omega!"
        if self.Omu is not None:
            self.Omu[:] = self.Om @ s
        else:
            self.Omu = self.Om @ s

    # The Sigma property (for convenient interfacing)
    @property
    def Sig(self):
        assert self.Om is not None, "No Sig without Omega!"
        return np.linalg.inv(self.Om)

    @Sig.setter
    def Sig(self, S):
        if S is None:
            self.Om = None
            return
        if self.Om is not None:
            self.Om[:] = np.linalg.inv(S)
        else:
            self.Om = np.linalg.inv(S)
        self._make_Om_diagonal()

    def init_filter(self, s=None):
        s = super()._init_filter_base(s)
        # Initialize Cov at equilibrium (given lambda)
        self.Sig = (self.tau_s / 2.) * self.Lam**2
        # Set Omu after Om
        self.mu = s.copy()
        self.archive["mu"] = [self.mu.copy().reshape(self.D,self.M)]
        self.archive["Sig"] = [self.Sig.copy()]

    def _set_C_dependent_functions(self, C):
        """For feature learning only."""
        self.C[:] = C
        # Update other dependent functions,  e.g., call f.cache_clear() for lru_cached functions.
        # This inevitably makes the code really slow when feature learning is enabled.
        if self.feature_learning_method is not None:
            self.f_CTCsobs2.cache_clear()
            self.f_CTsobs2.cache_clear()

    # Wrapper for the scipy integrator (the "self" will nontheless match the required function signature)
    def dzdt(self, t, Z, v):
        """Uses a flattened version of the natural parameters:
           Z = [Om.flatten()] + [Omu]
        """
        DM = self.D * self.M
        if isinstance(v, np.ma.core.MaskedArray):
            vmask = tuple(v.mask)  # tuple-type is hashable by lru_cache
        else:   # all umasked
            vmask = tuple(np.zeros(v.shape, dtype=bool))
        assert len(Z) == DM**2 + DM
        Om = Z[:DM**2].reshape(DM,DM)
        Omu = Z[-DM:]
        dOmdt = self.dOmdt(Om, vmask)
        dOmudt = self.dOmudt(Om, Omu, v, vmask)
        res = np.concatenate((dOmdt.flatten(), dOmudt))
        return res

    def propagate_and_integrate_observation(self, v, t):
        assert self.t_last is not None, " > ERROR: Initialize filter state first!"
        assert t >= self.t_last, " > ERROR: Negative interval since last observation!"
        # For structure-from-motion, we assign the best matching (local) input. 
        if self.bSFM:
            v = self._assign_best_matching_inputs(v)
        # Call signature: self.MYDFDT(t, state, v)
        y0 = np.concatenate((self.Om.flatten(), self.Omu))
        kwargs = dict(fun=self.dzdt, args=(v,), y0=y0, t_span=(self.t_last, t) )
        sol = self.integrator(**kwargs)
        if not sol.success:
            log.error("Numerical integration failed with message: " + sol.message)
        assert np.isclose(sol.t[-1], t), "Numerical integration did not reach requested time."
        Z = sol.y[:,-1]
        DM = self.D * self.M
        Om = Z[:DM**2].reshape(DM,DM)
        Omu = Z[-DM:]
        self.Om = Om
        self.Omu = Omu
        self._make_Om_diagonal()
        self.t_last = t
        # store results
        self.archive["t"].append(t)
        self.archive["mu"].append( self.mu.copy().reshape(self.D,self.M) )
        self.archive["Sig"].append( self.Sig.copy() )



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #     FILTER: ADIABATIC DIAGONAL APPROX           # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class FilterStrinfAdiabaticDiagonalSigma(FilterStrinfVeloBase):
    """The adiabatic observer model.
    
    The filter operates directly on the mean mu. The posterior uncertainty is calculated by fSigma(lam^2).
    See theory for a detailed description.
    
    Supports masked (hidden) inputs, as can be provided by class ObservationGeneratorVelo.
    """
    def __init__(self, D, C, lam0, tau_s, sig_obs, inf_method, inf_method_kwargs=None, feature_learning_method=None, feature_learning_kwargs=None, integrator=default_integrator, bOminvAssumeAllVisible=False, polarIdx=None, RFCoord=None, bSFM=False):
        """See FilterStrinfVeloBase for a description of most parameters.
        
        Additional parameters:
            
            bOminvAssumeAllVisible : bool       (optional) when calculating fSigma, disregard possibly masked inputs
                                                in the norm of the motion features. Default: False
        """
        super().__init__(D, C, lam0, tau_s, sig_obs, inf_method, inf_method_kwargs=inf_method_kwargs, feature_learning_method=feature_learning_method, feature_learning_kwargs=feature_learning_kwargs, integrator=integrator, polarIdx=polarIdx, RFCoord=RFCoord, bSFM=bSFM)
        # current state estimates
        self.mu = None
        self.bOminvAssumeAllVisible = bOminvAssumeAllVisible  # if True, f_Ominv will assume that vmask=[False]*(D,K)
        self._lastVmask = None  # last input visibility for self.Sig
        # Auxiliary quantities
        diagTauSInv = np.diag(1/self.tau_s)
        f_sobs2 = lambda mask: np.logical_not(mask) / self.sig_obs**2
        # Adiabatic approximation of Covariance (assumes: diagonal and converged to steady state)
        self.f_Cnormsobs2 = lru_cache()( lambda mask: (self.C.T @ np.diag(f_sobs2(mask)) @ self.C ).diagonal() )
        self.dmudt = lambda t, mu, v, mask: - diagTauSInv @ mu + self.f_Ominv(mask) @ np.ma.dot(self.C.T, (f_sobs2(mask) * ( v - self.C @ mu )))

    def f_Ominv(self, mask):
        # return 0.05 * self.Lam**2  # Just a test if this works also with a linear approximation (no, it does not; as expected).
        if self.bOminvAssumeAllVisible is True:
            mask = np.zeros(self.D*self.K, dtype=bool)
        Cnormsobs2 = self.f_Cnormsobs2(tuple(mask))
        ominv = []
        for tau, lam, c2 in zip(self.tau_s, self.Lam.diagonal(), Cnormsobs2):
            if c2 == 0:
                ominv.append( tau / 2 * lam**2 )
            else:
                ominv.append( 1 / (tau * c2) * ( -1 + np.sqrt( 1 + lam**2 * c2 * tau**2 ) ) )
        return np.diag( ominv )


    @property
    def Sig(self):
        if self._lastVmask is None:
            mask = np.zeros(self.D*self.K, dtype=bool)
        else:
            mask = self._lastVmask
        return self.f_Ominv(mask)

    def init_filter(self, s=None):
        s = super()._init_filter_base(s)
        self.mu = s.copy()
        self.archive["mu"] = [self.mu.copy().reshape(self.D,self.M)]
        self.archive["Sig"] = [self.Sig.copy()]

    def _set_C_dependent_functions(self, C):
        """For feature learning only."""
        self.C[:] = C
        # Update other dependent functions,  e.g., call f.cache_clear() for lru_cached functions.
        if self.feature_learning_method is not None:
            self.f_Cnormsobs2.cache_clear()

    def propagate_and_integrate_observation(self, v, t):
        assert self.t_last is not None, " > ERROR: Initialize filter state first!"
        assert t >= self.t_last, " > ERROR: Negative interval since last observation!"
        # For structure-from-motion, we assign the best matching (local) input. 
        if self.bSFM:
            v = self._assign_best_matching_inputs(v)
        if isinstance(v, np.ma.core.MaskedArray):
            vmask = v.mask
        else:   # all umasked
            vmask = np.zeros(v.shape, dtype=bool)
        self._lastVmask = vmask  # Used for self.Sig
        # Call signature: self.MYDFDT(t, state, v, vmask)
        kwargs = dict(fun=self.dmudt, args=(v,vmask), y0=self.mu, t_span=(self.t_last, t) )
        sol = self.integrator(**kwargs)
        if not sol.success:
            log.error("Numerical integration failed with message: " + sol.message)
        assert np.isclose(sol.t[-1], t), "Numerical integration did not reach requested time."
        self.mu = sol.y[:,-1]
        self.t_last = t
        # store results
        self.archive["t"].append(t)
        self.archive["mu"].append( self.mu.copy().reshape(self.D,self.M) )
        self.archive["Sig"].append( self.Sig.copy() )





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #     I N F E R E N C E   M E T H O D S  (USE WITH FILTERS)     # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def inf_method_none(self):
    """Simply return the current strengths as new estimate."""
    return self.lam


def inf_method_MAP_sliding_lam2(self, fps, tau_s, tau_lam, useSigma=True, kappa=0., nu=0.):
    """MAP estimate of lam, using mu and (optionally) Sigma. This is the standard method of the theory.
    
    Parameters (coming from the experiment description):
    
        fps : float                   observation frame rate (typically, fps=1/dt in ObservationGeneratorVelo)
        tau_s : float or M array      time constant tau_s (should match the one of the filter)
        tau_lam : float or M array    time constant tau_lam (defines the effective number of samples)
        useSigma : bool               (optional) if True, include the posterior variance in the lambda inference
                                      Default: True
        kappa : float or M array      (optional) value of pseudo observations (hyper-prior; see comments)
        nu : float or M array         (optional) number of pseudo observations (hyper-prior; see comments)
        
    Comments:
      * kappa techically supports negative values; but use this with care as it is unrealistic and could lead to negative lam2!
      * nu = 0 is the Jeffreys prior; nu = -2/D is a uniform prior leading to maximum likelihood inference.
    """
    D, M = self.D, self.M
    # Set up lam^2 running average
    if not hasattr(self, "lam2"):
        self.lam2 = self.lam**2
    # Calc mu2_m = \sum_d mu_d,m^2
    mu2  = (self.mu.reshape(D,M)**2).sum(0)
    # same for sigma (already squared)
    if useSigma:
        sig2  = self.Sig.diagonal().reshape(D,M).sum(0)
    else:
        sig2 = 0.
    # Calc instantaneous target
    target = (D * tau_s/2)**(-1) * ( tau_s/2 * nu * np.sign(kappa) * kappa**2 + tau_lam/tau_s * (mu2 + sig2) ) / ( nu + tau_lam/tau_s + 2/D )
    self.lam2 += - 1/(tau_lam*fps) * ( self.lam2 - target )
    return np.sqrt(self.lam2)


def inf_method_MAP_sliding_lam2_CTC_interaction(self, fps, tau_s, tau_lam, J0=0., useSigma=True, kappa=0., nu=0.):
    """MAP estimate of lam, using the Kalman filter's mu and Sig approximations.
       Further, it includes negative interactions of strength J_mn = J0 * (C^T C)_mn for m != n
    """
    D, M = self.D, self.M
    # Pre-calculate interaction matrix
    if not hasattr(self, "_Jinter"):
        Jinter = np.abs(self.C0.T @ self.C0)
        Jinter -= np.diag(Jinter.diagonal())  # remove self-interaction
        self._Jinter = Jinter
    # Set up lam^2 running average
    if not hasattr(self, "lam2"):
        self.lam2 = self.lam**2
    # Calc mu2_m = \sum_d mu_d,m^2
    mu2  = (self.mu.reshape(D,M)**2).sum(0)
    # same for sigma (already squared)
    if useSigma:
        sig2  = self.Sig.diagonal().reshape(D,M).sum(0)
    else:
        sig2 = 0.
    # Calc instantaneous target (np.array(nu).reshape(-1,1) ensures correct broadcasting with _Jinter if nu is an array of length M)
    target = (D * tau_s/2)**(-1) \
            * (np.eye(M) - J0 * 2 / ( np.array(nu).reshape(-1,1) + tau_lam/tau_s + 2/D ) * np.diag(self.lam2**2) @ self._Jinter) @ ( ( tau_s/2 * nu * np.sign(kappa) * kappa**2 + tau_lam/tau_s * (mu2 + sig2) ) \
           / ( nu + tau_lam/tau_s + 2/D ) )
    # Integrate
    self.lam2 += - 1/(tau_lam*fps) * ( self.lam2 - target )
    self.lam2[:] = np.maximum(0, self.lam2)    # The interaction can (numerically) press components below 0.
    return np.sqrt(self.lam2)


def inf_method_MAP_sliding_lam2_manual_interaction(self, fps, tau_s, tau_lam, Jinter, J0=0., useSigma=True, kappa=0., nu=0.):
    """MAP estimate of lam, using the Kalman filter's mu and Sig approximations.
       Further, it includes negative interactions of strength J_mn = J0 * Jinter
    """
    D, M = self.D, self.M
    self._Jinter = Jinter
    # Set up lam^2 running average
    if not hasattr(self, "lam2"):
        self.lam2 = self.lam**2
    # Calc mu2_m = \sum_d mu_d,m^2
    mu2  = (self.mu.reshape(D,M)**2).sum(0)
    # same for sigma (already squared)
    if useSigma:
        sig2  = self.Sig.diagonal().reshape(D,M).sum(0)
    else:
        sig2 = 0.
    # Calc instantaneous target (np.array(nu).reshape(-1,1) ensures correct broadcasting with _Jinter if nu is an array of length M)
    target = (D * tau_s/2)**(-1) \
            * (np.eye(M) - J0 * 2 / ( np.array(nu).reshape(-1,1) + tau_lam/tau_s + 2/D ) * np.diag(self.lam2**2) @ self._Jinter) @ ( ( tau_s/2 * nu * np.sign(kappa) * kappa**2 + tau_lam/tau_s * (mu2 + sig2) ) \
           / ( nu + tau_lam/tau_s + 2/D ) )
    # Integrate
    self.lam2 += - 1/(tau_lam*fps) * ( self.lam2 - target )
    self.lam2[:] = np.maximum(0, self.lam2)    # The interaction can (numerically) press components below 0.
    return np.sqrt(self.lam2)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #     F E A T U R E   L E A R N I N G   M E T H O D S  (USE WITH FILTERS)     # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def feature_learning_gradient_based(self, v, eta, fps, tau_s, numSamples, f_regularize=lambda C: 0.):
    """Experimental learning of the C matrix.

    Parameters: 
    
        v : D*K array                       observed (noisy) velocity
        eta : float                         learning rate
        fps : float                         observation frame rate
        tau_s : float or M array            time constant tau_s
        numSamples : float                  scales the strength of the regularizer
        f_regularize : callable             regularizer on C
    """
    # Data contribution
    if isinstance(self.sig_obs, (float, np.floating)):
        sig_obs = self.sig_obs * np.ones(self.D * self.K)
    else:
        sig_obs = self.sig_obs
    dCdt_data = (sig_obs**(-2))[:,None] * ( np.outer(v, self.mu) - self.C @ ( np.outer(self.mu, self.mu) + self.Sig ) )
    dCdt_regularizer = (sig_obs**(-2))[:,None] * 1/numSamples * f_regularize(self.C)
    dC = eta / (fps * numSamples * tau_s) * ( dCdt_data + dCdt_regularizer )
    return self.C + dC

