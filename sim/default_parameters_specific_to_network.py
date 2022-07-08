# # # # # # # # #
# # #
# # #   Base parameters for network experiments, only network-specific additions
# # #   Import * into individual config files and override parameters where needed
# # #
# # # # # # # # # # # # # # # # # # # # #
import numpy as np
from default_parameters_location_indexed import tau_s

inputNeuronKwargs = dict(Na=16, Nr=12, psi=0.1, rhomin=0.10, rhomax=8.0, \
                         exponent=1.25, kappaa = 1/0.35**2, sig2rho = 0.35**2)

latentNeuronKwargs = dict(N=100, f_readout=lambda dim, N, rng: rng.normal(0, 1., (np.prod(dim),N)), \
                          rngseed=0xDEC0DE, afSig=0.001, tau_pe=tau_s/2.)
                          
# For logging, we print what will be imported
from strinf import log as logger
for name in ('inputNeuronKwargs','latentNeuronKwargs'):
    logger.info(f"[Network] Additional default parameter: {name} = {globals()[name]}")


