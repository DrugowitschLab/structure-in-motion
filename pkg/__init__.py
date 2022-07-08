"""
Classes and functions for online motion structure inference.
Written by Johannes Bill (johannes_bill@hms.harvard.edu or bill.scientific@gmail.com)
"""

from .mpl_defaults import cm2inch
from .func import create_logger
log = create_logger()

from .func import create_dsl, init_logging, asciiL, copy_cfg, import_config, print_progress, trialNumberAware

from .classIO import TrialStorage, load_dataset

from .clsPopCode import LinearPopulationCode, OneToOnePopulationCode, LinearInputPopulationCode

from .classDdim import WorldOUSources, WorldOUSourcesLamlist, WorldOscillator, WorldVeloFunction,\
                       ObservationGeneratorVelo, \
                       FilterStrinfNaturalParameters, FilterStrinfAdiabaticDiagonalSigma,\
                       inf_method_none, inf_method_MAP_sliding_lam2, inf_method_MAP_sliding_lam2_CTC_interaction,\
                       inf_method_MAP_sliding_lam2_manual_interaction,\
                       feature_learning_gradient_based

from .classNet2dim import RateNetworkMTInputStrinfAdiabaticDiagonalSigma, \
                          make_MT_tuning_function

# General plotting functions which can also be used elsewhere
import strinf.plottools as plot

# Some specific plotting functions for structure inference 
import strinf.colors as colors
from .plotting import plot_inferred_lambda, plot_inferred_source, imshow_C_matrix

