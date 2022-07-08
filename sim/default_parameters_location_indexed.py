# # # # # # # # #
# # #
# # #   Base parameters for location-indexed experiments
# # #   Import * into individual config files and override parameters where needed
# # #
# # # # # # # # # # # # # # # # # # # # #
from numpy import sqrt


# We take object-indexed parameters as baseline
from default_parameters_object_indexed import *

# Dividing these three by the same factor, will
# (a) compress time by this factor, and
# (b) leave the inferred motion strengths (almost) unchanged (the inferred s will be smaller by the sqrt-of-factor, though)
tau_s /= 3.            # Time constant of motion sources inference
tau_lam /= 3.          # Time constant of lambda inference
sig_obs /= 3.          # Observation noise (fps independent)

# For logging, we print what will be imported
from strinf import log as logger
for name in ('tau_s','tau_lam','sig_obs'):
    logger.info(f"[location-based] Override default parameter: {name} = {globals()[name]}")


