# # # # # # # # #
# # #
# # #   Base parameters for object-indexed experiments
# # #   Import * into individual config files and override parameters where needed
# # #
# # # # # # # # # # # # # # # # # # # # #

tau_s = 0.300           # Time constant of motion sources inference
tau_lam = 1.000         # Time constant of lambda inference
fps = 60.               # Stimulus presentation frames per second
sig_obs = 0.05          # Observation noise (fps independent)
lam0 = 0.50             # Initial value of lambda
nu = 0.                 # Num pseudo-observations in hyperprior
kappa = 0.              # Avg value of pseudo obs.; obsolete when nu = 0

# For logging, we print what will be imported
from strinf import log as logger
for name in ('tau_s','tau_lam','fps','sig_obs','lam0','nu','kappa'):
    logger.info(f"Import default parameter: {name} = {globals()[name]}")


