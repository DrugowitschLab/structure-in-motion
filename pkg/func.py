"""Helper functions"""

import numpy as np
import logging

# # #  L O G G I N G  # # #

# # # COLOR FORMATTER from https://stackoverflow.com/a/384125 # # #
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
#The background is set with 40 plus the number of the color, and the foreground with 30
#These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"

def formatter_message(message, use_color = True):
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message

COLORS = {
    'WARNING': RED,
    'INFO': WHITE,
    'DEBUG': BLUE,
    'CRITICAL': RED,
    'ERROR': RED
}

class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color = True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        from copy import copy
        record = copy(record)
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname[0] + RESET_SEQ
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)


def create_logger():
    logger = logging.getLogger("Logger")
    logger.setLevel(level="INFO")
    consoleHandler = logging.StreamHandler()
    logFormatter = ColoredFormatter(formatter_message("[$BOLD%(levelname)-12.12s$RESET] %(message)s"))
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    # Also catch assertion error,... (does not work withing IPython)
    import sys
    sys.excepthook = lambda *args: logger.error("Uncaught exception:", exc_info=args)
    logger.debug("Logger created.")
    return logger

def init_logging(loglevel, dryrun, outdir, logger=None):
    if logger is None:
        from . import log
        logger = log
    logger.setLevel(level=loglevel)
    # logger._cache.clear()                     # the cache might contain old level values
    logger.info("Log level is %s." % loglevel)
    if not dryrun:
        import os
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        logging_fname = os.path.join(outdir, "logfile.log")
        fileHandler = logging.FileHandler(logging_fname)
        logFormatter = logging.Formatter("%(asctime)s [%(levelname)-8.8s] %(module)10.10s:%(lineno)4s: %(message)s")
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)
        logger.debug("Logging to file '%s'." % logging_fname)
    else:
        logger.warning("Dryrun. No data will be saved!\n\n \
                        \t\t * * * * * * * * * * * * * * * * * * *\n \
                        \t\t * * * * * *  D R Y R U N  * * * * * *\n \
                        \t\t * * * * * * * * * * * * * * * * * * *\n")


# # #  H E L P E R S  # # #

def import_config(globals):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('cfgfile', type=str, help="Name of the config file to use.")
    args = parser.parse_args()
    fnamebase = args.cfgfile.replace('.py', '')
    importstr = f"from {fnamebase} import cfg"
    exec(importstr, globals)
    return fnamebase + ".py"

def copy_cfg(cfgfname, outdir, dryrun):
    from . import log
    if dryrun:
        return
    from os import path
    outfname = path.join(outdir, "config.py")
    from shutil import copy
    copy(cfgfname, outfname)
    log.debug(f"Config copied to '{outfname}'")

def create_dsl(hrd=""):
    from datetime import datetime
    dsl = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f") + "_" + hrd
    return dsl

def asciiL(L, indent=0):
    indent = " "*indent
    theta = np.array([0.05, 0.500, 0.999, 1.001])
    chars = {
        'zero' : " ",
        3 : "█",
        2 : "▓",
        1 : "▒",
        0 : "░"
        }
    vmax, vmin = L.max(), L.min()
    char = lambda val: chars['zero'] if val == 0. else chars[( (val - vmin)/(vmax-vmin) < theta ).argmax()]
    s = indent + "┌" + "─"*(2*L.shape[1]) + "┐\n"
    for line in L:
        s += indent + "│" + "".join( [ char(v)*2 for v in line] ) + "│\n"
    s += indent + "└" + "─"*(2*L.shape[1]) + "┘"
    return s


def print_progress(t0, dt_obs, PRINTEVERY, indent=" "*6):
    import sys
    t_last = 0.
    t = t0
    print(indent + "Progress: " + " "*6, sep='', end='', file=sys.stdout, flush=True)
    while True:
        if t >= t_last:
            t_last += PRINTEVERY
            print("\b"*8 + "%7.1fs" % t, sep='', end='', file=sys.stdout, flush=True)
        t += dt_obs
        yield t




# # #  D E C O R A T O R S  # # #

import functools

class trialNumberAware:
    """
    Decorator class for making a function aware of the current trial number r (repetition).
    The decorated function f(t, *args, **kwargs) will be called with f(*args, t=t, trialNumber=r, **kwargs).
    The decorator will increase the int r whenever the function is called with argument t=0.,
    unless the previous call was at t=0. (or it is the first call).
    """
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.r = 0
        self.readyForIncrease = False

    def __call__(self, t, *args, **kwargs):
        if (t==0.) and self.readyForIncrease:
            self.r += 1
            self.readyForIncrease = False
        elif t > 0.:
            self.readyForIncrease = True
        return self.func(*args, t=t, trialNumber=self.r, **kwargs)


