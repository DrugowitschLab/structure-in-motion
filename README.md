# Code for online motion structure inference

Code for the research paper:
_Johannes Bill, Samuel J Gershman, Jan Drugowitsch: Structure in motion: visual motion perception as online hierarchical inference (2022)_. Preprint: https://www.biorxiv.org/content/10.1101/2021.10.21.465346
This repository contains the Python code for simulations, analyses and plotting.

```
Remark (2022-07-07): This is a pre-release of the software during the review process. A final version will be uploaded upon publication.
```

## System requirements

The software has been developed and tested for Python 3.8 on Linux Mint 20.3. 
Detailed requirements on software packages are provided in the file `requirements.txt`.
Any modern desktop computer should be sufficient to run the code.

## Installation

We assume a Linux installation based on Ubuntu 20.04. On Mac, you should be able to homebrew with sip and pyqt. First, clone this repository. In the cloned repository, use a virtual environment with Python 3.8 (execute the following _line by line_):

```
python3 -m pip install --user --upgrade pip   # Install pip (if not yet installed)
sudo apt-get install python3-venv             # May be needed for environment creation
python3.8 -m venv .venv                       # Create environment with the right python interpreter (must be installed)
source .venv/bin/activate                     # Activate env
python3 -m pip install --upgrade pip          # Make sure the local pip is up to date
pip3 install wheel                            # Install wheel first
pip3 install -r requirements.txt              # Install other required packages
deactivate                                    # Deactivate env
```

## Directory structure
```
/
|-- ana     # data analysis
|-- data    # input and output data for the simulations
|-- pkg     # package with all classes and functions
|-- plt     # plotting
+-- sim     # simulations (algorithms and network)
```

## Usage

### Running simulations

Make sure that the `.venv` environment is activated via `source .venv/bin/activate`, and `cd` into `./sim/`.
Every experiment is fully described by a config file `cfg_NAME.py`.
To run the experiment, simply type `python run.py cfg_NAME.py`.
(Personally, I prefer to use `ipython`.)

### Demo: Reproducing Figure 2
To get acquainted with the software, it is helpful to go through the software pipeline for creating Figure 2.
1. Activate the environment and cd to the simulation descriptions:
   ```
   source .venv/bin/activate
   cd sim
   ```

2. Run the simulations for the Johansson experiment and the Duncker Wheel:
   ```
   python run.py cfg_020_Johansson_3dot.py
   python run.py cfg_030_DunckerWheel.py
   ```
   Each simulation will create a Data Set Label (`DSL`) using the current system time.
   The `DSL` is printed on the terminal near the beginning of the simulation,
   and the results are stored in `./data/<DSL>/`.
   
3. Plot the results:  
   a) Edit the file `./plt/plot_figure_2.py` and enter the two `DSL`'s from your simulations in the Python `dict`'s named `johansson` and `duncker`. Save the changes.  
   b) Go to the plotting directory, create a directory for the figures (if not yet present), and run the plotting script:
   ```
   cd ../plt
   mkdir fig
   python plot_figure_2.py
   ```
   This should plot the figure on the screen and save two files: `./fig/Figure_2.pdf` and `./fig/Figure_2.png`.


## The simulation's inner workings
As we have seen, all experiments are specified by a config file `cfg_NAME.py`
which is executed by `run.py`. So what happens in `run.py`?

Up to `line 14`, we have only administrative work: import packages, load the `cfg` file, and create a log file.
Until `line 40`, the relevant `object`'s are created (see below) with the parameters from the `cfg` file.
Until `line 64`, the data storage is prepared by fetching the required number of inputs, motion sources,... from the `object`'s.
The main loop starts at `line 67`, and its interesting parts end already at `line 88`.
The inner workings of these 20 lines of code are described below.
Starting from `line 89`, we only deal with saving the data for every trial (within the main loop) and some meta data (after the main loop).
In `line 135`, the simulation is complete and all data has been saved to disk.
Thereafter, we have an optional code block for plotting some technical results (time evolution of the motion strengths and sources of the first simulation trial).

**The interesting 20 lines:** 
The inner workings of simulations are as follows. A World object `wld` creates the true velocities, either stochastically or deterministically.
An ObservationGenerator object `obs` creates noisy observations of the true velocities at the observation times.
Hierarchical inference models and neural networks are called filters `fil` in the code.
At the heart of the main loop (`lines 82-85`), we advance time from observation to observation (default: 1/60 sec per step). At each such time step `t`, we first present the noisy observation `v` to the filter via `fil.propagate_and_integrate_observation(v, t)` which advances the filter through time and infers the motion sources `s` (E-step). 
Then, we call the filter's `fil.infer_strengths()` method to update the motion strengths `lam` (M-step).
Afterwards (`line 88`), we may call the function for online learning of the components in the C-matrix (not part of this research paper).
