# Code for online motion structure inference

Code for the research paper:
_Johannes Bill, Samuel J Gershman, Jan Drugowitsch: Visual motion perception as online hierarchical inference (2022)_. Preprint: https://www.biorxiv.org/content/10.1101/2021.10.21.465346 . 
This repository contains the Python code for simulations, analyses and plotting.

This version has a DOI: [![DOI](https://zenodo.org/badge/511930898.svg)](https://zenodo.org/badge/latestdoi/511930898)

## Table of contents
- [System requirements](#system-requirements)
- [Installation](#installation)
- [Directory structure](#directory-structure)
- [Basic usage (reproducing Figure 2)](#basic-usage--reproducing-figure-2-)
- [The simulation's inner workings](#the-simulation-s-inner-workings)
- [Details for reproducing Figures 3 to 7](#details-for-reproducing-figures-3-to-7)

## System requirements

The software has been developed and tested for Python 3.8 on Linux Mint 20.3. 
Detailed requirements on software packages are provided in the file `requirements.txt`.
Any modern desktop computer with 16GB+ of memory should be sufficient to run the code within one or two work days.

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

**Remark:** The file `requirements.txt` contains exact version numbers of all packages. For some users, it may be more convenient to use the versions shipped with their distribution. To this end, use `pip3 install -r requirements_no_version.txt` which does not constrain the packages' version numbers.

## Directory structure
```
/
|-- ana     # data analysis
|-- data    # input and output data for the simulations
|-- pkg     # package with all classes and functions
|-- plt     # plotting
+-- sim     # simulations (algorithms and network)
```

## Basic usage (reproducing Figure 2)

### Running simulations

Make sure that the `.venv` environment is activated via `source .venv/bin/activate`, and `cd` into `./sim/`.
Every experiment is fully described by a config file `cfg_<NAME>.py`.
To run the experiment, simply type `python run.py cfg_<NAME>.py`.
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
   Each simulation will create a DataSet Label (`DSL`) using the current system time.
   The `DSL` is printed on the terminal near the beginning of the simulation,
   and the results are stored in `./data/<DSL>/`.
   
3. Plot the results:  
   a) Edit the file `./plt/plot_figure_2.py` and enter the two `DSL`'s from your simulations in the Python `dict`'s named `johansson` (in `line 14`) and `duncker` (in `line 22`). Save the changes.  
   b) Go to the plotting directory, create a directory for the figures (if not yet present), and run the plotting script:
   ```
   cd ../plt
   mkdir fig
   python plot_figure_2.py
   ```
   This should plot the figure on the screen and save two files: `./fig/Figure_2.pdf` and `./fig/Figure_2.png`.


## The simulation's inner workings
As we have seen, all experiments are specified by a config file `cfg_<NAME>.py`
which is executed by `run.py`. So what happens in `run.py`?

Up to `line 14` of the file `run.py`, we perform administrative work: import packages, load the `cfg` file, and create a log file.
Until `line 40`, the relevant `object`'s are created (see below) with the parameters from the `cfg` file.
Until `line 64`, the data storage is prepared by fetching the required number of inputs, motion sources, etc. from the `object`'s.
The main loop starts at `line 67`, and its interesting parts end already at `line 88`.
The inner workings of these 20 lines of code are described below.
Starting from `line 89`, we only deal with saving the data for every trial (within the main loop) and some meta data (after the main loop).
In `line 135`, the simulation is complete and all data has been saved to disk.
Thereafter, we have an optional code block for plotting some technical results (time evolution of the motion strengths and sources of the first simulation trial).

**The central 20 lines:** 
The inner workings of simulations are as follows. A World object `wld` creates the true velocities, either stochastically or deterministically.
An ObservationGenerator object `obs` creates noisy observations of the true velocities at the observation times.
Hierarchical inference models and neural networks are called filters `fil` in the code.
At the heart of the main loop (`lines 82-85`), we advance time from observation to observation (default: 1/60 sec per step). At each such time step `t`, we first present the noisy observation `v` to the filter via `fil.propagate_and_integrate_observation(v, t)` which advances the filter through time and infers the motion sources `s` (E-step). 
Then, we call the filter's `fil.infer_strengths()` method to update the motion strengths `lam` (M-step).
Afterwards (`line 88`), we may call the function for online learning of the components in the C-matrix (not part of this research paper).

Neural network simulations (`class RateNetworkMTInputStrinfAdiabaticDiagonalSigma`) simplify the above scheme even further in that `fil.propagate_and_integrate_observation(v, t)` advances the activity of all neurons, thereby performing the E- and M-steps simultaneously, while the method `fil.infer_strengths()` is a dummy that performs no operation.


## Details for reproducing Figures 3 to 7

The experiments in Figures 3 to 7 are more involved. Here we outline the steps for reproducing them.

Before you start, create a sub-directory `/ana/fig/` for technical figures which are generated during analysis.


**Remark:** When following the instructions, always `cd` into the script's directory, i.e., "Run `/ana/Yang_regress_and_choice.py`" means `cd ana` and then `python Yang_regress_and_choice.py`.


### Figure 3 

#### Download the trials
Download the experiment trials of (Yang et al., 2021) from their [GitHub repository](https://github.com/DrugowitschLab/motion-structure-identification/tree/master/data/exp1) and store them in your project directory `/data/data_yang2021_experiment_1/`. All you need are the files `<N>_exp1.dat` for the N=1..12 participants, e.g., `/data/data_yang2021_experiment_1/3/3_exp1.dat` for the third participant.

#### Run the online model
Run the simulation on the trials using the config file `cfg_040_YangETAL_2021.py`. The simulation must be run for each participant separately, with the participant ID (`pid=1..12`) entered in `line 51` of the config file. Store the dataset labels in the dict `DSL` in `/ana/YangETAL_dsl.py` for the subsequent analysis.  

#### Fit the choice model on the inferred motion strengths 
Run `/ana/Yang_regress_and_choice.py` which fits the choice model and stores the results in `/ana/data_yang2021_experiment_1/yang_predicted_choices_by_motion_structure_inference_algorithm_CV.pkl`.

#### Plot the figure
Run `/plt/plot_figure_3.py` to plot Figure 3. The script calculates the confusion matrices and choice log-likelihoods based on the file saved in the previous step. For the comparison in panel 3F, the log-likelihoods under the model of (Yang et al., 2021) are loaded from `./data/data_yang2021_experiment_1/yang_logL_exp1_4paramModel.txt`.


### Figure 4

#### Run the model for the basic MDR experiments (panels 4e, f, g)

Run simulations for angle dependence (`cfg_100_direction_repulsion_Braddick.py`), contrast dependence (`cfg_103_direction_repulsion_ChenETAL_by_contrast.py`), and speed dependence (`cfg_105_direction_repulsion_Braddick_speed_2nd_component.py`).
For contrast dependence, the simulation must be run separately for every noise factor listed in `line 16` of the config file: select the noise factor in `line 17` and store the dataset labels in the dict `DSL` in `/ana/ChenETAL_direction_repulsion_by_contrast_dsl.py`. 

#### Analyze the simulation data for the basic MDR experiments

Run the following analysis scripts in `/ana/`, with the dataset labels from the previous step entered as `DSL` in `line 26` (panels e and g only):
* `motion_direction_repulsion_by_angle.py`for panel 4e,
* `motion_direction_repulsion_by_2nd_component_contrast.py` for panel 4f (`DSL` are loaded from file),
* `motion_direction_repulsion_by_2ndspeed.py` for panel 4g.

The results will be stored in files named `/data/analysis_<SOMETHING>.pkl`. Remember the file names.

#### Run the model for the extended MDR experiment (panels 4h-l)

Run the simulation defined by `cfg_107_nested_direction_repulsion_Takemura.py` separately for each of the five stimulus conditions in panels 4h to 4l:
In (Takemura et al., 2011), the inner stimuli had a fixed speed in x-direction and an integer-multiple of that speed in y-direction in order to stay within the screen's pixel grid. Here we only consider the speed factors `multy = 0` and `multy = 1`, entered in `line 27`, for horizontal and diagonal inner motion, respectively. The directions of the outer motion are selected in `line 28` with values `-1` (down), `+1` (up), and `0` (bidirectional).
You should obtain simulation data with dataset labels ending with, e.g., `_inner_y_1_outer_up`.

#### Analyze the simulation data for the extended MDR experiment

Enter the obtained dataset labels in `line 31` of the analysis script `/ana/motion_direction_repulsion_nested_Takemura.py`. Running the script will write the perceived directions to file `/data/analysis_direction_repulsion_Takemura.pkl`.

#### Plot the figure

In `/plt/plot_figure_4.py`, enter the four file names (pattern: `/data/analysis_<SOMETHING>.pkl`) in `lines 15`, `19`, `24`, and `31`. The human data from (Braddick et al., 2002) is loaded from `/data/data_Braddick_2002_Fig3C.txt`. Run the plot script to plot Figure 4.


### Figure 5

Run two simulations from the config file `cfg_110_lorenceau_1996.py` with the `noise_factor` in `line 24` set to `1.` (low noise) and `25.` (high noise), respectively. In `/plt/plot_figure_5.py`, enter the two obtained `DSL` in `line 14` and `line 15`. Run the script to plot Figure 5. 


###  Figure 6

#### Run the model for the single cylinder condition

Using `cfg_121_structure_from_motion_nested.py`, run two simulations:
* a short one for plotting the motion source in panel 6d: set `T = 200.` in `line 125`, and
* a long one for the switching time distribution in panel 6e: set `T = 10000.` in `line 125`. 

For this single cylinder condition, use `lines 28 to 30` of the config file (and comment `lines 32 to 34`) to define the cylinder's radius, rotation speed, and the observation noise. It is furthermore recommended to choose a meaningful `human_readable_dsl` in `line 20` (optional). Note down the two obtained `DSL`.

#### Analyze the simulation data for perceptual switching times

In the analysis script `/ana/SFM_switching_times.py`, enter the `DSL` of the long simulation in `line 11`. Running the script will extract and store the switching times in `/data/analysis_SFM_<DSL>.pkl`. Furthermore, the script will detect the threshold value for perceptual switches (printing, e.g., `> Threshold set to ±1.18`). Note down this value.

#### Run the model for the nested cylinder conditions

Using again `cfg_121_structure_from_motion_nested.py`, run three simulations for the nested conditions in panels 6h-j. This time, use `lines 32 to 34` of the config file (and comment `lines 28 to 30`). Set `T = 200.` in `line 125` for all simulations, and choose meaningful `human_readable_dsl` in `line 20` (optional).

The three simulations only differ in the rotational speeds of the outer and inner cylinders which are defined in `line 33`:
* same speed: `Omega = ... [1.0,1.0] ...`,
* fast inner cylinder: `Omega = ... [1.0,1.5] ...`,
* fast outer cylinder: `Omega = ... [1.5,1.0] ...`.

#### Plot the figure

In `/plt/plot_figure_6.py`, enter the `DSL` of the short single-cylinder simulation and of the three nested cylinder simulations in `line 15` and `lines 18 - 20`, respectively. Enter the file name of the switching times in `line 16` and the switching threshold in `line 17`. Run the script to plot Figure 6.


### Figure 7

Run two simulations using the config files `cfg_200_network_polar_plus_global.py` (for panels 7g-j) and `cfg_220_network_ramp_lambda.py` (for panel 7m) and enter the obtained `DSL` in `line 25` and `line 53` of `/plt/plot_figure_7.py`, respectively. Run the script to plot Figure 7.
