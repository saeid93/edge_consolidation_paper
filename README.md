# Consolidation of Services in Mobile Edge Clouds using a Learning-based Framework
Source code of "Consolidation of Services in Mobile Edge Clouds using a Learning-based Framework"

# Setup the Python environment
1. Download source code from GitHub
   ```
    git clone https://github.com/saeid93/edge_consolidation_paper.git
   ```
2. Create [conda](https://docs.conda.io/en/latest/miniconda.html) virtual-environment
   ```
    conda create --name EdgeSim python=3
   ```
3. Activate conda environment
   ```
    source activate EdgeSim
   ```
4. Install requirements
   ```
    pip install -r requirements.txt
   ```

# Description
1. The code is separated into two modules inside the /code folder
   1. /code/src/edge_simulator: contains the source code of the environemnt that is build on top of gym library
   2. /code/experiments_scripts: contains the set of codes to replicate the results of the paper and do experiments with the environments

# Setting up the simulator
1. go to the /code/src/edge_simulator and install the library in the editable mode with
   ```
   pip install -e .
   ```

# Running the experiments
1. First go to the constants.py and add the address of data, model and results to the Python file
2. Go to the code/experiments_scripts, the generate_initial_states.py can genearte a dataset of user specified criteria
   1. Specify the dataset specifications in config_generate_dataset.json
   2. Use the generate_initial_states.py from the command line to generate the dataset
   ```
   python generate_initial_states.py config_generate_dataset.json
   ```
   the results will be saved in the data folder.
3. Specify your desired config in one of the config_v(0-3).json based on the environment you want to run the experiments
   ```
   python runner.py config_v0.json
   ```
   the results will be saved in the models folder.
4. In order to plot the results of several experiments in a single plot go to the config_aggregate.json
   ```
   python runner.py config_aggregate.json
   ```
   the results will be saved in the results folder.