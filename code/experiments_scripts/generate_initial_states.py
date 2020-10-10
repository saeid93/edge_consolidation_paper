"""
   scripts is used to generate
   initial dataset for the experiments
   it uses functions implemented in
   the gym_edgesimulator.dataset module to
   generate a dataset with given specs
"""
import os
import pickle
import json
import click

from gym_edgesimulator.dataset import DatasetGenerator
from gym_edgesimulator.dataset import DatasetSampleGenerator

from constants import DATASETS_BASE_PATH, MODELS_PATH

import matplotlib
matplotlib.use('Agg')

def generate_dataset(num_of_services,
                     MIN_SERVICE_MEM, MAX_SERVICE_MEM,
                     MERGE_FAC, num_of_GT,
                     PERC_LAT, latency_factor,
                     num_of_users, num_of_stations, num_of_samples,
                     seed):
    """
        use the random_initializer.py and random_state_initializer.py
        to make and save initial_states
    """


    initial = DatasetGenerator(num_of_services=num_of_services,
                               MIN_SERVICE_MEM=MIN_SERVICE_MEM,
                               MAX_SERVICE_MEM=MAX_SERVICE_MEM,
                               MERGE_FAC=MERGE_FAC,
                               num_of_GT=num_of_GT,
                               PERC_LAT=PERC_LAT,
                               num_of_users=num_of_users,
                               num_of_stations=num_of_stations,
                               latency_factor=latency_factor,
                               seed=seed) 

    # Save the Generated States in a Pickle
    # Write the generated initial state in a
    # folder for further use in other notebooks
    content = os.listdir(DATASETS_BASE_PATH)
    new_dataset = len(content)
    dir2save = os.path.join(DATASETS_BASE_PATH, str(new_dataset))
    os.mkdir(dir2save)

    # generate initial state
    initial_state, info = initial.make_initial_observation()
    print(info)

    with open(f'{dir2save}/info.txt', 'x') as out_file:
        out_file.write(info)

    with open(f'{dir2save}/initial_state.pickle', 'wb') as out_pickle:
        pickle.dump(initial_state, out_pickle)

    initial_state['simulation'].visualize_debug().savefig(f'{dir2save}/fig_debug.png')
    initial_state['simulation'].visualize_paper_style().savefig(f'{dir2save}/visualize_paper_style.png')
    print(f"\n\nGenerated data saved in {dir2save}\n\n")
    
    # generate initail states (samples)
    state_init = DatasetSampleGenerator(initial_state)
    initial_observations = state_init.make_observation(num_of_samples)
    with open(f'{dir2save}/initial_observations.pickle', 'wb') as out_pickle:
        pickle.dump(initial_observations, out_pickle)


@click.command()
@click.argument('config_file')
def main(config_file):

    with open(config_file) as cf:
        config = json.loads(cf.read())

    generate_dataset(**config)

if __name__ == "__main__":
    main()