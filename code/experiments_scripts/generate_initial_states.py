import os
import pickle

from gym_edgesimulator.dataset import DatasetGenerator
from gym_edgesimulator.dataset import DatasetSampleGenerator

from constants import DATASETS_BASE_PATH, MODELS_PATH

import matplotlib
matplotlib.use('Agg')

def generate_dataset(num_of_services,
                     MIN_SERVICE_MEM, MAX_SERVICE_MEM,
                     MERGE_FAC, num_of_GT, # number of ground turth
                     PERC_LAT, latency_factor,
                     num_of_users, num_of_stations, seed):
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


def load_data(dataset):
    dir2load = os.path.join(DATASETS_BASE_PATH, str(dataset))
    with open(f'{dir2load}/initial_state.pickle', 'rb') as in_pickle:
        initial_state = pickle.load(in_pickle)
    return initial_state, dir2load


def generate_samples(dataset, num):
    initial_state, dir2load = load_data(dataset)
    state_init = DatasetSampleGenerator(initial_state)
    initial_observations = state_init.make_observation(num)
    dir2save = os.path.join(DATASETS_BASE_PATH, str(dataset))
    with open(f'{dir2save}/initial_observations.pickle', 'wb') as out_pickle:
        pickle.dump(initial_observations, out_pickle)


generate_dataset(num_of_services=21,
                 MIN_SERVICE_MEM=2, MAX_SERVICE_MEM=8,
                 MERGE_FAC=3, num_of_GT=5,
                 PERC_LAT=0.5,
                 latency_factor=1,
                 num_of_users=20, num_of_stations=10, seed=80)

# generate_samples(0, 10)