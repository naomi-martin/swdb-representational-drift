"""code from https://github.com/kaitken17/drift_geometry/blob/main/passive_2photon_drift.ipynb """
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)

import allensdk
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import pprint
import pickle
import pkg_resources


n_frames = 900
n_repeats = 10
n_divs = 30



def save_data_summary(path, exp_id, n_divs, dff_values, cell_ids):

    save_the_file = True

    if os.path.exists(path):
        print('File already exists at:', path)
        override = input('Override? (Y/N):')
        if override == 'Y':
            save_the_file = True
        else:
            save_the_file = False

    if save_the_file:
        with open(path, 'wb') as save_file:
            pickle.dump(dff_values, save_file, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(cell_ids, save_file, protocol=pickle.HIGHEST_PROTOCOL)
        print('Data Saved')
    else:
        print('Data Not Saved')

def load_data_summary(root_path, exp_id, n_divs):

    save_string = get_save_string_summary(exp_id, n_divs)
    path = root_path + save_string + '.pickle'
    

    if not os.path.exists(path):
        raise ValueError('No file at path:', path)
    else:
        with open(path, 'rb') as load_file:
            dff_values_load = pickle.load(load_file)
            cell_ids_load = pickle.load(load_file)
 
        return dff_values_load, cell_ids_load

def get_save_string_summary(exp_id, n_divs):
    """ Returns a unique string identifier for the data """
    return ('summary _datanaturalmovieone_ndivs' + str(n_divs) + '_expid' + str(exp_id))

def get_align_angle(x, y):
    dot = np.dot(x,y)/(
         np.linalg.norm(x) * np.linalg.norm(y)
     )
    if dot > 1.0:
         dot = 1.0
    elif dot < -1.0:
        dot = -1.0
    
    return 180/np.pi * np.arccos(dot)
def get_dff_vals_dataset(data_set, n_divs = 30):

    if n_frames % n_divs > 0:
        raise ValueError('Number of frames does not divide evenly.')
    frames_per_repeat = int(n_frames/n_divs)

    time, dff_traces = data_set.get_dff_traces()
    cell_ids = data_set.get_cell_specimen_ids()
    stim_table = data_set.get_stimulus_table('natural_movie_one')

    n_cells = len(cell_ids)

    frame_idxs = np.zeros((n_repeats, n_divs, int(n_frames/n_divs)))
    dff_vals = np.zeros((n_repeats, n_divs, n_cells))

    for repeat_idx in range(n_repeats):
        repeat_frames = np.array(stim_table.query('repeat == @repeat_idx')['start'])
        for div_idx in range(n_divs):
            div_repeat_idxs = repeat_frames[
               div_idx*frames_per_repeat:(div_idx+1)*frames_per_repeat
            ]
            frame_idxs[repeat_idx, div_idx] = div_repeat_idxs
            dff_vals[repeat_idx, div_idx] = np.mean(dff_traces[:, div_repeat_idxs],
                                                   axis=1)

    return dff_vals, cell_ids
