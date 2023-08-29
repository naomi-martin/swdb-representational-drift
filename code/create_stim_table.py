from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import pandas as pd

"""
This file contains utility functions for getting events.
"""
def create_stim_df(boc:BrainObservatoryCache ,session_id:int)-> pd.DataFrame:
    """This function returns stimulis table with columns[start,end,stim_id,stim_category]"""
    stim_df = pd.DataFrame()
    stim_list = ['drifting_gratings','static_gratings','natural_movie_one',
                 'natural_movie_two','natural_movie_three','natural_scenes']
    data_set = boc.get_ophys_experiment_data(ophys_experiment_id=session_id)
    data_set_stims = [i for i in data_set.list_stimuli() if i in stim_list]
    for stim in data_set_stims:
        data_df = data_set.get_stimulus_table(stim)
        if 'natural_movie' in stim:
            data_df['stim_id'] = data_df['frame']
            data_df['stim_category'] = str(stim) 
            data_df = data_df.drop(columns = ['frame','repeat'])
            stim_df = pd.concat([stim_df,data_df],ignore_index = True)
            del data_df
        elif stim == 'natural_scenes':
            data_df['stim_id'] = data_df['frame']
            data_df['stim_category'] = str(stim) 
            data_df = data_df.drop(columns = ['frame'])
            stim_df = pd.concat([stim_df,data_df],ignore_index = True)
            del data_df
        elif stim == 'drifting_gratings':
            data_df['temporal_frequency'] = data_df['temporal_frequency'].astype(str)
            data_df['orientation'] = data_df['orientation'].astype(str)
            data_df['blank_sweep'] = data_df['blank_sweep'].astype(str)
            data_df['stim_id'] = data_df['orientation'] + '_' + data_df['temporal_frequency'] + '_' +     data_df['blank_sweep']
            data_df['stim_category'] = str(stim) 
            data_df = data_df.drop(columns = ['orientation','temporal_frequency','blank_sweep'])
            stim_df = pd.concat([stim_df,data_df],ignore_index = True)
            del data_df
        elif stim == 'static_gratings':
            data_df['orientation'] = data_df['orientation'].astype(str)
            data_df['spatial_frequency'] = data_df['spatial_frequency'].astype(str)
            data_df['phase'] = data_df['phase'].astype(str)
            data_df['stim_id'] = data_df['orientation'] + '_' + data_df['spatial_frequency'] + '_'+ data_df['phase']
            data_df['stim_category'] = str(stim) 
            data_df = data_df.drop(columns = ['orientation','spatial_frequency','phase'])
            stim_df = pd.concat([stim_df,data_df],ignore_index = True)
            del data_df

    return stim_df