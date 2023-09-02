# We need to import these modules to get started
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import warnings
from event_utils import get_events
from create_stim_table import create_stim_df


def find_divide_indices(stim_table):
    '''
    Finds the indices where each new stimulus presentation after the first presentation
    starts.
    
    Returns:
        List of indices representing the start of the next stimulus
    '''
    # Find the indices where each new stimulus presentation after the first presentation starts
    divide_indices = []

    for i in range(min(stim_table.index), max(stim_table.index)):
        try:
            if stim_table.start[i+1]-stim_table.end[i] > 100:
                divide_indices.append(i) # saves index where new stim presentation begins
        except:
            pass
    return divide_indices


def divide_stim_table(stim_table, divide_indices=[]):
    '''
    Divides input stimulus table based on indices corresponding to separate stimulus
    presentations.
    If there were 3 presentations, three dataframes for each presentation are returned. 
    If there were 2 presentations, three dataframes for each presentation are still 
    returned with the third being null.
    
    Returns:
        3 divided stimulus tables corresponding to first, second, and third or null stimulus
        presentation start. 
    '''
    if len(divide_indices)==2: # 3 presentations 
        return stim_table.loc[:divide_indices[0]], stim_table.loc[divide_indices[0]+1:divide_indices[1]], stim_table.loc[divide_indices[1]+1:]
    elif len(divide_indices) == 1: # 2 presentations
        return stim_table.loc[:divide_indices[0]], stim_table.loc[divide_indices[0]+1:], pd.DataFrame()
    else:
        raise Exception("Stimulis has only one presentation. Nothing to divide.")
        
        
def generate_response_matrices(stim_table, all_events):
    '''
    This function generates response matrices given an input stimulus table. 
    Stimulus table can be divided based on stimulus presentations, or undivided.
    
    Returns:
        orientation_matrix: 1D array with orientation value for each frame
        frequency_matrix: 1D array with frequency value for each frame
        response: matrix of shape (# cells, # stimuli, 60) with response of each
            cell to each stimulus over its presentation period of 60 frames 
    '''
    stim_table = stim_table.reset_index(drop=True)
    if stim_table.stim_category[0] == "drifting_gratings": # need to split stim ID
        # generate response matrix
        response = np.empty((len(all_events.index), len(stim_table), 60)) # 3d array for responses at each frame to each stimulus for each neuron
        for i in range(len(stim_table)):
            for cell_index in range(len(all_events.index)):
                response[cell_index,i,:] = all_events.iloc[cell_index, stim_table.start[i]:stim_table.start[i]+60]
        return response
    elif stim_table.stim_category[0] == "static_gratings" or stim_table.stim_category[0] == "natural_scenes":
        # generate response matrix
        response = np.empty((len(all_events.index), len(stim_table), 7)) # 3d array for responses at each frame to each stimulus for each neuron
        stim_table = stim_table.reset_index(drop=True)
        for i in range(len(stim_table)):
            for cell_index in range(len(all_events.index)):
                response[cell_index,i,:] = all_events.iloc[cell_index, stim_table.start[i]:stim_table.start[i]+7]
        return response
    
    
def responses_per_stimulus(stim_table, response_matrix):
    '''
    response_matrix: contains all recorded responses in the divided time frame 
    '''
    responses_by_stimulus = pd.DataFrame(columns=["stim_id","response_matrix"])
    
    for stim_id in np.unique(stim_table.stim_id):
        stim_table_reset = stim_table.reset_index(drop=True) # reset full stim table indices 
        temp_stim_id_df = stim_table_reset[stim_table_reset.stim_id==stim_id] # to index specific stimuli
        response_stimulus_matrix = response_matrix[:,list(temp_stim_id_df.index),:]
        responses_by_stimulus.loc[len(responses_by_stimulus)] = [stim_id, response_stimulus_matrix]
    
    return responses_by_stimulus


def get_mean_matrices_three_epochs(stim1_responses_by_stimulus, stim2_responses_by_stimulus, stim3_responses_by_stimulus):
        
    # Average each cell's response to each stimulus (# stimuli, # neurons) across three sessions
    # Session 1
    mean_responses1 = []
    for i in range(len(stim1_responses_by_stimulus.stim_id)):
        reshape_dims = (stim1_responses_by_stimulus.response_matrix[i].shape[0], stim1_responses_by_stimulus.response_matrix[i].shape[1]*stim1_responses_by_stimulus.response_matrix[i].shape[2])
        reshaped = np.reshape(stim1_responses_by_stimulus.response_matrix[i], reshape_dims)
        mean_matrix = np.mean(reshaped, axis=1)
        mean_responses1.append(mean_matrix)
    mean_responses1 = np.asarray(mean_responses1)
    # Session 2
    mean_responses2 = []
    for i in range(len(stim2_responses_by_stimulus.stim_id)):
        reshape_dims = (stim2_responses_by_stimulus.response_matrix[i].shape[0], stim2_responses_by_stimulus.response_matrix[i].shape[1]*stim2_responses_by_stimulus.response_matrix[i].shape[2])
        reshaped = np.reshape(stim2_responses_by_stimulus.response_matrix[i], reshape_dims)
        mean_matrix = np.mean(reshaped, axis=1)
        mean_responses2.append(mean_matrix)
    mean_responses2 = np.asarray(mean_responses2)
    # Session 3
    mean_responses3 = []
    for i in range(len(stim3_responses_by_stimulus.stim_id)):
        reshape_dims = (stim3_responses_by_stimulus.response_matrix[i].shape[0], stim3_responses_by_stimulus.response_matrix[i].shape[1]*stim3_responses_by_stimulus.response_matrix[i].shape[2])
        reshaped = np.reshape(stim3_responses_by_stimulus.response_matrix[i], reshape_dims)
        mean_matrix = np.mean(reshaped, axis=1)
        mean_responses3.append(mean_matrix)
    mean_responses3 = np.asarray(mean_responses3)
    
    return mean_responses1, mean_responses2, mean_responses3


def get_overlapping_cells(container_id, boc):
    
    # Load in experiment container data 
    desired_container_id = container_id
    desired_container = boc.get_ophys_experiments(experiment_container_ids=[desired_container_id])
    desired_container = sorted(desired_container, key=lambda x: x['session_type']) # sort based on session type so A comes first
    
    # Create a list of three arrays that contains cell IDs for each session
    cells_in_each_session = []
    i=0
    for session in desired_container:
        session_id = session["id"]
        session_data = boc.get_ophys_experiment_data(ophys_experiment_id=session_id)
        specimen_ids = session_data.get_cell_specimen_ids()
        cells_in_each_session.append(specimen_ids)
        i+=1
        
    # isolate the cell IDs for only the overlapping sessions
    cells1 = list(cells_in_each_session[0])
    cells2 = list(cells_in_each_session[1])
    cells3 = list(cells_in_each_session[2])
    overlapping_cell_ids = [cell for cell in cells1 if cell in cells2 and cell in cells3]
    
    return overlapping_cell_ids


def round_frame(frame:int,events_array:pd.DataFrame,up_down:str)->int:
    """This function will round start, end frame to nearest bin in events """
    col_arr = events_array.columns.to_numpy()
    if up_down == 'up':
        bin_frame = col_arr[col_arr > frame].min()
    elif up_down == 'down':
        bin_frame = col_arr[col_arr < frame].max()
    return bin_frame


def get_mean_response_matrix_movie_one(session_id, container_id, bin_size, boc):
    # Get all event traces for overlapping neurons across sessions 
    all_events = get_events(boc, session_id, "VISp", bin_size)
    overlapping_cells = get_overlapping_cells(container_id, boc)
    overlapping_cell_events = all_events.loc[overlapping_cells]

    # Get full stimulus table for a given session
    stim_df = create_stim_df(boc, session_id)

    movie_df = stim_df[stim_df['stim_category']=="natural_movie_one"]
    bin_size = all_events.columns.to_list()[1]
    num_movie_chunks = int(len(movie_df)/bin_size)
    mean_response_mtx = np.zeros((num_movie_chunks,len(overlapping_cells)))

    for i in range(num_movie_chunks):
        chunk_df = movie_df.iloc[0 + (bin_size*i):(bin_size+(bin_size*i))]
        start,end = chunk_df.start.to_list()[0],chunk_df.end.to_list()[-1]
        try:
            response = np.mean(overlapping_cell_events.loc[:,np.arange(round_frame(start,overlapping_cell_events,'down'),round_frame(end,overlapping_cell_events,'up')+bin_size, bin_size)],axis=1)
            mean_response_mtx[i,:] = response
        except Exception as e:
            print(e)
    
    return mean_response_mtx


def get_mean_response_matrix_movie_three(session_id, container_id, bin_size, boc):
    # Get all event traces for overlapping neurons across sessions 
    all_events = get_events(boc, session_id, "VISp", bin_size)
    overlapping_cells = get_overlapping_cells(container_id, boc)
    overlapping_cell_events = all_events.loc[overlapping_cells]

    # Get full stimulus table for a given session
    stim_df = create_stim_df(boc, session_id)
    movie_df = stim_df[stim_df['stim_category']=="natural_movie_three"]
    
    divide_indices = find_divide_indices(movie_df)
    movie_df1, movie_df2, null = divide_stim_table(movie_df, divide_indices)
    
    # Generate response matrix for first epoch
    num_movie_chunks1 = int(len(movie_df1)/bin_size)
    mean_response_mtx1 = np.zeros((num_movie_chunks1,len(overlapping_cells)))
    for i in range(num_movie_chunks1):
        chunk_df = movie_df1.iloc[0 + (bin_size*i):(bin_size+(bin_size*i))]
        start,end = chunk_df.start.to_list()[0],chunk_df.end.to_list()[-1]
        try:
            response = np.mean(overlapping_cell_events.loc[:,np.arange(round_frame(start,overlapping_cell_events,'down'),round_frame(end,overlapping_cell_events,'up')+bin_size, bin_size)],axis=1)
            mean_response_mtx1[i,:] = response
        except Exception as e:
            print(e)
            
    # Generate response matrix for second epoch
    num_movie_chunks2 = int(len(movie_df2)/bin_size)
    mean_response_mtx2 = np.zeros((num_movie_chunks2,len(overlapping_cells)))
    for i in range(num_movie_chunks2):
        chunk_df = movie_df2.iloc[0 + (bin_size*i):(bin_size+(bin_size*i))]
        start,end = chunk_df.start.to_list()[0],chunk_df.end.to_list()[-1]
        try:
            response = np.mean(overlapping_cell_events.loc[:,np.arange(round_frame(start,overlapping_cell_events,'down'),round_frame(end,overlapping_cell_events,'up')+bin_size, bin_size)],axis=1)
            mean_response_mtx2[i,:] = response
        except Exception as e:
            print(e)
    
    return mean_response_mtx1, mean_response_mtx2


def get_all_relevant_tables_movie_one(container_id, bin_size, boc):

    # Select the relevant data for chosen container ID
    desired_container = boc.get_ophys_experiments(experiment_container_ids=[container_id])
    desired_container = sorted(desired_container, key=lambda x: x['session_type']) # sort based on session type so A comes first
    
    # Get full response matrix for each presentation on each day 
    response_matrix1 = get_mean_response_matrix_movie_one(desired_container[0]["id"], container_id, bin_size, boc)
    response_matrix2 = get_mean_response_matrix_movie_one(desired_container[1]["id"], container_id, bin_size, boc)
    response_matrix3 = get_mean_response_matrix_movie_one(desired_container[2]["id"], container_id, bin_size, boc)
    
    return response_matrix1, response_matrix2, response_matrix3


def get_all_relevant_tables_movie_three(container_id, bin_size, boc):

    # Select the relevant data for chosen container ID
    desired_container = boc.get_ophys_experiments(experiment_container_ids=[container_id])
    desired_container = sorted(desired_container, key=lambda x: x['session_type']) # sort based on session type so A comes first
    
    # Get full response matrix for each presentation on each day 
    response_matrix1, response_matrix2 = get_mean_response_matrix_movie_three(desired_container[0]["id"], container_id, bin_size, boc)
    
    return response_matrix1, response_matrix2


def pca_transform_data_three_epochs(mean_responses1, mean_responses2, mean_responses3, pca_n_components):
    
    # Concatenate mean matrices to fit PCA model
    concat_mean_matrices = np.concatenate((mean_responses1, mean_responses2, mean_responses3), axis=0)
    
    # Perform PCA
    warnings.filterwarnings('ignore')
    pca = PCA(n_components=pca_n_components) # create PCA model
    pca.fit_transform(concat_mean_matrices) # fit the model with the dataset
    
    pca_results_dict = {}
    transformed_data_pca1 = pca.transform(mean_responses1) # transform dataset 
    components = pca.components_ # list of principal components (PCs)
    explained_variance_ratio = pca.explained_variance_ratio_ # list of proportion of explained variance for each PC
    pca_results_dict[0] = {"components": components, "explained_variance_ratio": explained_variance_ratio} # add results to dictionary
    
    transformed_data_pca2 = pca.transform(mean_responses2) # transform dataset 
    components = pca.components_ # list of principal components (PCs)
    explained_variance_ratio = pca.explained_variance_ratio_ # list of proportion of explained variance for each PC
    pca_results_dict[1] = {"components": components, "explained_variance_ratio": explained_variance_ratio} # add results to dictionary
    
    transformed_data_pca3 = pca.transform(mean_responses3) # transform dataset 
    components = pca.components_ # list of principal components (PCs)
    explained_variance_ratio = pca.explained_variance_ratio_ # list of proportion of explained variance for each PC
    pca_results_dict[2] = {"components": components, "explained_variance_ratio": explained_variance_ratio} # add results to dictionary
    
    pca_results = pd.DataFrame(pca_results_dict).T # make a DataFrame
    
    # Concatenate dataframes for easy plotting
    transformed_data_pca1_df = pd.DataFrame(transformed_data_pca1)
    transformed_data_pca1_df["Session"] = list("1"*len(transformed_data_pca1))
    transformed_data_pca2_df = pd.DataFrame(transformed_data_pca2)
    transformed_data_pca2_df["Session"] = list("2"*len(transformed_data_pca2))
    transformed_data_pca3_df = pd.DataFrame(transformed_data_pca3)
    transformed_data_pca3_df["Session"] = list("3"*len(transformed_data_pca3))
    transformed_data_pca = pd.concat((transformed_data_pca1_df, transformed_data_pca2_df, transformed_data_pca3_df), axis=0)

    return transformed_data_pca, pca_results


def pca_transform_data_movie_three(mean_responses1, mean_responses2, pca_n_components):    
    # Concatenate mean matrices to fit PCA model
    concat_mean_matrices = np.concatenate((mean_responses1, mean_responses2), axis=0)
    
    # Perform PCA
    warnings.filterwarnings('ignore')
    pca = PCA(n_components=pca_n_components) # create PCA model
    pca.fit_transform(concat_mean_matrices) # fit the model with the dataset
    
    pca_results_dict = {}
    transformed_data_pca1 = pca.transform(mean_responses1) # transform dataset 
    components = pca.components_ # list of principal components (PCs)
    explained_variance_ratio = pca.explained_variance_ratio_ # list of proportion of explained variance for each PC
    pca_results_dict[0] = {"components": components, "explained_variance_ratio": explained_variance_ratio} # add results to dictionary
    
    transformed_data_pca2 = pca.transform(mean_responses2) # transform dataset 
    components = pca.components_ # list of principal components (PCs)
    explained_variance_ratio = pca.explained_variance_ratio_ # list of proportion of explained variance for each PC
    pca_results_dict[1] = {"components": components, "explained_variance_ratio": explained_variance_ratio} # add results to dictionary
    
    pca_results = pd.DataFrame(pca_results_dict).T # make a DataFrame
    
    # Concatenate dataframes for easy plotting
    transformed_data_pca1_df = pd.DataFrame(transformed_data_pca1)
    transformed_data_pca1_df["Session"] = list("1"*len(transformed_data_pca1))
    transformed_data_pca2_df = pd.DataFrame(transformed_data_pca2)
    transformed_data_pca2_df["Session"] = list("2"*len(transformed_data_pca2))
    transformed_data_pca = pd.concat((transformed_data_pca1_df, transformed_data_pca2_df), axis=0)

    return transformed_data_pca, pca_results


