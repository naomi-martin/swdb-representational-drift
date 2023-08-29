from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import pandas as pd

"""
This file contains utility functions for getting events.
"""

def get_events(boc: BrainObservatoryCache, session_id: int, region: str, bin_width: int = 1) -> pd.DataFrame:
    """
    This function returns binned event magnitudes for one session that includes all cells in a region.
    Keyword arguments:
    boc -- Brain Observatory Cache object
    session_id -- Session ID of the returned events
    region -- Brain region to restrict returned cells to
    bin_width -- Window size for binning the events
    Returns:
    Dataframe indexed by cell_ids containing event magnitudes summed over windows
    """
    
    data_set = boc.get_ophys_experiment_data(ophys_experiment_id=session_id)
    cell_ids = data_set.get_cell_specimen_ids()
    cell_indices = data_set.get_cell_specimen_indices(cell_ids)
    areas = [cell['area'] for cell in boc.get_cell_specimens(ids=cell_ids)]
    zipped = [(area, cell_index, cell_id) for area, cell_index, cell_id in zip(areas,cell_indices, cell_ids) if area == "VISp"]
    region_df = pd.DataFrame(zipped, columns=["area", "cell_index", "cell_id"]).set_index("cell_id")
    unbinned_df = pd.DataFrame(boc.get_ophys_experiment_events(ophys_experiment_id=session_id)[region_df.cell_index.values], index=region_df.index)
    return unbinned_df.rolling(window=bin_width, min_periods=None, axis=1, step=10).sum().fillna(0)