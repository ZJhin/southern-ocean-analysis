from clef.code import *
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import glob
import os
from typing import List, Dict

class Zlook:
    """
    A class to search, retrieve, and process CMIP6 datasets using constraints.
    """

    def __init__(self, constraints: Dict[str, str]) -> None:
        """
        Initialize the Zlook class with search constraints.
        Args:
            constraints (dict): A dictionary of search constraints, e.g., {'variable': 'tas'}.
        """
        self._constraints = constraints
        self._variable = constraints.get("variable")
        db = connect()
        s = Session()
        self._df = search(s, project="CMIP6", latest=True, **self._constraints)

    def path(self, path_num: int = 0) -> str:
        """
        Get the path for a specific row in the search results.
        Args:
            path_num (int): The index of the path to retrieve.
        Returns:
            str: The path as a string.
        """
        if self.df_empty():
            raise ValueError("There is no document in the path")
        if path_num >= len(self._df):
            raise IndexError(f"path_num {path_num} is out of bounds.")
        return self._df.iloc[path_num]["path"]

    def file_path(self, path_num: int = 0, file_num: int = 0) -> str:
        """
        Get the file path for a specific path and file index.
        Args:
            path_num (int): The index of the path.
            file_num (int): The index of the file.
        Returns:
            str: The full file path.
        """
        path = self.path(path_num)
        files = self.file_in_path(path)
        if file_num >= len(files):
            raise IndexError(f"file_num {file_num} is out of bounds.")
        return os.path.join(path, files[file_num])

    def file_in_path(self, path: str) -> List[str]:
        """
        List all files in the given directory path.
        Args:
            path (str): The directory path.
        Returns:
            list: Sorted list of file names in the directory.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"The path {path} does not exist.")
        return sorted(os.listdir(path))

    def df_empty(self) -> bool:
        """
        Check if the DataFrame is empty.
        Returns:
            bool: True if empty, False otherwise.
        """
        return self._df.empty

    def kw_search(self, dir_path: str, keywords: str) -> List[str]:
        """
        Search for NetCDF files in a directory containing specific keywords.
        Args:
            dir_path (str): The directory to search.
            keywords (str): Keywords to match in file names.
        Returns:
            list: A list of matching file paths.
        """
        return sorted(glob.glob(f"{dir_path}/*{keywords}*.nc"))

    def load_data_range(self, data_files: List[str], start_year: int, end_year: int) -> List[str]:
        """
        Filter files based on a given date range.
        Args:
            data_files (list): A list of file names.
            start_year (int): Start of the range.
            end_year (int): End of the range.
        Returns:
            list: Filtered list of file names.
        """
        selected_files = []
        for file_name in data_files:
            try:
                file_range = file_name.split("_")[-1].split(".")[0].split("-")
                file_start_year, file_end_year = int(file_range[0][:4]), int(file_range[1][:4])
                if file_end_year >= start_year and file_start_year <= end_year:
                    selected_files.append(file_name)
            except (IndexError, ValueError):
                continue
        return selected_files

    def smart_open(self, data_path: str, start_year: int, end_year: int):
        """
        Open files within a specified time range and assemble them.
        Args:
            data_path (str): Directory containing the data files.
            start_year (int): Start of the time range.
            end_year (int): End of the time range.
        Returns:
            xr.Dataset: Assembled xarray dataset.
        """
        data_files = self.file_in_path(data_path)
        files_to_open = self.load_data_range(data_files, start_year, end_year)
        full_paths = [os.path.join(data_path, file) for file in files_to_open]
        return xr.open_mfdataset(full_paths, combine="nested", concat_dim="time", parallel=True)[self.v()].sel(
            time=slice(str(start_year), str(end_year))
        )

    def v(self) -> str:
        """
        Get the variable name.
        Returns:
            str: The variable name.
        """
        return self._variable

    def show(self):
        """
        Show the search results DataFrame.
        Returns:
            pd.DataFrame: The search results DataFrame.
        """
        return self._df


class Zsearch:
    """
    A class to search and explore CMIP6 datasets.
    """

    def __init__(self, constraints: Dict[str, str]) -> None:
        """
        Initialize the Zsearch class with search constraints.
        Args:
            constraints (dict): A dictionary of search constraints.
        """
        self._constraints = constraints
        db = connect()
        s = Session()
        allvalues = ["variable_id"]
        fixed = ["source_id", "member_id"]
        self.results, selection = matching(s, allvalues, fixed, project="CMIP6", **constraints)
        self.selection = pd.DataFrame(selection)

    def variable_paths(self, source_id: str) -> List[str]:
        """
        Get all file paths for a specific model.
        Args:
            source_id (str): The source ID of the model.
        Returns:
            list: List of file paths for the model.
        """
        if source_id not in self.selection.index:
            raise KeyError(f"Source ID {source_id} not found.")
        row_indices = self.selection.loc[source_id]["index"]
        return [self.results.loc[idx]["path"] for idx in row_indices[0]]

    def model_list(self) -> List[str]:
        """
        Get a list of unique model names.
        Returns:
            list: List of model names.
        """
        return self.selection.index.get_level_values(0).unique().tolist()

    def show(self):
        """
        Show the selection DataFrame.
        Returns:
            pd.DataFrame: The selection DataFrame.
        """
        return self.selection


def variable_df(variable_list: List[str], realm: List[str] = ['ocean'], frequency: List[str] = ['mon'],
                experiment: List[str] = ['historical'], member_id: List[str] = ['r1i1p1f1', 'r1i1p1f2'],
                grid_label: List[str] = ['gn', 'gr']) -> Zsearch:
    """
    Generate a Zsearch object based on variable constraints.
    Args:
        variable_list (list): List of variable IDs to search for.
        realm (list): List of realms (default is ['ocean']).
        frequency (list): Frequency of data (default is ['mon']).
        experiment (list): List of experiments (default is ['historical']).
        member_id (list): List of member IDs (default is ['r1i1p1f1', 'r1i1p1f2']).
        grid_label (list): List of grid labels (default is ['gn', 'gr']).
    Returns:
        Zsearch: An instance of the Zsearch class.
    """
    constraints = {
        'variable_id': variable_list,
        # 'realm': realm,
        'frequency': frequency,
        'experiment_id': experiment,
        'member_id': member_id,
        'grid_label': grid_label
    }
    return Zsearch(constraints)


def load_data(model: str, yrst: str, yren: str, df: Zsearch, variable_name: str) -> xr.DataArray:
    """
    Load and concatenate NetCDF files for a given model and variable.
    Args:
        model (str): The model name to search for.
        yrst (str): Start year for the data.
        yren (str): End year for the data.
        df (Zsearch): An instance of Zsearch containing the search results.
        variable_name (str): The variable name to load.
    Returns:
        xr.DataArray: The loaded data as an xarray DataArray.
    """
    path = df.variable_paths(model)[0] + '/*.nc'
    if variable_name == 'areacello':
        data = xr.open_mfdataset(path)[variable_name]
    else:
        data = xr.open_mfdataset(path)[variable_name].sel(time=slice(yrst, yren))
    
    return data


def get_lat_lon_coords(dataset: xr.Dataset) -> tuple:
    """
    Determine the latitude and longitude coordinate names in an xarray dataset.

    Args:
        dataset (xr.Dataset): The dataset to check.

    Returns:
        tuple: A tuple containing the names of the latitude and longitude coordinates (lat_coord, lon_coord).
    """
    lat_coord = (
        'nav_lat' if 'nav_lat' in dataset.coords else
        'latitude' if 'latitude' in dataset.coords else
        'lat' if 'lat' in dataset.coords else None
    )
    lon_coord = (
        'nav_lon' if 'nav_lon' in dataset.coords else
        'longitude' if 'longitude' in dataset.coords else
        'lon' if 'lon' in dataset.coords else None
    )

    if not lat_coord or not lon_coord:
        raise ValueError("Latitude and/or longitude coordinates not found in the dataset.")

    return lat_coord, lon_coord


def plot_south_polar(data, vmax = 0.2,vmin = -0.2, cmap = 'RdBu_r'):
    """
    Plot the first available time slice of the data in South Polar stereographic coordinates.
    
    Parameters:
    - data: xarray DataArray or Dataset containing the data to plot.
    
    Example usage:
    plot_south_polar(siv_cm)
    """
    # Select the first available time slice if time dimension exists
    if 'time' in data.dims:
        data_selected = data.isel(time=0)  # Select the first time index
        time_value = pd.to_datetime(data.coords['time'].values[0]).strftime('%Y-%m')  # Format as YYYY-MM
    else:
        data_selected = data
        time_value = "Unknown Time"  # Placeholder if no time dimension is found

    # Set up the figure and projection
    fig, ax = plt.subplots(figsize=(5, 5), 
                           subplot_kw=dict(projection=ccrs.Orthographic(0, -90)))

    # Determine latitude and longitude coordinates
    lat_coord, lon_coord = get_lat_lon_coords(data_selected)
    
    # Plot the data with appropriate transformations
    data_selected.plot(
        x=lon_coord,
        y=lat_coord,
        ax=ax,
        transform=ccrs.PlateCarree(),  # Use PlateCarree projection for source data
        cmap=cmap,  # Or any other preferred colormap
        cbar_kwargs={'shrink': 0.5, 'label': 'Data Value'},  # Adjust colorbar
        vmax = vmax,
        vmin = vmin
    )
    
    # Set map extent and add gridlines
    ax.set_extent([-180, 180, -90, -50], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    # Add coastlines for context
    ax.coastlines()

    # Set the title to include the formatted time
    plt.title(f"Data for {time_value} in South Polar Projection")
    plt.show()

# Example usage:
# plot_south_polar(siv_cm)

