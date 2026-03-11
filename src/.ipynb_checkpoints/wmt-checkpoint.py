import numpy as np
import pandas as pd
import xarray as xr
import gsw

cp = 3992   # Specific heat of seawater (J/kg/K)        
g = 9.81   # Acceleration due to gravity (m/s^2)

def buoyancy_flux(H, FWF, S, T, p, formula='default'):
    """
    Calculates buoyancy flux in terms of heat flux and freshwater flux with salinity.
    Uses the Gitrs SeaWater (gsw) Oceanographic Toolbox to calculate specific heat and thermal expansion coefficient of seawater.
    
    Parameters:
    H (xr.DataArray): Net heat flux at the ocean surface (W/m^2, averaged) or (W, total).
    FWF (xr.DataArray): Net freshwater flux at the ocean surface (kg/m^2 s, averaged) or (kg/s, total)
    S (xr.DataArray): Salinity of seawater (psu).
    T (xr.DataArray): Temperature of seawater (deg C).
    p (xr.DataArray or float): Pressure of seawater (dbar). Default value is 0.
    formula (str): The formula to use for calculating the buoyancy flux. Valid options are 'default' and 'hf_term'. Default is 'default'.
    
    Returns:
    B (xr.DataArray): Anveraged Buoyancy flux (m^2/s^3) or Total flux (m^4/s^3). The unit depends on the flux input.
    """
        # Calculate specific heat and thermal expansion coefficient using gsw toolbox
    density,alpha,beta = gsw.rho_alpha_beta(S, T, p)    # Density of seawater (kg/m^3)
                                                        # Thermal expansion coefficient of seawater (1/K)
                                                        # Saline contraction coefficient of seawater (1/psu)

    # Calculate buoyancy flux
    if formula == 'default':
        B_HF = -g/density * (alpha*H/cp)
        B_FWF = -g/density * (beta*FWF*S)
        B = B_HF + B_FWF
    elif formula == 'hf_term':
        B_HF = H
        B_FWF = beta*FWF*S*cp/alpha
        B = B_HF + B_FWF
    else:
        raise ValueError("Invalid formula specified. Valid options are 'default', 'hf_term'.") 
    
    df_list = [B, B_HF, B_FWF]

    for df in df_list:
        if formula == 'default':
            df.attrs['units'] = 'm$^{2}$/s$^{3}$'
            df.attrs['comments'] = 'Buoyancy flux calculated from net heat flux and net freshwater flux with salinity. lev indicates the depth of salinity.'
            df.attrs['standard_name'] = 'Buoyancy_flux'
        elif formula == 'hf_term' :
            df.attrs['units'] = 'W/m$^{-2}$'
            df.attrs['comments'] = 'Buoyancy flux in term of heat flux, calculated from net heat flux and net freshwater flux with salinity. lev indicates the depth of salinity.'
            df.attrs['standard_name'] = 'Heat_flux'
            
    return B, B_HF, B_FWF



def buoyancy_flux_heat(H, S, T, p):
    # Calculate specific heat and thermal expansion coefficient using gsw toolbox
    density,alpha,beta = gsw.rho_alpha_beta(S, T, p)    # Density of seawater (kg/m^3)
                                                        # Thermal expansion coefficient of seawater (1/K)
                                                        # Saline contraction coefficient of seawater (1/psu)
    bf = -g/density * (alpha*H/cp)
    
    return bf



def buoyancy_flux_water(FWF, S, T, p):
    # Calculate specific heat and thermal expansion coefficient using gsw toolbox
    density,alpha,beta = gsw.rho_alpha_beta(S, T, p)    # Density of seawater (kg/m^3)
                                                        # Thermal expansion coefficient of seawater (1/K)
                                                        # Saline contraction coefficient of seawater (1/psu)
    bf = -g/density * (beta*FWF*S)
    
    return bf

def buoyancy_flux_water_to_heat(FWF, S, T, p):
    # Calculate specific heat and thermal expansion coefficient using gsw toolbox
    density,alpha,beta = gsw.rho_alpha_beta(S, T, p)    # Density of seawater (kg/m^3)
                                                        # Thermal expansion coefficient of seawater (1/K)
                                                        # Saline contraction coefficient of seawater (1/psu)
    bf = beta*FWF*S*cp/alpha
    
    return bf

def trans_rate_heat(H, S, T, p):
    
    alpha = gsw.alpha(S,T,p)                             
    tr = -alpha*H/cp
    
    return tr

def trans_rate_water(FWF, S, T, p):
    
    beta = gsw.beta(S,T,p)                             
    tr = -beta*FWF*S
    
    return tr

def trans_rate_seaice(SeaiceWF, S, T, p):

    S_sea_ice = 5
    # S_sea_ice = S/3
    # density_freshwater = 1000
    # density_seaice = 925
    correction = 1-S_sea_ice/S
    
    beta = gsw.beta(S,T,p) 
    tr = -beta*correction*S*SeaiceWF
    
    return tr



def trans_rate_mix_sal(U, V, S, T, p):
    """ Calaculate the mixing term of water mass transformation
    """
    beta = gsw.beta(S,T,p)  
    # Calculate the gradients along the x and y dimensions
    gradient_x, gradient_y = np.gradient(S, axis=(1, 2))
    mix_x = U*gradient_x
    mix_y = V*gradient_y
    density = gsw.rho(S, T, p)
    tr = -beta*S*(mix_x + mix_y)*cp*density
    
    return tr

def calculate_wmt_monthly_from_dict(
    surface_flux,
    density_data,
    temperature_data,
    salinity_data,
    variable_name=None,
    density_min=1026,
    density_max=1031,
    step=0.1,
    unit="Sv",
    v_type="sea ice",
    model_name=None,
):
    """
    Calculate density-binned monthly WMT from externally provided surface flux
    and surface density / temperature / salinity fields.

    Parameters
    ----------
    surface_flux : xr.DataArray
        Surface forcing field used to compute WMT, e.g. sidd_weighted_monthly,
        hfds, or wfo.
    density_data : xr.DataArray
        Monthly surface density field.
    temperature_data : xr.DataArray
        Monthly surface temperature field.
    salinity_data : xr.DataArray
        Monthly surface salinity field.
    variable_name : str, optional
        Variable name, only used for model-specific logic if needed.
    density_min, density_max : float
        Density range for binning.
    step : float
        Density bin width.
    unit : str
        "Sv" or "Pg".
    v_type : str
        "sea ice", "water", "heat", or other.
    model_name : str, optional
        Only used if you still want to keep model-specific fixes.

    Returns
    -------
    xr.DataArray
        Density-binned WMT.
    """

    # -----------------------------
    # 1. Unit conversion
    # -----------------------------
    if unit == "Pg":
        Pg = 1e12
        surface_flux_in_unit = surface_flux / Pg * 365.25 * 24 * 360
    else:
        Sv = 1e6
        surface_flux_in_unit = surface_flux / Sv

    # -----------------------------
    # 2. Restrict to Southern Ocean
    # -----------------------------
    lat_coord, lon_coord = get_lat_lon_coords(surface_flux_in_unit)
    surface_flux_in_unit = surface_flux_in_unit.where(surface_flux_in_unit[lat_coord] < -45)

    # -----------------------------
    # 3. Optional model-specific fixes
    # -----------------------------
    if model_name is not None:
        if 'CMCC' in model_name:
            density_data = density_data[:, :-1, 1:-1]
            temperature_data = temperature_data[:, :-1, 1:-1]
            salinity_data = salinity_data[:, :-1, 1:-1]

        if 'NorESM' in model_name and variable_name is not None and 'sid' in variable_name:
            density_data = density_data[:, :-1, :]
            temperature_data = temperature_data[:, :-1, :]
            salinity_data = salinity_data[:, :-1, :]

        if 'FGOALS' in model_name:
            density_data = density_data.isel(j=slice(None, None, -1))
            temperature_data = temperature_data.isel(j=slice(None, None, -1))
            salinity_data = salinity_data.isel(j=slice(None, None, -1))

        if 'CAS-ESM' in model_name:
            surface_flux_in_unit = surface_flux_in_unit.roll(
                {surface_flux_in_unit.dims[2]: 1}, roll_coords=True
            )

    # -----------------------------
    # 4. Check shape consistency
    # -----------------------------
    if density_data.shape != surface_flux_in_unit.shape:
        raise ValueError(
            f"Shape mismatch: density_data {density_data.shape} does not match "
            f"surface_flux {surface_flux_in_unit.shape}."
        )

    # -----------------------------
    # 5. Compute buoyancy transformation rate
    # -----------------------------
    use_numpy = model_name is not None and (
        'CAS-ESM' in model_name or 'CESM2' in model_name or 'CIESM' in model_name
    )

    if v_type == "water":
        buoy_rate = wmt.trans_rate_water(
            surface_flux_in_unit, salinity_data, temperature_data, p=0
        )

    elif v_type == "heat":
        buoy_rate = wmt.trans_rate_heat(
            surface_flux_in_unit, salinity_data, temperature_data, p=0
        )

    elif v_type == "sea ice":
        if use_numpy:
            buoy_values = wmt.trans_rate_seaice(
                surface_flux_in_unit.values,
                salinity_data.values,
                temperature_data.values,
                p=0
            )
            buoy_rate = xr.DataArray(
                buoy_values,
                dims=surface_flux_in_unit.dims,
                coords=surface_flux_in_unit.coords
            )
        else:
            buoy_rate = wmt.trans_rate_seaice(
                surface_flux_in_unit,
                salinity_data,
                temperature_data,
                p=0
            )
    else:
        buoy_rate = surface_flux_in_unit

    # -----------------------------
    # 6. Convert to WMT
    # -----------------------------
    wmt_rate = -buoy_rate / step

    # -----------------------------
    # 7. Density bins
    # -----------------------------
    density_centers = np.arange(density_min, density_max, step)
    accumulated_wmt = []

    # -----------------------------
    # 8. Integrate WMT in density space
    # -----------------------------
    for center in density_centers:
        lower_bound = center - step / 2
        upper_bound = center + step / 2

        mask = (density_data >= lower_bound) & (density_data < upper_bound)

        if use_numpy:
            accumulated_value = wmt_rate.where(mask.values).sum()
        else:
            accumulated_value = wmt_rate.where(mask).sum()

        accumulated_wmt.append(accumulated_value / 12)

    # -----------------------------
    # 9. Output
    # -----------------------------
    accumulated_wmt_array = xr.DataArray(
        accumulated_wmt,
        coords={'density': density_centers},
        dims=['density']
    )

    return accumulated_wmt_array


def calculate_wmt_monthly(
    surface_flux,
    density_data,
    temperature_data,
    salinity_data,
    density_min=1026,
    density_max=1031,
    step=0.1,
    unit="Sv",
    v_type="sea ice",
):
    """
    Calculate density-binned monthly water-mass transformation (WMT).

    Parameters
    ----------
    surface_flux : xr.DataArray
        Surface forcing field used to compute WMT.
    density_data : xr.DataArray
        Monthly surface density field.
    temperature_data : xr.DataArray
        Monthly surface temperature field.
    salinity_data : xr.DataArray
        Monthly surface salinity field.
    density_min : float
        Minimum density for binning.
    density_max : float
        Maximum density for binning.
    step : float
        Density bin width.
    unit : str
        Output input unit handling: "Sv" or "Pg".
    v_type : str
        Type of transformation: "sea ice", "water", "heat", or other.

    Returns
    -------
    xr.DataArray
        Density-binned WMT.
    """

    # convert input flux to requested unit
    if unit == "Pg":
        Pg = 1e12
        surface_flux_in_unit = surface_flux / Pg * 365.25 * 24 * 360
    else:
        Sv = 1e6
        surface_flux_in_unit = surface_flux / Sv

    # restrict to Southern Ocean south of 45S
    lat_coord, lon_coord = get_lat_lon_coords(surface_flux_in_unit)
    surface_flux_in_unit = surface_flux_in_unit.where(surface_flux_in_unit[lat_coord] < -45)

    # basic shape check
    if density_data.shape != surface_flux_in_unit.shape:
        raise ValueError(
            f"Shape mismatch: density_data {density_data.shape} does not match "
            f"surface_flux {surface_flux_in_unit.shape}."
        )

    # compute buoyancy transformation rate
    if v_type == "water":
        buoy_rate = wmt.trans_rate_water(
            surface_flux_in_unit, salinity_data, temperature_data, p=0
        )
    elif v_type == "heat":
        buoy_rate = wmt.trans_rate_heat(
            surface_flux_in_unit, salinity_data, temperature_data, p=0
        )
    elif v_type == "sea ice":
        buoy_rate = wmt.trans_rate_seaice(
            surface_flux_in_unit, salinity_data, temperature_data, p=0
        )
    else:
        buoy_rate = surface_flux_in_unit

    # convert buoyancy forcing to WMT
    wmt_rate = -buoy_rate / step

    # define density bins
    density_centers = np.arange(density_min, density_max, step)
    accumulated_wmt = []

    # integrate WMT within each density bin
    for center in density_centers:
        lower_bound = center - step / 2
        upper_bound = center + step / 2

        mask = (density_data >= lower_bound) & (density_data < upper_bound)
        accumulated_value = wmt_rate.where(mask).sum()

        # monthly climatological mean contribution
        accumulated_wmt.append(accumulated_value / 12)

    # return density-space WMT
    accumulated_wmt_array = xr.DataArray(
        accumulated_wmt,
        coords={"density": density_centers},
        dims=["density"]
    )

    return accumulated_wmt_array


def tr_to_fr(transformation_rate):
    """
    Convert transformation rate to formation rate for 1D and 2D arrays.
    """
    # Ensure the input is a DataArray
    if not isinstance(transformation_rate, xr.DataArray):
        raise ValueError("Input must be an xarray DataArray")
    
    # Initialize formation rate array with the same shape as the transformation rate array
    fr = xr.zeros_like(transformation_rate)
    
    # Handle 1D case
    if transformation_rate.ndim == 1:
        for index, value in enumerate(transformation_rate):
            value = value.item()  # Get the value from the DataArray
            if value > 0:
                # index_2 = index + 1
                fr[index] += abs(value)
                fr[index-1] -= abs(value)
            else:
                # index_2 = index - 1
                fr[index-1] += abs(value)
                fr[index] -= abs(value)
            # fr[index] -= abs(value)
            # if index_2 >= 0 and index_2 < len(fr):
            #     fr[index_2] += abs(value)
    
    # Handle 2D case
    elif transformation_rate.ndim == 2:
        for i in range(transformation_rate.shape[0]):
            for j in range(transformation_rate.shape[1]):
                value = transformation_rate[i, j].item()
                if value > 0:
                    # index_2 = j + 1
                    fr[i,j] += abs(value)
                    fr[i,j-1] -= abs(value)
                else:
                    fr[i,j-1] += abs(value)
                    fr[i,j] -= abs(value)
                    # index_2 = j - 1

                # fr[i, j] -= abs(value)

                # if index_2 >= 0 and index_2 < transformation_rate.shape[1]:
                #     fr[i, index_2] += abs(value)
    
    else:
        raise ValueError("Input must be a 1D or 2D xarray DataArray")
    
    return fr


def load_data_in_range(start_year, end_year, file_pattern):
    # Calculate the start and end years for each dataset based on the 10-year interval
    start_year_range = (start_year // 10) * 10
    end_year_range = ((end_year // 10) + 1) * 10

    # Create an empty list to store the loaded datasets
    datasets = []

    # Loop through each range of years
    for year_range in range(start_year_range, end_year_range, 10):
        # Construct the file name for the dataset
        file_name = file_pattern.format(year_range, year_range + 9)

        # Load the dataset
        dataset = xr.open_dataset(file_name)

        # Append the dataset to the list
        datasets.append(dataset)

    # Concatenate the datasets along the time dimension
    combined_dataset = xr.concat(datasets, dim='time')

    # Select the desired range of time steps
    subset_dataset = combined_dataset.sel(time=slice(f'{start_year}-01-01', f'{end_year}-12-31'))

    return subset_dataset

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
