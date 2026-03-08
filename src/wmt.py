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
