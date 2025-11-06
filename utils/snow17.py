import os
import re
import glob
import numpy as np
import rasterio
from datetime import datetime
from dotenv import load_dotenv


def ensure_directory(path):
    """
    Ensures that a directory exists and is actually a directory.
    If a file exists with that name, removes it first.
    If a directory exists, does nothing.
    """
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            return  # Directory already exists, nothing to do
    os.makedirs(path, exist_ok=True)


def read_geotiff(path):
    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float32)
        profile = src.profile
    array[array == src.nodata] = np.nan
    return array, profile

def write_geotiff(path, array, profile):
    nodata_value = -9999.0
    array = np.where(np.isnan(array), nodata_value, array)
    export_profile = profile.copy()  # Hacer copia del perfil
    export_profile.update(driver='GTiff', dtype=rasterio.float32, count=1, compress='lzw', nodata=nodata_value)
    with rasterio.open(path, 'w', **export_profile) as dst:
        dst.write(array.astype(np.float32), 1)  # Asegurarse que array es float32

def parse_datetime_from_filename(filename):
    match = re.search(r'(\d{8,12})\.tif$', filename)
    if match:
        date_str = match.group(1)
        fmt = '%Y%m%d%H%M' if len(date_str) == 12 else '%Y%m%d'
        return datetime.strptime(date_str, fmt)
    return None


  #SCF: 0.9684652219729578
  #MFMAX: 1.0363317019171228
  #MFMIN: 0.05
  #PXTEMP: 0.5
  #UADJ: 0.08
  #PLWHC: 0.02
  
def dem_to_snow17_param(dem):
    dem = np.where(dem == -9999.0, np.nan, dem)
    # Asignación directa según altitud (hardcodeado)
    SCF    = np.where(dem >= 1500, 1.2, 1.1)
    MFMAX  = np.where(dem >= 1500, 0.9, 0.95)
    MFMIN  = np.where(dem >= 1500, 0.07, 0.08)
    PXTEMP = np.where(dem >= 1500, 1.5, 1.5)
    UADJ   = np.where(dem >= 1500, 0.12, 0.08)
    PLWHC  = np.where(dem >= 1500, 0.02, 0.12)
    MBASE  = np.where(dem >= 1500, 0.0, 0.0)
    TIPM   = np.where(dem >= 1500, 0.25, 0.25)
    NMF    = np.where(dem >= 1500, 0.15, 0.15)
    DAYGM  = np.where(dem >= 1500, 0.3, 0.3)
    params = {
        'SCF': SCF,
        'PXTEMP': PXTEMP,
        'MFMAX': MFMAX,
        'MFMIN': MFMIN,
        'UADJ': UADJ,
        'MBASE': MBASE,
        'TIPM': TIPM,
        'PLWHC': PLWHC,
        'NMF': NMF,
        'DAYGM': DAYGM
    }
    return params

def create_initial_states(template):
    template = np.where(np.isnan(template), 0, template)
    return {
        'W_i': np.zeros_like(template),
        'ATI': np.zeros_like(template),
        'W_q': np.zeros_like(template),
        'Deficit': np.zeros_like(template)
    }

def create_input_layers(dem, prec, temp, date_str, params):
    jday = datetime.strptime(date_str, '%Y-%m-%d').timetuple().tm_yday
    return {
        'elev': dem,
        'prcp': prec,
        'tavg': temp,
        'jday': np.full_like(dem, jday, dtype=np.float32),
        **params
    }

# -------------- SNOW-17 GRIDDING FUNCTION -------------- #

def snow17_gridded(x, ini_states, dtt, dtp, hemisphere='N'):

    """
    :param x:  static parameters
    :param ini_states: initial states rasters (dict) o None
    :param dtt: delta time of temperature (in hours)
    :param dtp: delta time of precipitation (in hours)
    :param hemisphere: 'N' o 'S'
    :return:
    """

    SCF = x['SCF']
    PXTEMP = x['PXTEMP']
    MFMAX = x['MFMAX']
    MFMIN = x['MFMIN']
    UADJ = x['UADJ']
    MBASE = x['MBASE']
    TIPM = x['TIPM']
    PLWHC = x['PLWHC']
    NMF = x['NMF']
    DAYGM = x['DAYGM']

    elev = x['elev']
    jdate = x['jday']

    # Set initial states
    W_i = ini_states['W_i']
    ATI = ini_states['ATI']
    W_q = ini_states['W_q']
    Deficit = ini_states['Deficit']

    # Current temperature and precipitation
    Ta = x['tavg']
    Pr = x['prcp']

    # FORM OF PRECIPITATION
    SNOW = np.where(Ta <= PXTEMP, Pr, 0)
    RAIN = np.where(Ta > PXTEMP, Pr, 0)

    # ACCUMULATION OF THE SNOW COVER
    Pn = SNOW * SCF
    W_i += Pn
    E = np.zeros_like(Pr)

   # Adjustment absed on hemisphere
    ref_day = 80 if hemisphere.upper() == 'N' else 264  # 21 march for north, 21 september for south
    N_ref = jdate - ref_day
    
    Sv = (0.5 * np.sin((N_ref * 2 * np.pi) / 365)) + 0.5
    Av = np.ones_like(Pr)
    Mf = dtt / 6 * ((Sv * Av * (MFMAX - MFMIN)) + MFMIN)

    # New snow temperature and heat deficit from new snow
    T_snow_new = np.where(Ta < 0, Ta, 0)

    # Change in the heat deficit due to new snowfall [mm], 80 cal/g: latent heat of fusion, 0.5 cal/g/C:
    # specific heat of ice
    delta_HD_snow = -(T_snow_new * Pn) / (80.0 / 0.5)
    # Heat Exchange due to a temperature gradient change in heat deficit due to a temperature gradient
    # [mm], 80 cal/g: latent heat of
    delta_HD_T = NMF * dtp / 6.0 * Mf / MFMAX * (ATI - T_snow_new)

    ATI = np.where(Pn > 1.5 * dtp, T_snow_new, ATI + (1 - ((1 - TIPM) ** (dtt / 6))) * (Ta - ATI))
    ATI = np.minimum(ATI, 0)

    # SNOW MELT
    T_rain = np.maximum(Ta, 0)      # Temperature of rain (deg C), Ta or 0C, whichever greater

    stefan = 6.12 * (10 ** (-10))  # Stefan-Boltzman constant (mm/K/hr)
    e_sat = 2.7489 * (10 ** 8) * np.exp((-4278.63 / (Ta + 242.792)))  # Saturated vapor pressure at Ta (mb)
    P_atm = 33.86 * (29.9 - (0.335 * (elev / 100)) + (0.00022 * ((elev / 100) ** 2.4)))  # Atmospheric pressure (mb)
    term1 = stefan * dtp * (((Ta + 273) ** 4) - (273 ** 4))
    term2 = 0.0125 * RAIN * T_rain
    term3 = 8.5 * UADJ * (dtp / 6) * ((0.9 * e_sat - 6.11) + (0.00057 * P_atm * Ta))

    Melt = np.where(
        RAIN > 0.25 * dtp,
        np.maximum(term1 + term2 + term3, 0),
        np.where(
            (RAIN <= 0.25 * dtp) & (Ta > MBASE),
            np.maximum(((Mf * (Ta - MBASE) * (dtp / dtt)) + (0.0125 * RAIN * T_rain)), 0),
            0
        )
    )

    # Ripeness of the snow cover W_i : water equivalent of the ice portion of the snow cover W_q :
    # liquide water held by the snow W_qx: liquid water storage capacity Qw : Amount of available water
    # due to melt and rain

    # Update Deficit based on new snowfall and temperature gradient
    Deficit = np.maximum(Deficit + delta_HD_snow + delta_HD_T, 0)   # Deficit <- heat deficit [mm]
    Deficit = np.where(Deficit > (0.33 * W_i), 0.33 * W_i, Deficit)

    # Condition 01
    W_i = np.where(Melt < W_i, W_i-Melt,W_i)

    # Calculate conditions for updating states
    Qw = np.where(Melt < W_i, Melt + RAIN, 0)
    W_qx = np.where(Melt < W_i, PLWHC * W_i, 0)

    con_01 = (Melt < W_i) & ((Qw + W_q) > (Deficit + Deficit * PLWHC + W_qx))

    # con_02 = (Melt < W_i) & ((Qw + W_q) >= Deficit) - con_01

    condition = ((Melt < W_i) & ((Qw + W_q) >= Deficit)).astype(float)
    # Subtract the value of con_01 based on the condition
    con_02 = condition - con_01

    condition = ((Melt < W_i) & ((Qw + W_q) < Deficit)).astype(float)
    con_03 = condition - con_02
    # con_03 = ((Melt < W_i) & ((Qw + W_q) < Deficit)) - con_02
    con_03[con_03 < 0] = 0

    # Update states based on conditions
    E = np.where(con_01, Qw + W_q - W_qx - Deficit - (Deficit * PLWHC), E) # Excess liquid water [mm]
    W_i = np.where(con_01, W_i + Deficit, W_i)     # W_i increases because water refreezes as heat deficit is decreased
    W_q = np.where(con_01, W_qx + PLWHC * Deficit, W_q) # fills liquid water capacity
    Deficit = np.where(con_01, 0, Deficit)

    E = np.where(con_02, 0, E)
    W_i = np.where(con_02, W_i + Deficit, W_i)  # W_i increases because water refreezes as heat deficit is decreased
    W_q = np.where(con_02, W_q + Qw - Deficit, W_q)
    Deficit = np.where(con_02, 0, Deficit)

    E = np.where(con_03, 0, E)  # THEN the snow is NOT yet ripe
    W_i = np.where(con_03, W_i + Qw + W_q, W_i) # W_i increases because water refreezes as heat deficit is decreased
    Deficit = np.where(con_03, Deficit - Qw - W_q, Deficit)

    # Condition 02
    # Update for melt greater or equal to W_i
    Melt = np.where(Melt >= W_i, W_i + W_q, Melt)
    W_i = np.where(Melt >= W_i, 0, W_i)
    W_q = np.where(Melt >= W_i, 0, W_q)
    Qw = np.where(Melt >= W_i, Melt + RAIN, Qw)
    E = Qw

    ATI = np.where(Deficit == 0, 0, ATI)

    # First, calculate and store the results based on the original W_i condition
    condition = W_i > DAYGM
    # Define a small positive threshold to avoid division by zero
    epsilon = 1E-15

    # Calculate gmwlos, first check if W_i is greater than epsilon to avoid division by zero.
    gmwlos = np.where((W_i > DAYGM) & (W_i > epsilon), (DAYGM / np.maximum(W_i, epsilon)) * W_q, 0)

    # gmwlos = np.where(condition, (DAYGM / W_i) * W_q, 0)
    gmslos = np.where(condition, DAYGM, 0)

    # Use the stored conditions to update gmro, W_i, and W_q
    gmro = np.where(condition, gmwlos + gmslos, W_i + W_q)
    updated_W_i = np.where(condition, W_i - gmslos, 0)  # Save the calculation results before updating W_i.
    updated_W_q = np.where(condition, W_q - gmwlos, 0)  # Save the calculation results before updating W_q

    # Now it is safe to update W_i and W_q
    W_i = updated_W_i
    W_q = updated_W_q

    # **************************************************************** #
    E = np.where(W_i > DAYGM, E + gmro, E + gmro)
    SWE = np.where(W_i > DAYGM, W_i + W_q, 0)

    # Packing the results and initial states
    results = {
        'Rain': RAIN,
        'Tavg': Ta,
        'SNOW': SNOW,
        'SNOW_E': Pn,
        'Melt': Melt,
        'E': E,
        'SWE': SWE
    }

    ini_states = {
        'W_i': W_i,
        'ATI': ATI,
        'W_q': W_q,
        'Deficit': Deficit
    }

    return results, ini_states

# -------------------- MAIN FUNCTION --------------------

def main():
    """
    Main function for SNOW-17 gridded model.
    Reads all configuration from .env file:
    - dem_path: Path to DEM file
    - rain_output_path: Output path from preprocessing (rain files)
    - tavg_output_path: Output path from preprocessing (temperature files)
    - rainmelt_output_path: Output path for rainmelt results
    - start_date, end_date: Processing date range
    
    Initial states are created as zeros (no file loading).
    """
    # Load environment variables from .env (find it in parent directory)
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    load_dotenv(env_path)
    
    # Read paths from .env
    dem_path = os.getenv('dem_path')
    rain_output_path = os.getenv('rain_output_path')
    tavg_output_path = os.getenv('tavg_output_path')
    rainmelt_output_path = os.getenv('rainmelt_output_path')
    start_date_str = os.getenv('start_date')
    end_date_str = os.getenv('end_date')
    
    # Validate required environment variables
    required_vars = {
        'dem_path': dem_path,
        'rain_output_path': rain_output_path,
        'tavg_output_path': tavg_output_path,
        'rainmelt_output_path': rainmelt_output_path,
        'start_date': start_date_str,
        'end_date': end_date_str
    }
    
    for var_name, var_value in required_vars.items():
        if not var_value:
            raise ValueError(f"Missing required environment variable: {var_name}")
    
    # Parse dates
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    print(f"[SNOW17] Configuration loaded from .env")
    print(f"[SNOW17] DEM: {dem_path}")
    print(f"[SNOW17] Rain input: {rain_output_path}")
    print(f"[SNOW17] Tavg input: {tavg_output_path}")
    print(f"[SNOW17] Rainmelt output: {rainmelt_output_path}")
    print(f"[SNOW17] Processing period: {start_date_str} to {end_date_str}")
    
    # Create output directory
    ensure_directory(rainmelt_output_path)
    print(f"[SNOW17] Output directory ready: {rainmelt_output_path}")
    
    # Read DEM and create parameters
    if not os.path.exists(dem_path):
        raise FileNotFoundError(f"DEM file not found: {dem_path}")
    
    dem, profile = read_geotiff(dem_path)
    dem_params = dem_to_snow17_param(dem)
    print(f"[SNOW17] DEM loaded with shape: {dem.shape}")
    
    # Initialize states as zeros (default)
    ini_states = create_initial_states(dem)
    print(f"[SNOW17] Initial states created (zeros)")
    
    hemisphere = 'N'
    dtt = 24  # hours: 24 for daily timestep
    dtp = 24  # hours: 24 for daily timestep
    
    # Get rain and temperature files
    prec_files = sorted([f for f in glob.glob(os.path.join(rain_output_path, '*.tif')) if parse_datetime_from_filename(os.path.basename(f))])
    temp_files = sorted([f for f in glob.glob(os.path.join(tavg_output_path, '*.tif')) if parse_datetime_from_filename(os.path.basename(f))])
    
    if not prec_files:
        raise FileNotFoundError(f"No precipitation files found in: {rain_output_path}")
    if not temp_files:
        raise FileNotFoundError(f"No temperature files found in: {tavg_output_path}")
    
    print(f"[SNOW17] Found {len(prec_files)} precipitation files")
    print(f"[SNOW17] Found {len(temp_files)} temperature files")
    
    # Process each day
    processed_count = 0
    for pfile, tfile in zip(prec_files, temp_files):
        current_date = parse_datetime_from_filename(os.path.basename(pfile))
        
        # Skip if outside date range
        if current_date < start_date or current_date > end_date:
            continue
        
        if not os.path.exists(tfile):
            print(f"[SNOW17] Warning: Temperature file not found for {current_date.strftime('%Y-%m-%d')}, skipping")
            continue
        
        print(f"[SNOW17] Processing {current_date.strftime('%Y-%m-%d')}")
        prec, _ = read_geotiff(pfile)
        temp, _ = read_geotiff(tfile)
        x = create_input_layers(dem, prec, temp, current_date.strftime('%Y-%m-%d'), dem_params)
        results, ini_states = snow17_gridded(x, ini_states, dtt=dtt, dtp=dtp, hemisphere=hemisphere)
        
        out_arr = np.where(np.isnan(dem), np.nan, results['E'])
        date_str = current_date.strftime('%Y%m%d')
        out_file = os.path.join(rainmelt_output_path, f"rainmelt.{date_str}.tif")
        
        write_geotiff(out_file, out_arr, profile)
        processed_count += 1
        print(f"[SNOW17] Saved: {os.path.basename(out_file)}")
    
    print(f"[SNOW17] Done. Processed {processed_count} days.")
    return rainmelt_output_path


if __name__ == '__main__':
    main()
