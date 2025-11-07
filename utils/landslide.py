"""
Landslide Model - Standalone version
Combines Random Forest and Physical (Infinite Slope) models
Reads all configuration from .env
"""

import os
import numpy as np
import rasterio
import pandas as pd
from glob import glob
from scipy.stats import norm
from rasterio.warp import reproject, Resampling
import joblib
from datetime import datetime
from dotenv import load_dotenv


# Physical constants
G = 9.81
DW = 1000
DS = 2000
B = 30


def ensure_directory(path):
    """
    Ensures that a directory exists and is actually a directory.
    If a file exists with that name, removes it first.
    If a directory exists, does nothing.
    """
    # Remove trailing slashes
    path = path.rstrip('/')
    
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            return  # Directory already exists, nothing to do
    os.makedirs(path, exist_ok=True)


def load_config():
    """Load all configuration from .env file"""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    load_dotenv(env_path)
    
    config = {
        'CREST_output_path': os.getenv('CREST_output_path'),
        'rainmelt_output_path': os.getenv('rainmelt_output_path'),
        'swe_output_path': os.getenv('swe_output_path'),
        'static_data_path': os.getenv('static_data_path'),
        'RF_model_path': os.getenv('RF_model_path'),
        'landslide_output_path': os.getenv('landslide_output_path'),
        'dem_path': os.getenv('dem_path'),
        'start_date': os.getenv('start_date'),
        'end_date': os.getenv('end_date'),
    }
    
    # Validate required variables
    for key, value in config.items():
        if not value:
            raise ValueError(f"Missing environment variable: {key}")
    
    return config


def reproject_to_match(src_path, reference_profile, reference_crs='EPSG:4326'):
    """Reproject raster to match reference profile"""
    with rasterio.open(src_path) as src:
        nodata_value = src.nodata
        src_array = src.read(1).astype(np.float32)
        
        if nodata_value is not None:
            src_array[src_array == nodata_value] = np.nan
        
        src_crs = src.crs or reference_crs
        dst_crs = reference_profile.get('crs') or reference_crs
        
        dst_array = np.empty((reference_profile['height'], reference_profile['width']), dtype=np.float32)
        
        reproject(
            source=src_array,
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=reference_profile['transform'],
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=np.nan,
            dst_nodata=np.nan
        )
        
        dst_array[dst_array == -3.4e+38] = np.nan
        if nodata_value is not None:
            dst_array[dst_array == nodata_value] = np.nan
    
    return dst_array


def reclassify(grid, mapping, nodata=np.nan):
    """Reclassify grid values based on mapping dictionary"""
    output = np.full_like(grid, fill_value=nodata, dtype=np.float32)
    for k, v in mapping.items():
        output[grid == k] = v
    return output


def run_rf_model(date, rf_model, sta_layers, file_maps, profile):
    """
    Execute Random Forest model for a given date
    
    Args:
        date: Date string in YYYYMMDD format
        rf_model: Trained RF model object
        sta_layers: Static layers dictionary
        file_maps: Dynamic file maps dictionary
        profile: Reference raster profile
    
    Returns:
        RF probability raster or None if data missing
    """
    dyn_layers = {}
    required_vars = ['BFExcess', 'infiltration', 'sm']
    
    for var in required_vars:
        path = file_maps[var].get(date)
        if not path:
            return None
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            if var == 'BFExcess':
                data *= 24
            dyn_layers[var] = data
    
    # Try to load rainmelt (optional, for RF features)
    rainmelt_data = None
    if 'rainmelt' in file_maps and file_maps['rainmelt']:
        path = file_maps['rainmelt'].get(date)
        if path:
            try:
                with rasterio.open(path) as src:
                    rainmelt_data = src.read(1).astype(np.float32)
            except Exception:
                pass
    
    # If no rainmelt available, use sm as proxy
    if rainmelt_data is None:
        rainmelt_data = dyn_layers['sm'].copy()
    
    ordered_vars = ["cumflow", "Ks", "slopes", "z", "BFExcess", "infiltration", "rainmelt", "sm"]
    
    stack = np.stack([
        sta_layers[v] if v in sta_layers else 
        dyn_layers.get(v, rainmelt_data if v == 'rainmelt' else dyn_layers['sm'])
        for v in ordered_vars
    ], axis=-1)
    
    flat = stack.reshape(-1, len(ordered_vars))
    mask = ~np.any(np.isnan(flat), axis=1)
    
    preds = np.full(flat.shape[0], np.nan)
    X_pred = pd.DataFrame(flat[mask], columns=ordered_vars)
    preds[mask] = rf_model.predict_proba(X_pred)[:, 1]
    raster = preds.reshape(stack.shape[:2])
    
    final_mask = dyn_layers['sm'] >= 0
    raster[~final_mask] = np.nan
    
    return raster


def stability_model(x, qa, qe, sm, swe, 
                    slope, h, porosity, Ks, tanphi_mean, tanphi_sd, C_soil, C_soil_sd, C_lulc_sd, 
                    unc_unstable, use_swe=False):
    """
    Calculate probability of failure using physical (infinite slope) model
    
    Args:
        x: Physical parameter vector
        qa, qe: Baseflow and infiltration
        sm: Soil moisture
        swe: Snow water equivalent
        slope: Slope angle
        h: Soil depth
        porosity: Soil porosity
        Ks: Saturated conductivity
        tanphi_mean, tanphi_sd: Mean and std of friction angle
        C_soil, C_soil_sd: Cohesion values
        C_lulc_sd: Land use cohesion uncertainty
        unc_unstable: Uncertainty mask
        use_swe: Whether to use snow water equivalent
    
    Returns:
        Probability of failure raster
    """
    qa = qa * x[4] * 24
    Kmmd = Ks * 1000 * 3600 * 24
    ha = B * qa / (Kmmd * h * np.sin(slope) * np.cos(slope))
    ha = np.minimum(ha, 1) * h
    
    sm = sm / 100.0
    nf = porosity * (1 - sm) + x[8]
    nf[nf < 0] = 1e-4
    
    qe = qe * x[6]
    water_table = ha + qe / 1000 / nf
    water_table = np.minimum(water_table, h)
    
    # Apply SWE effect if enabled
    if use_swe:
        swe_flag = (swe > 0).astype(int)
    else:
        swe_flag = np.zeros_like(swe, dtype=int)

    A = h * DS * G * np.sin(2 * slope) / 2

    D_fin = np.tan(slope) / (1 - (water_table / h) * (DW / DS))
    SF_mean = tanphi_mean / D_fin + C_soil / A
    SF_sd = np.sqrt((A ** 2) * (tanphi_sd ** 2) + (D_fin ** 2) * (C_soil_sd ** 2 + C_lulc_sd ** 2)) / (D_fin * A)

    PoF = norm.cdf(1, loc=SF_mean, scale=SF_sd)
    PoF[unc_unstable > 0.5] = 0
    PoF[swe_flag == 1] = 0

    return PoF


def save_raster(data, output_path, filename, profile):
    """Save raster to GeoTIFF"""
    ensure_directory(output_path)
    
    out_file = os.path.join(output_path, filename)
    
    output_profile = profile.copy()
    output_profile.update({
        'dtype': 'float32',
        'count': 1,
        'compress': 'lzw',
        'nodata': -9999,
        'tiled': True,
        'blockxsize': 512,
        'blockysize': 512,
        'num_threads': 4
    })
    
    with rasterio.open(out_file, 'w', **output_profile) as dst:
        dst.write(data, 1)
    
    return out_file


def main():
    """Main function for standalone landslide model"""
    print("\n" + "="*70)
    print("LANDSLIDE MODEL - STANDALONE VERSION")
    print("="*70)
    
    # Load configuration
    config = load_config()
    print(f"\n[CONFIG] Configuration loaded from .env")
    print(f"  - CREST output: {config['CREST_output_path']}")
    print(f"  - Static data: {config['static_data_path']}")
    print(f"  - RF model: {config['RF_model_path']}")
    print(f"  - Output: {config['landslide_output_path']}")
    print(f"  - Date range: {config['start_date']} to {config['end_date']}")
    
    # Parse dates
    start_date = datetime.strptime(config['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(config['end_date'], '%Y-%m-%d')
    
    # Create output directory
    ensure_directory(config['landslide_output_path'])
    print(f"\n[OUTPUT] Directory ready: {config['landslide_output_path']}")
    
    # Load DEM reference
    print(f"\n[DEM] Loading from: {config['dem_path']}")
    if not os.path.exists(config['dem_path']):
        raise FileNotFoundError(f"DEM not found: {config['dem_path']}")
    
    with rasterio.open(config['dem_path']) as src:
        ref_profile = src.profile.copy()
    print(f"✓ DEM loaded - Profile: {ref_profile['width']}x{ref_profile['height']}")
    
    # Load RF model
    print(f"\n[RF] Loading model from: {config['RF_model_path']}")
    if not os.path.exists(config['RF_model_path']):
        raise FileNotFoundError(f"RF model not found: {config['RF_model_path']}")
    
    rf_model = joblib.load(config['RF_model_path'])
    print(f"✓ RF model loaded")
    
    # Load static layers for RF
    print(f"\n[STATIC] Loading static layers from: {config['static_data_path']}")
    static_vars = ['cumflow', 'Ks', 'slopes', 'z']
    sta_layers = {}
    
    for var in static_vars:
        path = os.path.join(config['static_data_path'], f"{var}.tif")
        if not os.path.exists(path):
            print(f"  ✗ {var}.tif not found")
            continue
        arr = reproject_to_match(path, ref_profile)
        sta_layers[var] = arr
        print(f"  ✓ {var}.tif loaded")
    
    # Load soil and land use data
    print(f"\n[SOIL/LULC] Loading classification data...")
    soil_path = os.path.join(config['static_data_path'], 'soil_grid_30m.tif')
    hmtu_path = os.path.join(config['static_data_path'], 'hmtu_grid_30m.tif')

    if not os.path.exists(soil_path):
        raise FileNotFoundError(f"Soil file not found: {soil_path}")
    if not os.path.exists(hmtu_path):
        raise FileNotFoundError(f"Land use file not found: {hmtu_path}")
    
    soil_grid = reproject_to_match(soil_path, ref_profile)
    hmtu_grid = reproject_to_match(hmtu_path, ref_profile)
    print(f"  ✓ Soil grid loaded")
    print(f"  ✓ LULC grid loaded")
    
    # Load soil and HMTU parameters from CSV
    soil_csv_path = os.path.join(config['static_data_path'], 'soil.csv')
    hmtu_csv_path = os.path.join(config['static_data_path'], 'hmtu.csv')
    
    if not os.path.exists(soil_csv_path):
        raise FileNotFoundError(f"Soil CSV not found: {soil_csv_path}")
    if not os.path.exists(hmtu_csv_path):
        raise FileNotFoundError(f"HMTU CSV not found: {hmtu_csv_path}")
    
    soil = pd.read_csv(soil_csv_path).drop(index=0).reset_index(drop=True)
    hmtu = pd.read_csv(hmtu_csv_path).drop(index=0).reset_index(drop=True)
    soil['index'] = soil['index'].astype(int)
    hmtu['index'] = hmtu['index'].astype(int)
    print(f"  ✓ Soil parameters loaded")
    print(f"  ✓ HMTU parameters loaded")
    
    # Calibration parameters
    x = [0.824, 0.955, -4.968, -4.872, 1.000, 0.000, 1.000, 0.894, 0.000]
    print(f"\n[PARAMS] Calibration parameters: {x}")
    
    # Create parameter grids
    print(f"\n[GRIDS] Creating parameter grids...")
    soil_dicts = {
        'h': dict(zip(soil['index'], soil['h'].astype(float))),
        'Ks': dict(zip(soil['index'], soil['Ks'].astype(float))),
        'Cmin': dict(zip(soil['index'], soil['Cmin'].astype(float))),
        'Cmax': dict(zip(soil['index'], soil['Cmax'].astype(float))),
        'phimin': dict(zip(soil['index'], soil['phimin'].astype(float))),
        'phimax': dict(zip(soil['index'], soil['phimax'].astype(float))),
        'porosity': dict(zip(soil['index'], soil['porosity'].astype(float)))
    }
    hmtu_dicts = {
        'Cr_min': dict(zip(hmtu['index'], hmtu['Cr_min'].astype(float))),
        'Cr_max': dict(zip(hmtu['index'], hmtu['Cr_max'].astype(float)))
    }
    
    soil_depth = reclassify(soil_grid, {k: v + x[5] for k, v in soil_dicts['h'].items()})
    Ks = reclassify(soil_grid, {k: v * x[7] for k, v in soil_dicts['Ks'].items()})
    porosity = reclassify(soil_grid, soil_dicts['porosity'])
    C_soil_min = reclassify(soil_grid, {k: v * 1000 for k, v in soil_dicts['Cmin'].items()})
    C_soil_max = reclassify(soil_grid, {k: v * x[0] * 1000 for k, v in soil_dicts['Cmax'].items()})
    C_soil_mean = (C_soil_min + C_soil_max) / 2
    C_soil_sd = (C_soil_max - C_soil_min) / 4
    phi_min = np.radians(reclassify(soil_grid, {k: v + x[2] for k, v in soil_dicts['phimin'].items()}))
    phi_max = np.radians(reclassify(soil_grid, {k: v + x[3] for k, v in soil_dicts['phimax'].items()}))
    tanphi_mean = (np.tan(phi_min) + np.tan(phi_max)) / 2
    tanphi_sd = (np.tan(phi_max) - np.tan(phi_min)) / 4
    Cr_min = reclassify(hmtu_grid, {k: v * 1000 for k, v in hmtu_dicts['Cr_min'].items()})
    Cr_max = reclassify(hmtu_grid, {k: v * x[1] * 1000 for k, v in hmtu_dicts['Cr_max'].items()})
    Cr_lulc_mean = (Cr_min + Cr_max) / 2
    C_lulc_sd = (Cr_max - Cr_min) / 4
    
    # Load other static grids
    slope_path = os.path.join(config['static_data_path'], 'slopes.tif')
    unc_unstable_path = os.path.join(config['static_data_path'], 'unc_unstable_M.tif')
    
    slope_array = reproject_to_match(slope_path, ref_profile)
    unc_unstable = reproject_to_match(unc_unstable_path, ref_profile)
    print(f"  ✓ All parameter grids created")
    
    # Build file maps for dynamic inputs from CREST
    print(f"\n[CREST] Loading CREST output files...")
    crest_output = config['CREST_output_path']
    rainmelt_output = config['rainmelt_output_path']
    swe_output = config['swe_output_path']
    
    def build_file_map(var, folder):
        """Build dictionary mapping dates to file paths"""
        file_map = {}
        for f in glob(os.path.join(folder, f"{var}*.tif")):
            name = os.path.basename(f)
            parts = name.split(".")
            if len(parts) >= 2:
                date = parts[1].split("_")[0]
                file_map[date] = f
        return file_map
    
    # Get date bounds for filtering
    start_date_str = config['start_date'].replace('-', '')
    end_date_str = config['end_date'].replace('-', '')
    
    # Map all dynamic input variables
    file_maps = {
        'BFExcess': build_file_map('BFExcess', crest_output),
        'infiltration': build_file_map('infiltration', crest_output),
        'sm': build_file_map('sm', crest_output),
        'rainmelt': build_file_map('rainmelt', rainmelt_output),
        'swe': build_file_map('swe', swe_output)
    }
    
    # Filter all file maps by date range [start_date, end_date]
    for var in file_maps:
        file_maps[var] = {
            date: path for date, path in file_maps[var].items()
            if start_date_str <= date <= end_date_str
        }
    
    # Get common dates within range (only require BFExcess, infiltration, and sm)
    required_vars = ['BFExcess', 'infiltration', 'sm']
    common_dates = sorted(set.intersection(*(set(m.keys()) for m in [file_maps[v] for v in required_vars] if m)))
    
    print(f"  ✓ Date range filter applied: {start_date_str} to {end_date_str}")
    
    print(f"  ✓ Found {len(common_dates)} valid dates in range")
    
    if not common_dates:
        raise RuntimeError("No valid dates found in CREST output within specified range")
    
    # Process each date
    print(f"\n[PROCESSING] Exporting landslide analysis for {len(common_dates)} dates...")
    print(f"-" * 70)
    
    count_processed = 0
    
    for idx, date in enumerate(common_dates, 1):
        date_formatted = f"{date[0:4]}-{date[4:6]}-{date[6:8]}"
        print(f"[{idx}/{len(common_dates)}] {date_formatted}...", end=" ")
        
        try:
            # Load dynamic data from CREST
            qa = reproject_to_match(file_maps['BFExcess'][date], ref_profile)
            qe = reproject_to_match(file_maps['infiltration'][date], ref_profile)
            sm = reproject_to_match(file_maps['sm'][date], ref_profile)
            
            # Load SWE from SNOW17 if available
            swe = np.zeros_like(sm)
            if 'swe' in file_maps and file_maps['swe'] and date in file_maps['swe']:
                try:
                    swe = reproject_to_match(file_maps['swe'][date], ref_profile)
                except Exception:
                    pass  # Use zeros if SWE unavailable
            
            valid_mask = ~np.isnan(sm)
            
            # Run RF model
            rf_raster = run_rf_model(date, rf_model, sta_layers, file_maps, ref_profile)
            
            # Run Physical model with SWE
            phys_raster = stability_model(
                x, qa, qe, sm, swe,
                slope_array, soil_depth, porosity, Ks,
                tanphi_mean, tanphi_sd, C_soil_mean + Cr_lulc_mean,
                C_soil_sd, C_lulc_sd, unc_unstable,
                use_swe=True  # Enable SWE effect
            )
            
            # Compute weighted average: (2*RF + 1*Physical) / 3
            mean_raster = np.full_like(phys_raster, -9999, dtype=np.float32)
            
            both_valid = valid_mask & ~np.isnan(rf_raster) & ~np.isnan(phys_raster)
            mean_raster[both_valid] = (2 * rf_raster[both_valid] + phys_raster[both_valid]) / 3.0
            
            only_phys_valid = valid_mask & np.isnan(rf_raster) & ~np.isnan(phys_raster)
            mean_raster[only_phys_valid] = phys_raster[only_phys_valid]
            
            only_rf_valid = valid_mask & ~np.isnan(rf_raster) & np.isnan(phys_raster)
            mean_raster[only_rf_valid] = rf_raster[only_rf_valid]
            
            # Save all three rasters
            save_raster(rf_raster, config['landslide_output_path'], f"PoF_RF_{date}.tif", ref_profile)
            save_raster(phys_raster, config['landslide_output_path'], f"PoF_InfiniteSlope_{date}.tif", ref_profile)
            save_raster(mean_raster, config['landslide_output_path'], f"PoF_Ensemble_{date}.tif", ref_profile)
            
            count_processed += 1
            print(f"✓")
        
        except Exception as e:
            print(f"✗ {e}")
            continue
    
    print(f"\n" + "="*70)
    print(f"✓ LANDSLIDE MODEL COMPLETED")
    print(f"  - Dates processed: {count_processed}/{len(common_dates)}")
    print(f"  - Output directory: {config['landslide_output_path']}")
    print(f"  - Rasters generated: {count_processed * 3}")
    print("="*70 + "\n")
    
    return config['landslide_output_path']


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
