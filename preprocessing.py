"""
Preprocessing Module - Standalone version
Reads configuration from .env and performs meteorological data interpolation.
Can be imported and used in main.py
"""

import pandas as pd
import os
import numpy as np
import rasterio
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression
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


class PreprocessingPipeline:
    def process_etp(self, start_date=None, end_date=None):
        """Process evapotranspiration (ETP) using Hargreaves-Samani method."""
        print("\n" + "="*60)
        print("Processing Evapotranspiration (ETP) Data")
        print("="*60)

        if not os.path.exists(self.meteo_data_path):
            raise FileNotFoundError(f"Meteorological data not found: {self.meteo_data_path}")

        if not os.path.exists(self.coordinates_path):
            raise FileNotFoundError(f"Coordinates file not found: {self.coordinates_path}")

        if self.dem_data is None:
            self.load_dem_info()

        # Load meteorological data
        df = pd.read_csv(self.meteo_data_path)
        df['date'] = pd.to_datetime(df['date'])

        # Load station coordinates
        print(f"Loading station coordinates from: {self.coordinates_path}")
        try:
            coords_df = pd.read_excel(self.coordinates_path, sheet_name='Temp')
        except:
            coords_df = pd.read_excel(self.coordinates_path, sheet_name=0)

        # Filter by required temperature columns
        required_cols = ['min_temperature', 'max_temperature', 'mean_temperature']
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found. Available columns: {df.columns.tolist()}")
                return self

        df_etp = df.dropna(subset=required_cols).copy()

        if start_date:
            df_etp = df_etp[df_etp['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df_etp = df_etp[df_etp['date'] <= pd.to_datetime(end_date)]

        print(f"Processing {len(df_etp)} ETP records")
        print(f"Date range: {df_etp['date'].min()} to {df_etp['date'].max()}")

        # Process each unique date
        unique_dates = df_etp['date'].unique()
        print(f"Unique dates: {len(unique_dates)} days\n")

        for date_idx, date in enumerate(unique_dates):
            date_obj = pd.to_datetime(date)
            date_str = date_obj.strftime('%Y%m%d')
            print(f"[{date_idx+1}/{len(unique_dates)}] Processing: {date_str}", end=" ")

            # Get data for this date
            df_day = df_etp[df_etp['date'] == date_obj].copy()
            df_day = df_day.merge(coords_df, left_on='station', right_on='station', how='inner')

            if df_day.empty:
                print("⚠ No station data found")
                continue

            stations = df_day[['Longitude', 'Latitude']].values
            tmin_values = df_day['min_temperature'].values
            tmax_values = df_day['max_temperature'].values
            tmean_values = df_day['mean_temperature'].values

            # Interpolate each temperature field
            tmin_grid = self.mlr_interpolation(stations, tmin_values)
            tmax_grid = self.mlr_interpolation(stations, tmax_values)
            tmean_grid = self.mlr_interpolation(stations, tmean_values)

            # Use latitude grid from DEM (self.grid_y)
            etp_grid = self.hargreaves_samani(tmin_grid, tmax_grid, tmean_grid, self.grid_y, date_obj)

            # Save result
            output_file = self.save_raster(etp_grid, self.pet_output_path, f"pet.{date_str}")
            print(f"✓ ({len(stations)} stations)")

        print(f"\nEvapotranspiration (ETP) processing completed ({len(unique_dates)} days)")
        return self
    """
    Standalone preprocessing pipeline for meteorological data interpolation.
    Reads all configuration from .env file.
    """
    
    def __init__(self, env_path=".env"):
        """Initialize the pipeline by loading environment variables."""
        load_dotenv(env_path)
        
        # Load paths from .env
        self.meteo_data_path = os.getenv('meteo_data_path')
        self.dem_path = os.getenv('dem_path')
        self.coordinates_path = os.getenv('coordinates_path')
        self.pet_output_path = os.getenv('pet_output_path')
        self.rain_output_path = os.getenv('rain_output_path')
        self.tavg_output_path = os.getenv('tavg_output_path')
        self.static_data_path = os.getenv('static_data_path')
        
        # Load dates from .env
        self.start_date = os.getenv('start_date')
        self.end_date = os.getenv('end_date')
        
        # Initialize data containers
        self.grid_x = None
        self.grid_y = None
        self.transform = None
        self.meta = None
        self.dem_data = None
        self.nodata_value = -9999
        self.bounds = None
        
        print("Preprocessing Pipeline initialized")
        print(f"DEM Path: {self.dem_path}")
        print(f"Coordinates Path: {self.coordinates_path}")
        print(f"Meteo Data Path: {self.meteo_data_path}")
        print(f"Date Range: {self.start_date} to {self.end_date}")
        print(f"Output paths - Rain: {self.rain_output_path}, PET: {self.pet_output_path}, Tavg: {self.tavg_output_path}")
    
    def load_dem_info(self):
        """Loads DEM raster data and extracts grid coordinates and metadata."""
        print(f"\nLoading DEM from: {self.dem_path}")
        
        if not os.path.exists(self.dem_path):
            raise FileNotFoundError(f"DEM file not found: {self.dem_path}")
        
        with rasterio.open(self.dem_path) as src:
            self.transform = src.transform
            width, height = src.width, src.height
            resolution = self.transform.a
            
            xmin, ymax = self.transform.c, self.transform.f  
            xmax = xmin + (width * resolution)
            ymin = ymax - (height * resolution)

            x_steps = np.linspace(xmin, xmax, width)
            y_steps = np.linspace(ymax, ymin, height)
            self.grid_x, self.grid_y = np.meshgrid(x_steps, y_steps)

            self.dem_data = src.read(1)
            self.nodata_value = src.nodata if src.nodata is not None else -9999
            self.meta = src.meta.copy()
            self.bounds = (xmin, xmax, ymin, ymax)
        
        print(f"DEM loaded successfully - Shape: {self.dem_data.shape}, NoData: {self.nodata_value}")
        return self


    def idw_interpolation(self, stations, values, power=2, num_neighbors=5):
        """Performs Inverse Distance Weighting (IDW) interpolation."""
        tree = cKDTree(stations)
        grid_points = np.c_[self.grid_x.ravel(), self.grid_y.ravel()]
        
        dists, idxs = tree.query(grid_points, k=min(num_neighbors, len(stations)))
        dists = np.where(dists == 0, 1e-6, dists)

        weights = 1 / (dists ** power)
        weights /= weights.sum(axis=1)[:, np.newaxis]

        interpolated_values = np.sum(weights * values[idxs], axis=1)
        return interpolated_values.reshape(self.grid_x.shape)

    def extract_dem_values(self, stations):
        """Extracts DEM values for given station coordinates."""
        extracted_values = []
        for lon, lat in stations:
            row, col = rasterio.transform.rowcol(self.transform, lon, lat)
            if 0 <= row < self.dem_data.shape[0] and 0 <= col < self.dem_data.shape[1]:
                extracted_values.append(self.dem_data[row, col])
            else:
                extracted_values.append(self.nodata_value)
        return extracted_values
    
    def mlr_interpolation(self, stations, values):
        """Performs Multiple Linear Regression (MLR) interpolation using DEM as covariate."""
        station_dem = np.array(self.extract_dem_values(stations))
        valid_mask = (station_dem != self.nodata_value) & (station_dem >= 0) & (~np.isnan(values))

        station_dem_valid = station_dem[valid_mask]
        values_valid = values[valid_mask]

        if len(station_dem_valid) < 2:
            print("Warning: Not enough valid stations for MLR, using mean value")
            return np.full(self.grid_x.shape, np.nanmean(values_valid) if len(values_valid) > 0 else self.nodata_value)

        X_train = station_dem_valid.reshape(-1, 1)
        y_train = values_valid

        mlr_model = LinearRegression().fit(X_train, y_train)

        grid_dem = self.dem_data.ravel().reshape(-1, 1)
        grid_valid_mask = (grid_dem[:, 0] != self.nodata_value) & (grid_dem[:, 0] >= 0)

        predicted_values = np.full(self.grid_x.shape, self.nodata_value, dtype=np.float32)
        predicted_values.ravel()[grid_valid_mask] = mlr_model.predict(grid_dem[grid_valid_mask])

        return predicted_values

    def hargreaves_samani(self, tmin, tmax, tmean, lat_deg, date):
        """Computes ETP using the Hargreaves-Samani method."""
        jd = date.timetuple().tm_yday
        phi = np.radians(lat_deg)
        
        es = 1 + 0.033 * np.cos(2 * np.pi / 365 * jd)
        solar_declination = 0.409 * np.sin((2 * np.pi / 365) * jd - 1.39)
        ws = np.arccos(-np.tan(phi) * np.tan(solar_declination))
        
        Ra = (24 * 60 / np.pi) * 0.0820 * es * (
            ws * np.sin(phi) * np.sin(solar_declination) +
            np.cos(phi) * np.cos(solar_declination) * np.sin(ws)
        ) / 2.45

        delta_temp = tmax - tmin
        delta_temp = np.where(delta_temp < 0, 0, delta_temp)
        
        et0 = 0.0023 * Ra * (tmean + 17.8) * np.sqrt(delta_temp)

        mask = (tmin == self.nodata_value) | (tmax == self.nodata_value) | (tmean == self.nodata_value)
        et0[mask] = self.nodata_value

        return et0


    def save_raster(self, data, output_path, date_str):
        """Saves interpolated data as GeoTIFF."""
        ensure_directory(output_path)
        
        output_file = os.path.join(output_path, f"{date_str}.tif")
        
        mask = (self.dem_data == self.nodata_value) | (self.dem_data < 0)
        data[mask] = self.nodata_value
        
        meta = self.meta.copy()
        meta.update({
            "count": 1,
            "dtype": "float32",
            "transform": self.transform,
            "driver": "GTiff",
            "nodata": self.nodata_value
        })
        
        with rasterio.open(output_file, "w", **meta) as dst:
            dst.write(data.astype(np.float32), 1)
        
        return output_file
    
    def process_precipitation(self, start_date=None, end_date=None):
        """Process precipitation data using IDW interpolation."""
        print("\n" + "="*60)
        print("Processing Precipitation Data")
        print("="*60)
        
        if not os.path.exists(self.meteo_data_path):
            raise FileNotFoundError(f"Meteorological data not found: {self.meteo_data_path}")
        
        if not os.path.exists(self.coordinates_path):
            raise FileNotFoundError(f"Coordinates file not found: {self.coordinates_path}")
        
        if self.dem_data is None:
            self.load_dem_info()
        
        # Load meteorological data
        df = pd.read_csv(self.meteo_data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Load station coordinates
        print(f"Loading station coordinates from: {self.coordinates_path}")
        try:
            coords_df = pd.read_excel(self.coordinates_path, sheet_name='Prec')
        except:
            # If 'Prec' sheet doesn't exist, try first sheet
            coords_df = pd.read_excel(self.coordinates_path, sheet_name=0)
        
        print(f"Loaded {len(coords_df)} stations")
        print(f"Columns: {coords_df.columns.tolist()}")
        
        # Filter by precipitation data
        df_precip = df[df['precipitation'].notna()].copy()
        
        if start_date:
            df_precip = df_precip[df_precip['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df_precip = df_precip[df_precip['date'] <= pd.to_datetime(end_date)]
        
        print(f"Processing {len(df_precip)} precipitation records")
        print(f"Date range: {df_precip['date'].min()} to {df_precip['date'].max()}")
        
        # Process each unique date
        unique_dates = df_precip['date'].unique()
        print(f"Unique dates: {len(unique_dates)} days")
        
        for date_idx, date in enumerate(unique_dates):  # Process all dates
            date_obj = pd.to_datetime(date)
            date_str = date_obj.strftime('%Y%m%d')
            print(f"[{date_idx+1}/{len(unique_dates)}] Processing: {date_str}", end=" ")
            
            # Get data for this date
            df_day = df_precip[df_precip['date'] == date_obj].copy()
            
            # Merge with coordinates
            df_day = df_day.merge(coords_df, left_on='station', right_on='station', how='inner')
            
            if df_day.empty:
                print("⚠ No station data found")
                continue
            
            # Extract station coordinates and values
            stations = df_day[['Longitude', 'Latitude']].values
            values = df_day['precipitation'].values
            
            # Perform IDW interpolation
            interpolated_grid = self.idw_interpolation(stations, values, power=2, num_neighbors=5)
            
            # Save result
            output_file = self.save_raster(interpolated_grid, self.rain_output_path, f"precip.{date_str}")
            print(f"✓ ({len(stations)} stations)")
        
        print(f"\nPrecipitation processing completed ({len(unique_dates)} days)")
        return self
    
    def process_temperature(self, start_date=None, end_date=None, temp_type='mean'):
        """Process temperature data using MLR interpolation."""
        print("\n" + "="*60)
        print(f"Processing {temp_type.capitalize()} Temperature Data")
        print("="*60)
        
        if not os.path.exists(self.meteo_data_path):
            raise FileNotFoundError(f"Meteorological data not found: {self.meteo_data_path}")
        
        if not os.path.exists(self.coordinates_path):
            raise FileNotFoundError(f"Coordinates file not found: {self.coordinates_path}")
        
        if self.dem_data is None:
            self.load_dem_info()
        
        # Load meteorological data
        df = pd.read_csv(self.meteo_data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Load station coordinates
        print(f"Loading station coordinates from: {self.coordinates_path}")
        try:
            coords_df = pd.read_excel(self.coordinates_path, sheet_name='Temp')
        except:
            coords_df = pd.read_excel(self.coordinates_path, sheet_name=0)
        
        # Filter by temperature data
        temp_col = f'{temp_type}_temperature'
        if temp_col not in df.columns:
            print(f"Warning: Column '{temp_col}' not found. Available columns: {df.columns.tolist()}")
            return self
        
        df_temp = df[df[temp_col].notna()].copy()
        
        if start_date:
            df_temp = df_temp[df_temp['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df_temp = df_temp[df_temp['date'] <= pd.to_datetime(end_date)]
        
        print(f"Processing {len(df_temp)} temperature records")
        print(f"Date range: {df_temp['date'].min()} to {df_temp['date'].max()}")
        
        # Process each unique date
        unique_dates = df_temp['date'].unique()
        print(f"Unique dates: {len(unique_dates)} days\n")
        
        for date_idx, date in enumerate(unique_dates):
            date_obj = pd.to_datetime(date)
            date_str = date_obj.strftime('%Y%m%d')
            print(f"[{date_idx+1}/{len(unique_dates)}] Processing: {date_str}", end=" ")
            
            # Get data for this date
            df_day = df_temp[df_temp['date'] == date_obj].copy()
            
            # Merge with coordinates
            df_day = df_day.merge(coords_df, left_on='station', right_on='station', how='inner')
            
            if df_day.empty:
                print("⚠ No station data found")
                continue
            
            # Extract station coordinates and values
            stations = df_day[['Longitude', 'Latitude']].values
            values = df_day[temp_col].values
            
            # Perform MLR interpolation
            interpolated_grid = self.mlr_interpolation(stations, values)
            
            # Save result
            prefix = f"tavg" if temp_type == 'mean' else f"t{temp_type[0]}"
            output_file = self.save_raster(interpolated_grid, self.tavg_output_path, f"{prefix}.{date_str}")
            print(f"✓ ({len(stations)} stations)")
        
        print(f"\n{temp_type.capitalize()} temperature processing completed ({len(unique_dates)} days)")
        return self


def main():
    """Main execution function for standalone testing."""
    print("="*60)
    print("PREPROCESSING PIPELINE - STANDALONE MODE")
    print("="*60)
    
    try:
        # Initialize pipeline
        pipeline = PreprocessingPipeline()

        # Load DEM
        pipeline.load_dem_info()

        # Process meteorological data using dates from .env
        print("\n" + "="*60)
        print(f"Processing data from {pipeline.start_date} to {pipeline.end_date}")
        print("="*60)

        pipeline.process_precipitation(start_date=pipeline.start_date, end_date=pipeline.end_date)
        pipeline.process_temperature(start_date=pipeline.start_date, end_date=pipeline.end_date, temp_type='mean')
        pipeline.process_etp(start_date=pipeline.start_date, end_date=pipeline.end_date)

        print("\n" + "="*60)
        print("✓ Pipeline executed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(main())

