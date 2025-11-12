# SCLAM: SNOW17-CREST-Integrated Landslide Model

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Description

The SCLAM model system integrates three main components:

1. **SNOW17**: Snowmelt model that simulates snow water balance using precipitation and temperature data
2. **CREST**: Distributed hydrological model that simulates subsurface flow and runoff generation
3. **Landslide**: Integrated Random Forest and infinite slope model for probability of failure prediction (landslide initiation areas)

## System Architecture

```
Meteorological Data (Precipitation + Temperature)
           │
           ▼
       ┌─────────────┐
       │   SNOW17    │ ← Simulates snowmelt and water balance
       │  (Snowmelt) │
       └─────────────┘
           │
           ▼
       ┌─────────────┐
       │    CREST    │ ← Distributed hydrological model
       │ (Hydrology) │
       └─────────────┘
           │
           ▼
       ┌─────────────┐
       │ Integrated  │ ← Inititation landslide areas prediction
       │ Landslide   │
       └─────────────┘
```

## System Requirements

### Hardware
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: 10GB+ free space
- **Processor**: Multi-core CPU (recommended 4+ cores)

### Software
- **Operating System**: Linux/macOS/Windows
- **Python**: 3.10

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/EnGeoModels/SCLAM.git
cd SCLAM
```

### 2. Create virtual environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Linux/macOS:
source sclam_env/bin/activate
# Windows:
sclam_env\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Configuration

### `.env` File

The system is fully configured through the `.env` file. Copy the example file and modify it:

**Main parameters (.env file example):**

```env
# Geospatial data paths
dem_path=CREST/basic/DEM.asc
rain_path=rain
tavg_path=tavg

# Output paths
rainmelt_output_path=CREST/rainmelt
swe_output_path=SNOW17/swe
CREST_exe_path=CREST/ef5
CREST_output_path=CREST/output
landslide_output_path=pof_landslide

# UTM projection for the study area (Valle de Arán, Spain is in UTM Zone 31N)
utm_projection=EPSG:32631

# Simulation period
start_date=2012-10-01
warm_up_date=2013-06-01
end_date=2013-06-30
time_state=2013-06-01

# Trained model
RF_model_path=RF_model.pkl
static_data_path=static
```
## Input Data

### Meteorological Data

Precipitation and temperature can be in **GeoTIFF** format with:
- **Projection**: Appropriate extent and coordinate system for the watershed
- **Resolution**: Consistent with DEM (CREST/basic)
- **Units**: Precipitation (mm/day), Temperature (°C)
- **Naming format**: `rain.YYYYMMDD.tif` and `tavg.YYYYMMDD.tif`

Portential evapotranspiration is desirable, specially for long term runs. We suggest to estimate it based on Hargreaves-Samani equation (only requires temperature).

### Digital Elevation Model (DEM), Flow Direction (FDR) and Flow Accumulation (FAC)

- **Format**: ESRI ASCII (.asc) or GeoTIFF (.tif)
- **Requirements**: Must match FDR and FAC layers

### Static data
- **Soil grids**: GeoTIFF (.tif). Used to run infinite slope and Random Forest model
- **Soil properties**: Comma-Separated Values (.tif). To indicate values of soil properties for each type of soil in the study area.


### Random Forest Model
- **File**: `RF_model.pkl` (pre-trained)

## SCLAM Usage

```bash
python main.py
```

### Individual module execution

```bash
# SNOW17 only
python models/snow17.py

# CREST only (silent mode)
python models/hydro_model.py

# CREST only (with verbose CREST output)
python models/hydro_model.py --verbose

# Landslide prediction only
python models/landslide.py
```
## Advanced Configuration

### Landslide Model Parameters

The pixel size (constant B) in the infinite slope model is **automatically extracted** from the DEM by reprojecting it to UTM coordinates. This ensures that B is always in meters and corresponds to the actual spatial resolution of your data.

**Configuration in `.env`:**
```env
# UTM projection for the study area
# Valle de Arán, Spain uses UTM Zone 31N (EPSG:32631)
utm_projection=EPSG:32631
```

The pixel size is automatically calculated when the landslide model runs:
- DEM is read in its original projection
- Reprojected to the specified UTM zone
- Pixel size extracted in meters
- Used as constant B in the infinite slope stability equation

**Common UTM Zones:**
- Spain (Iberian Peninsula): EPSG:32631 (Zone 31N)
- France (Alps): EPSG:32632 (Zone 32N)
- Italy: EPSG:32632-EPSG:32633 (Zones 32-33N)
- Pyrenees: EPSG:32631 (Zone 31N)

### CREST Parameter Modification

Edit `CREST/control.txt` to adjust:
- Soil physical parameters
- Routing parameters
- Additional detailed information in: https://chrimerss.github.io/EF5/docs/#si

### Simulation Period Change (example)

Modify in `.env`:
```env
time_state=2013-06-01      # Initial state for restart (used in CREST)
start_date=2012-10-01      # Start date
warm_up_date=2013-06-01    # Warm-up period (use it in CREST). Not estrictly needed when time_state is specified
end_date=2013-06-30        # End date
```

### Example data
- Example data provided in example_data.zip file in GeoTIFF format for reproducibility

## Author
- Alex Asurza. Contact: flavio.alexander.asurza@upc.edu