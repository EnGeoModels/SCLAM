# SCLAM - SNOW17-CREST Integrated Landslide Model

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

SCLAM (SNOW17-CREST Landslide Assessment Model) is an integrated hydrological-geological modeling system that combines the SNOW17 snowmelt model with the CREST distributed hydrological model for landslide prediction.

## 📋 Description

The SCLAM system integrates three main components:

1. **SNOW17**: Snowmelt model that simulates snow water balance using precipitation and temperature data
2. **CREST**: Distributed hydrological model that simulates subsurface flow and runoff generation
3. **Landslide**: Integrated Machine learning and infinite slope model for probability of failure prediction (ñandslide initiation areas)

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

**Main parameters:**

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

# Simulation period
start_date=2012-10-01
warm_up_date=2013-06-01
end_date=2013-06-30
time_state=2013-06-01

# Trained model
RF_model_path=RF_model.pkl
static_data_path=static
```

### Required Data Structure

```
SCLAM/
├── CREST/
│   ├── basic/
│   │   ├── DEM.asc          # Digital Elevation Model
│   │   ├── FDR.asc          # Flow Direction
│   │   └── FAC.asc          # Flow Accumulation
│   ├── ef5                  # CREST executable (Linux)
│   └── control.txt          # CREST configuration file
├── rain/                    # Precipitation files (GeoTIFF)
│   ├── rain.20120101.tif
│   ├── rain.20120102.tif
│   └── ...
├── tavg/                    # Temperature files (GeoTIFF)
│   ├── tavg.20121001.tif
│   ├── tavg.20121002.tif
│   └── ...
├── RF_model.pkl            # Trained Random Forest model
└── .env                     # Configuration
```

## Input Data

### Meteorological Data

Precipitation and temperature data must be in **GeoTIFF** format with:
- **Projection**: Appropriate extent and coordinate system for the watershed
- **Resolution**: Consistent with DEM (CREST/basic)
- **Units**: Precipitation (mm/day), Temperature (°C)
- **Naming format**: `rain.YYYYMMDD.tif` and `tavg.YYYYMMDD.tif`

Portential evapotranspiration is desirable, specially for long term runs. This demo provide it (`tavg.YYYYMMDD.tif` in CREST/pet) estimated based on Hargreaves-Samani equation (only requires temperature).


### Digital Elevation Model (DEM)

- **Format**: ESRI ASCII (.asc)
- **Requirements**: Must match flow direction (FDR) and flow accumulation (FAC) layers

### Static data
- **Format**: GeoTIFF (.tif). Used to run infinite slope model

### Random Forest Model

- **File**: `RF_model.pkl` (pre-trained)
- **Variables**: Based on hydrological and topographic data

## SCLAM Usage

```bash
python main.py
```

### Individual module execution

```bash
# SNOW17 only
python utils/snow17.py

# CREST only
python utils/hydro_model.py

# Landslide prediction only
python utils/landslide.py
```

## System Outputs

### SNOW17
- `SNOW17/swe/`: Snow Water Equivalent (SWE) daily
- `CREST/rainmelt/`: Rainmelt (rain-generated melt)

### CREST
- `CREST/output/`: Hydrological variables (runoff, soil moisture, etc.)
- `CREST/states/`: Model states for restart

### Landslide Model
- `pof_landslide/`: Landslide probability maps

## Advanced Configuration

### CREST Parameter Modification

Edit `CREST/control.txt` to adjust:
- Soil physical parameters
- Routing parameters
- Additional detailed information in: https://chrimerss.github.io/EF5/docs/#si

### Simulation Period Change

Modify in `.env`:
```env
start_date=2012-10-01      # Start date
warm_up_date=2013-06-01    # Warm-up period
end_date=2013-06-30        # End date
time_state=2013-06-01      # Initial state for restart
```

## Author

- Flavio Alexander Asurza. Contact: flavio.alexander.asurza@upc.edu