#!/usr/bin/env python3
"""
SCLAM - Main Pipeline
Integrates: SNOW17 -> CREST -> Landslide
All configuration from .env file
"""

import os
import sys
from dotenv import load_dotenv
from datetime import datetime
import models
from models.config import env_bool


def main():
    """Main pipeline execution"""
    print("\n" + "="*60)
    print("SCLAM - HYDROLOGICAL-LANDSLIDE MODELLING SYSTEM")
    print("="*60)
    
    if not os.path.exists('.env'):
        print("[ERROR] .env file not found!")
        sys.exit(1)
    
    load_dotenv()

    config = {
        'start_date': os.getenv('start_date'),
        'warm_up_date': os.getenv('warm_up_date'),
        'end_date': os.getenv('end_date'),
        'time_state': os.getenv('time_state'),
        'rain_output_path': os.getenv('rain_output_path'),
        'rain_path': os.getenv('rain_path'),
        'tavg_path': os.getenv('tavg_path'),
        'pet_path': os.getenv('pet_path') or os.getenv('pet_output_path'),
        'swe_output_path': os.getenv('swe_output_path'),
        'rainmelt_output_path': os.getenv('rainmelt_output_path'),
        'CREST_output_path': os.getenv('CREST_output_path'),
        'landslide_output_path': os.getenv('landslide_output_path'),
        'use_snow17': env_bool('use_snow17', True),
        'use_random_forest': env_bool('use_random_forest', True),
    }

    for key in ['start_date', 'warm_up_date', 'end_date']:
        if not config[key]:
            print(f"[ERROR] Missing {key} in .env file!")
            sys.exit(1)

    print(f"\nConfiguration: {config['start_date']} -> {config['end_date']}")
    print(f"Warm-up date: {config['warm_up_date']}\n")
    print(f"State date in hydrological model: {config['time_state']}\n")
    print(f"SNOW17: {'enabled' if config['use_snow17'] else 'disabled'}")
    print(f"Random Forest: {'enabled' if config['use_random_forest'] else 'disabled'}\n")

    start_time = datetime.now()

    # Run pipeline using models package entry points
    try:
        if config['use_snow17']:
            print("\n[RUNNING] SNOW17")
            models.run_snow17()
        else:
            print("\n[SKIPPING] SNOW17 disabled; CREST will use precipitation directly")
        print("\n[RUNNING] CREST")
        models.run_crest_model()
        print("\n[RUNNING] Landslide")
        models.run_landslide()
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        sys.exit(1)

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n" + "="*70)
    print(f"COMPLETED in {duration}")
    print(f"Outputs: {config['landslide_output_path']}/")
    print("="*70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n[ERROR] Pipeline interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
