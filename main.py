#!/usr/bin/env python3
"""
SCLAM - Main Pipeline
Integrates: Preprocessing -> SNOW17 -> CREST -> Landslide
All configuration from .env file
"""

import os
import sys
import subprocess
import shutil
from dotenv import load_dotenv
from datetime import datetime


def clean_intermediate_files(config):
    """Remove intermediate output files, keeping only landslide results"""
    print("\n[CLEANUP] Removing intermediate files...")
    
    # Directories to clean
    to_clean = [
        config.get('rain_output_path', 'SNOW17/rain'),
        config.get('tavg_output_path', 'SNOW17/tavg'),
        config.get('pet_output_path', 'CREST/pet'),
        config.get('swe_output_path', 'SNOW17/swe'),
        config.get('rainmelt_output_path', 'CREST/rainmelt'),
        config.get('CREST_output_path', 'CREST/output'),
    ]
    
    cleaned_count = 0
    for directory in to_clean:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
                cleaned_count += 1
                print(f"  ✓ Cleaned: {directory}")
            except Exception as e:
                print(f"  ✗ Error cleaning {directory}: {e}")
    
    print(f"[CLEANUP] Removed {cleaned_count} intermediate directories")


def run_module(module_path, description):
    """Run a Python module and handle errors"""
    print(f"\n[RUNNING] {description}")
    
    try:
        result = subprocess.run([sys.executable, module_path], 
                              cwd=os.getcwd(),
                              capture_output=False,
                              text=True)
        
        if result.returncode == 0:
            return True
        else:
            print(f"[ERROR] {description} failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"[ERROR] {description}: {e}")
        return False


def main():
    """Main pipeline execution"""
    print("\n" + "="*60)
    print("SCLAM - HYDROLOGICAL-LANDSLIDE MODELING SYSTEM")
    print("="*60)
    
    if not os.path.exists('.env'):
        print("[ERROR] .env file not found!")
        sys.exit(1)
    
    load_dotenv()

    config = {
        'start_date': os.getenv('start_date'),
        'warm_up_date': os.getenv('warm_up_date'),
        'end_date': os.getenv('end_date'),
        'clean_files': os.getenv('clean_files', 'FALSE').upper(),
        'rain_output_path': os.getenv('rain_output_path'),
        'tavg_output_path': os.getenv('tavg_output_path'),
        'pet_output_path': os.getenv('pet_output_path'),
        'swe_output_path': os.getenv('swe_output_path'),
        'rainmelt_output_path': os.getenv('rainmelt_output_path'),
        'CREST_output_path': os.getenv('CREST_output_path'),
        'landslide_output_path': os.getenv('landslide_output_path'),
    }

    for key in ['start_date', 'warm_up_date', 'end_date']:
        if not config[key]:
            print(f"[ERROR] Missing {key} in .env file!")
            sys.exit(1)

    print(f"\nConfiguration: {config['start_date']} -> {config['end_date']}")
    print(f"Warm-up date: {config['warm_up_date']}")
    print(f"Clean files: {'ON' if config['clean_files'] == 'TRUE' else 'OFF'}\n")
    
    pipeline = [
        ('utils/preprocessing.py', 'Preprocessing'),
        ('utils/snow17.py', 'SNOW17'),
        ('utils/hydro_model.py', 'CREST'),
        ('utils/landslide.py', 'Landslide')
    ]
    
    start_time = datetime.now()
    
    for module_path, description in pipeline:
        if not os.path.exists(module_path):
            print(f"[ERROR] Module not found: {module_path}")
            sys.exit(1)
        
        success = run_module(module_path, description)
        if not success:
            print(f"[ERROR] Pipeline failed at {description}")
            sys.exit(1)
    
    # Clean intermediate files if enabled
    if config['clean_files'] == 'TRUE':
        clean_intermediate_files(config)
    
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
