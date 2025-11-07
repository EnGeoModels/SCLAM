#!/usr/bin/env python3
"""
SCLAM - Main Pipeline
Integrates: Preprocessing -> SNOW17 -> CREST -> Landslide models
All configuration from .env file
"""

import os
import sys
import subprocess
from dotenv import load_dotenv
from datetime import datetime


def run_module(module_path, description):
    """Run a Python module and handle errors"""
    print(f"\n{'='*70}")
    print(f"RUNNING: {description}")
    print(f"Module: {module_path}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run([sys.executable, module_path], 
                              cwd=os.getcwd(),
                              capture_output=False,
                              text=True)
        
        if result.returncode == 0:
            print(f"\n✓ {description} - COMPLETED SUCCESSFULLY")
            return True
        else:
            print(f"\n✗ {description} - FAILED (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"\n✗ {description} - ERROR: {e}")
        return False


def clean_intermediate_files():
    """Clean intermediate files if requested"""
    clean_flag = os.getenv('clean_files', 'FALSE').upper()
    
    if clean_flag == 'TRUE':
        print(f"\n{'='*70}")
        print(f"CLEANING INTERMEDIATE FILES")
        print(f"{'='*70}")
        
        intermediate_dirs = [
            'CREST/pet',
            'SNOW17/rain', 
            'SNOW17/tavg',
            'CREST/rainmelt',
            'SNOW17/swe',
            'CREST/output'
        ]
        
        total_cleaned = 0
        for dir_path in intermediate_dirs:
            if os.path.exists(dir_path):
                import shutil
                try:
                    shutil.rmtree(dir_path)
                    print(f"  ✓ Cleaned: {dir_path}")
                    total_cleaned += 1
                except Exception as e:
                    print(f"  ✗ Failed to clean {dir_path}: {e}")
        
        print(f"\n✓ Cleaned {total_cleaned} intermediate directories")
    else:
        print(f"\n[INFO] Intermediate files preserved (clean_files={clean_flag})")


def main():
    """Main pipeline execution"""
    print("\n" + "="*70)
    print("SCLAM - COMPLETE HYDROLOGICAL-LANDSLIDE MODELING PIPELINE")
    print("="*70)
    
    if not os.path.exists('.env'):
        print("✗ ERROR: .env file not found!")
        sys.exit(1)
    
    load_dotenv()
    
    config = {
        'start_date': os.getenv('start_date'),
        'warm_up_date': os.getenv('warm_up_date'),
        'end_date': os.getenv('end_date'),
        'clean_files': os.getenv('clean_files', 'FALSE')
    }
    
    for key in ['start_date', 'warm_up_date', 'end_date']:
        if not config[key]:
            print(f"✗ ERROR: Missing {key} in .env file!")
            sys.exit(1)
    
    print(f"\n[CONFIG] Pipeline Configuration:")
    print(f"  - Simulation period: {config['start_date']} to {config['end_date']}")
    print(f"  - Warm-up date: {config['warm_up_date']}")
    print(f"  - Clean intermediate files: {config['clean_files']}")
    
    pipeline = [
        ('utils/preprocessing.py', 'DATA PREPROCESSING'),
        ('utils/snow17.py', 'SNOW17 MODEL'),
        ('utils/hydro_model.py', 'CREST HYDROLOGICAL MODEL'),
        ('utils/landslide.py', 'LANDSLIDE MODEL (RF + PHYSICAL + ENSEMBLE)')
    ]
    
    success_count = 0
    start_time = datetime.now()
    
    for module_path, description in pipeline:
        if not os.path.exists(module_path):
            print(f"\n✗ ERROR: Module not found: {module_path}")
            sys.exit(1)
        
        success = run_module(module_path, description)
        if success:
            success_count += 1
        else:
            print(f"\n✗ PIPELINE FAILED at: {description}")
            sys.exit(1)
    
    clean_intermediate_files()
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n" + "="*70)
    print(f"✓ SCLAM PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"="*70)
    print(f"  - Modules executed: {success_count}/{len(pipeline)}")
    print(f"  - Total duration: {duration}")
    print(f"  - Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  - End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n[OUTPUTS] Final results available in:")
    print(f"  - Landslide PoF rasters: {os.getenv('landslide_output_path', 'pof_landslide')}/")
    
    if config['clean_files'].upper() == 'TRUE':
        print(f"\n[CLEANUP] Intermediate files have been cleaned.")
    else:
        print(f"\n[PRESERVATION] Intermediate files preserved for inspection:")
        print(f"  - CREST outputs: CREST/output/")
        print(f"  - SNOW17 outputs: SNOW17/swe/, SNOW17/rain/")
        print(f"  - Rainmelt: CREST/rainmelt/")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n✗ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
