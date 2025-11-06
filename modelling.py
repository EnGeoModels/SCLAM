import os
import subprocess
import sys
from datetime import datetime
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.snow17 import main as snow17_main
import tempfile


# Load .env configuration
def load_config():
    """Load all configuration from .env file"""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    load_dotenv(env_path)
    
    config = {
        'CREST_exe_path': os.getenv('CREST_exe_path'),
        'CREST_output_path': os.getenv('CREST_output_path'),
        'start_date': os.getenv('start_date'),
        'warm_up_date': os.getenv('warm_up_date'),
        'end_date': os.getenv('end_date'),
    }
    
    # Validate required variables
    for key, value in config.items():
        if not value:
            raise ValueError(f"Missing environment variable: {key}")
    
    return config


def format_crest_date(date_str):
    """
    Convert date string from YYYY-MM-DD format to YYYYMMDD0000 format (CREST format)
    
    Args:
        date_str: Date in format 'YYYY-MM-DD'
    
    Returns:
        Date in format 'YYYYMMDD0000'
    """
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        return dt.strftime('%Y%m%d0000')
    except Exception as e:
        raise ValueError(f"Error parsing date {date_str}: {e}")


def modify_control_file(control_path, start_date, warm_up_date, end_date, logger=None):
    """
    Modify CREST control.txt file with new time parameters
    
    Args:
        control_path: Path to control.txt
        start_date: Start date in YYYY-MM-DD format
        warm_up_date: Warm-up end date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        logger: Optional logger
    """
    if not os.path.exists(control_path):
        raise FileNotFoundError(f"Control file not found: {control_path}")
    
    # Convert dates to CREST format
    time_begin = format_crest_date(start_date)
    time_warmend = format_crest_date(warm_up_date)
    time_end = format_crest_date(end_date)
    
    if logger:
        logger.info(f"[CREST] Modifying control.txt:")
        logger.info(f"  TIME_BEGIN={time_begin}")
        logger.info(f"  TIME_WARMEND={time_warmend}")
        logger.info(f"  TIME_END={time_end}")
    else:
        print(f"[CREST] Modifying control.txt:")
        print(f"  TIME_BEGIN={time_begin}")
        print(f"  TIME_WARMEND={time_warmend}")
        print(f"  TIME_END={time_end}")
    
    # Read control file
    with open(control_path, 'r') as f:
        lines = f.readlines()
    
    # Modify time parameters
    modified_lines = []
    for line in lines:
        if line.startswith('TIME_BEGIN='):
            modified_lines.append(f'TIME_BEGIN={time_begin}\n')
        elif line.startswith('TIME_WARMEND='):
            modified_lines.append(f'TIME_WARMEND={time_warmend}\n')
        elif line.startswith('TIME_END='):
            modified_lines.append(f'TIME_END={time_end}\n')
        else:
            modified_lines.append(line)
    
    # Write modified control file
    with open(control_path, 'w') as f:
        f.writelines(modified_lines)
    
    if logger:
        logger.info(f"[CREST] control.txt updated successfully")
    else:
        print(f"[CREST] control.txt updated successfully")


def run_snow17(logger=None):
    """Execute SNOW17 model"""
    if logger:
        logger.info("[SNOW17] Starting SNOW17 model...")
    else:
        print("[SNOW17] Starting SNOW17 model...")
    
    try:
        snow17_main()
        if logger:
            logger.info("[SNOW17] SNOW17 model finished successfully")
        else:
            print("[SNOW17] SNOW17 model finished successfully")
    except Exception as e:
        error_msg = f"[SNOW17] SNOW17 model failed: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        raise RuntimeError(error_msg)


def run_crest(control_path, logger=None):
    """Execute CREST model"""
    if not os.path.exists(control_path):
        raise FileNotFoundError(f"CREST control file not found: {control_path}")
    
    # Get CREST executable path
    config = load_config()
    crest_exe = config['CREST_exe_path']
    crest_dir = os.path.dirname(os.path.abspath(crest_exe))
    
    if logger:
        logger.info(f"[CREST] Starting CREST model from: {crest_dir}")
    else:
        print(f"[CREST] Starting CREST model from: {crest_dir}")
    
    try:
        result = subprocess.run(
            [crest_exe],
            shell=True,
            cwd=crest_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            error_msg = f"[CREST] CREST execution failed with code {result.returncode}"
            if result.stderr:
                error_msg += f"\nStderr: {result.stderr}"
            if logger:
                logger.error(error_msg)
            else:
                print(error_msg)
            raise RuntimeError(error_msg)
        
        if logger:
            logger.info("[CREST] CREST model finished successfully")
            if result.stdout:
                logger.info(f"[CREST] Output:\n{result.stdout}")
        else:
            print("[CREST] CREST model finished successfully")
            if result.stdout:
                print(f"[CREST] Output:\n{result.stdout}")
    
    except Exception as e:
        error_msg = f"[CREST] Error running CREST: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        raise RuntimeError(error_msg)

def run_landslide(crest_dir, logger=None, run_physical_model=False):
    """
    Executes landslide models in a modular way
    
    Args:
        crest_dir: CREST base directory
        logger: Logger to record operations
        run_physical_model: If True, also runs physical model (default False)
        
    TODO: Implement landslide model integration
    """
    if logger:
        logger.info("[LANDSLIDE] Landslide models are not yet implemented")
    else:
        print("[LANDSLIDE] Landslide models are not yet implemented")


def run_all_models(crest_dir=None, logger=None):
    """
    Execute all models in sequence: SNOW17 → CREST → Landslide
    
    Args:
        crest_dir: CREST base directory (optional, will use CREST_output_path from .env)
        logger: Optional logger
    """
    print("\n" + "="*70)
    print("MODELLING PIPELINE - ORCHESTRATING HYDRO & LANDSLIDE MODELS")
    print("="*70)
    
    # Load configuration from .env
    config = load_config()
    
    if crest_dir is None:
        crest_dir = os.path.dirname(os.path.abspath(config['CREST_output_path']))
    
    # Create output directories
    output_dir = os.path.join(crest_dir, "output")
    states_dir = os.path.join(crest_dir, "states")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(states_dir, exist_ok=True)
    
    if logger:
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"States directory: {states_dir}")
    else:
        print(f"Output directory: {output_dir}")
        print(f"States directory: {states_dir}")
    
    try:
        # Step 1: Run SNOW17
        print("\n" + "-"*70)
        print("STEP 1: RUNNING SNOW17")
        print("-"*70)
        run_snow17(logger)
        
        # Step 2: Modify CREST control file with dates from .env
        print("\n" + "-"*70)
        print("STEP 2: PREPARING CREST")
        print("-"*70)
        control_path = os.path.join(crest_dir, "control.txt")
        modify_control_file(
            control_path,
            config['start_date'],
            config['warm_up_date'],
            config['end_date'],
            logger
        )
        
        # Step 3: Run CREST
        print("\n" + "-"*70)
        print("STEP 3: RUNNING CREST")
        print("-"*70)
        run_crest(control_path, logger)
        
        # Step 4: Fix dates in CREST output files (optional)
        # print("\n" + "-"*70)
        # print("STEP 4: FIXING CREST OUTPUT DATES")
        # print("-"*70)
        # if logger:
        #     logger.info("Fixing dates in CREST output files...")
        # Utilities.fix_crest_date_formats(output_dir, logger)
        # if logger:
        #     logger.info("Date correction completed.")
        
        # Step 5: Run Landslide models (TODO)
        # print("\n" + "-"*70)
        # print("STEP 5: RUNNING LANDSLIDE MODELS")
        # print("-"*70)
        # run_landslide(crest_dir, logger, run_physical_model=False)
        
        print("\n" + "="*70)
        print("✓ MODELLING PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        
        return states_dir
    
    except Exception as e:
        print("\n" + "="*70)
        print(f"✗ MODELLING PIPELINE FAILED: {e}")
        print("="*70)
        if logger:
            logger.error(f"Pipeline failed: {e}")
        raise RuntimeError(f"Modelling pipeline failed: {e}")


if __name__ == '__main__':
    """
    Main entry point for the modelling pipeline.
    Can be executed standalone or imported in other scripts.
    
    Usage:
        python modelling.py
    """
    try:
        crest_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'CREST'))
        run_all_models(crest_base_dir, logger=None)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)