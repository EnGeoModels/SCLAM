"""
Hydrological Model - CREST Execution Module
Standalone script for running the CREST hydrological model
"""

import os
import re
import subprocess
import sys
import shutil
from datetime import datetime, timedelta
from dotenv import load_dotenv


def load_config():
    """Load all configuration from .env file"""
    # Get the path to the project root (parent of utils)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(project_root, '.env')
    load_dotenv(env_path)
    
    config = {
        'CREST_exe_path': os.getenv('CREST_exe_path'),
        'CREST_output_path': os.getenv('CREST_output_path'),
        'start_date': os.getenv('start_date'),
        'warm_up_date': os.getenv('warm_up_date'),
        'end_date': os.getenv('end_date'),
        'time_state': os.getenv('time_state'),
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


def format_crest_state_date(date_str):
    """
    Convert date string from YYYY-MM-DD format to YYYYMMDD format for TIME_STATE
    
    Args:
        date_str: Date in format 'YYYY-MM-DD'
    
    Returns:
        Date in format 'YYYYMMDD'
    """
    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        return dt.strftime('%Y%m%d')
    except Exception as e:
        raise ValueError(f"Error parsing date {date_str}: {e}")


def rename_output_file(output_dir, prefixes=None):
    """
    Renames all .tif files in output_dir that start with any of the given prefixes
    to the format <prefix>.YYYYMMDDHHMM.tif, removing any intermediate suffix.
    
    Args:
        output_dir: Directory containing CREST output files
        prefixes: List of prefixes to process (default: ['q'])
    
    Returns:
        Number of files renamed
    
    Example:
        Input:  q.20130620_0000_something.tif
        Output: q.201306200000.tif
    """
    if prefixes is None:
        prefixes = ['q']
    
    renamed_count = 0
    
    for fname in os.listdir(output_dir):
        for prefix in prefixes:
            if fname.startswith(prefix) and fname.endswith(".tif"):
                # Search for date and time with any separator and any suffix before .tif
                match = re.match(rf"{prefix}\.(\d{{8}})[_\.](\d{{4}}).*\.tif", fname)
                if match:
                    date_str = match.group(1)
                    hour_str = match.group(2)
                    new_name = f"{prefix}.{date_str}{hour_str}.tif"
                    old_path = os.path.join(output_dir, fname)
                    new_path = os.path.join(output_dir, new_name)
                    if old_path != new_path:
                        os.rename(old_path, new_path)
                        renamed_count += 1
                        print(f"  ✓ Renamed: {fname} → {new_name}")
                # If it doesn't have time, but only has date
                else:
                    match = re.match(rf"{prefix}\.(\d{{8}}).*\.tif", fname)
                    if match:
                        date_str = match.group(1)
                        new_name = f"{prefix}.{date_str}0000.tif"
                        old_path = os.path.join(output_dir, fname)
                        new_path = os.path.join(output_dir, new_name)
                        if old_path != new_path:
                            os.rename(old_path, new_path)
                            renamed_count += 1
                            print(f"  ✓ Renamed: {fname} → {new_name}")
            break
    
    return renamed_count


def fix_crest_date_formats(crest_output_dir, processing_date=None):
    """
    Fixes CREST file date formats for infiltration and BFExcess variables.
    
    LOGIC:
    1. Template files (YYYYMMDD_HHUU) are renamed with warm_up_date
    2. All regular files get +1 day added to their date
    3. Original warm_up_date file gets overwritten by template (now has +1 day)
    
    Uses a temporary subfolder to make corrections safely.
    
    Args:
        crest_output_dir: CREST output directory
        processing_date: Date being processed (not used, kept for compatibility)
    """
    
    if not os.path.exists(crest_output_dir):
        return
    
    # Load warm_up_date from .env to rename templates
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    load_dotenv(env_path)
    warm_up_date_str = os.getenv('warm_up_date')
    end_date_str = os.getenv('end_date')
    
    if not warm_up_date_str:
        return
    
    try:
        warm_up_date = datetime.strptime(warm_up_date_str, '%Y-%m-%d')
        warm_up_date_str_fmt = warm_up_date.strftime('%Y%m%d')
    except:
        return
    
    if end_date_str:
        try:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        except:
            end_date = None
    else:
        end_date = None
    
    # Only these variables need date correction (+1 day)
    variables_to_fix = ['infiltration', 'BFExcess']
    
    # Create temporary directory for corrections
    temp_dir = os.path.join(crest_output_dir, "temp_corrections")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        for variable in variables_to_fix:
            # 1. Search for all files of this variable in output
            all_files = [f for f in os.listdir(crest_output_dir) 
                        if f.startswith(f"{variable}.") and f.endswith('.tif')]
            
            if not all_files:
                continue
            
            # Separate templates from regular files
            template_files = [f for f in all_files if ('YYYYMMDD' in f or '_HHUU' in f)]
            regular_files = [f for f in all_files if not ('YYYYMMDD' in f or '_HHUU' in f)]
            regular_files.sort()
            
            archivos_procesados = []
            
            # Step 1: Process template files FIRST - rename with warm_up_date
            if template_files:
                for template_file in template_files:
                    try:
                        nuevo_nombre = f"{variable}.{warm_up_date_str_fmt}_0000.crestphys.tif"
                        src_path = os.path.join(crest_output_dir, template_file)
                        dst_path = os.path.join(temp_dir, nuevo_nombre)
                        
                        shutil.copy2(src_path, dst_path)
                        archivos_procesados.append(nuevo_nombre)
                    except Exception:
                        pass
            
            # Step 2: Process regular files - add +1 day to each
            for filename in regular_files:
                try:
                    # Extract date from name
                    match = re.search(r'(\d{8})_\d{4}', filename)
                    if not match:
                        continue
                    
                    fecha_str = match.group(1)
                    
                    # Add 1 day
                    fecha = datetime.strptime(fecha_str, '%Y%m%d')
                    nueva_fecha = fecha + timedelta(days=1)
                    nueva_fecha_str = nueva_fecha.strftime('%Y%m%d')
                    
                    # Validate that corrected date doesn't exceed end_date
                    if end_date:
                        if nueva_fecha > end_date:
                            continue
                    
                    # Build new name
                    nuevo_nombre = f"{variable}.{nueva_fecha_str}_0000.crestphys.tif"
                    
                    # Copy with new name directly to temporary
                    src_path = os.path.join(crest_output_dir, filename)
                    dst_path = os.path.join(temp_dir, nuevo_nombre)
                    
                    # Check for conflicts
                    if os.path.exists(dst_path):
                        continue
                    
                    shutil.copy2(src_path, dst_path)
                    archivos_procesados.append(nuevo_nombre)
                    
                except Exception:
                    pass
            
            # Step 3: Delete all original files of this variable from output
            for filename in all_files:
                file_path = os.path.join(crest_output_dir, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # Step 4: Move processed files from temporary to output
            for filename in archivos_procesados:
                src_path = os.path.join(temp_dir, filename)
                dst_path = os.path.join(crest_output_dir, filename)
                if os.path.exists(src_path):
                    shutil.move(src_path, dst_path)
    
    finally:
        # Clean temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def modify_control_file(control_path, start_date, warm_up_date, end_date, time_state):
    """
    Modify CREST control.txt file with new time parameters
    
    Args:
        control_path: Path to control.txt
        start_date: Start date in YYYY-MM-DD format
        warm_up_date: Warm-up end date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        time_state: State date in YYYY-MM-DD format
    """
    if not os.path.exists(control_path):
        raise FileNotFoundError(f"Control file not found: {control_path}")
    
    # Convert dates to CREST format
    time_begin = format_crest_date(start_date)
    time_warmend = format_crest_date(warm_up_date)
    time_end = format_crest_date(end_date)
    time_state_formatted = format_crest_state_date(time_state)
    
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
        elif line.startswith('TIME_STATE='):
            modified_lines.append(f'TIME_STATE={time_state_formatted}\n')
        else:
            modified_lines.append(line)
    
    # Write modified control file
    with open(control_path, 'w') as f:
        f.writelines(modified_lines)


def run_crest_model():
    """
    Execute CREST hydrological model
    
    Steps:
        1. Load configuration from .env
        2. Modify control.txt with dates
        3. Execute CREST binary
    
    Returns:
        Path to CREST output directory
    """
    # Load configuration
    config = load_config()
    
    # Get CREST paths
    crest_exe = config['CREST_exe_path']
    
    # Use absolute path
    if not os.path.isabs(crest_exe):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        crest_exe = os.path.join(project_root, crest_exe)
    
    if not os.path.exists(crest_exe):
        raise FileNotFoundError(f"CREST executable not found: {crest_exe}")
    
    if not os.access(crest_exe, os.X_OK):
        raise PermissionError(f"CREST executable not executable: {crest_exe}")
    
    crest_dir = os.path.dirname(crest_exe)
    control_path = os.path.join(crest_dir, "control.txt")
    
    # Create output directories
    output_dir = os.path.join(crest_dir, "output")
    states_dir = os.path.join(crest_dir, "states")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(states_dir, exist_ok=True)
    
    print(f"  CREST (executing model)...")
    
    # Step 1: Modify control file with dates from .env
    modify_control_file(
        control_path,
        config['start_date'],
        config['warm_up_date'],
        config['end_date'],
        config['time_state']
    )
    
    # Step 2: Execute CREST
    try:
        # Run CREST directly (Linux executable, not shell command)
        result = subprocess.run(
            [crest_exe],
            cwd=crest_dir,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            error_msg = f"CREST execution failed with code {result.returncode}"
            if result.stderr:
                error_msg += f"\nStderr: {result.stderr}"
            if result.stdout:
                error_msg += f"\nStdout: {result.stdout}"
            raise RuntimeError(error_msg)
    
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"CREST execution timed out (1 hour limit)")
    
    except Exception as e:
        raise RuntimeError(f"Error running CREST: {e}")
    
    # Step 3: Rename output files to standard format
    fix_crest_date_formats(output_dir, processing_date=None)
    
    return output_dir


def main():
    """Main entry point for standalone execution"""
    try:
        output_dir = run_crest_model()
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
