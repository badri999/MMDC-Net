import os
import shutil
from pathlib import Path
import argparse
import logging
from typing import List, Dict, Tuple

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('download_results.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def find_matching_files(base_path: Path, filename: str, split: str) -> List[Tuple[Path, str]]:
    """
    Find all files matching the given filename across all subdirectories
    
    Args:
        base_path: Base results directory path
        filename: Filename to search for (without extension)
        split: Dataset split (train/test/valid)
    
    Returns:
        List of tuples: (file_path, subfolder_name)
    """
    split_path = base_path / split
    
    if not split_path.exists():
        raise FileNotFoundError(f"Split directory not found: {split_path}")
    
    matching_files = []
    
    # Iterate through all subdirectories
    for subfolder in split_path.iterdir():
        if subfolder.is_dir():
            subfolder_name = subfolder.name
            
            # Look for files that match the filename (any extension)
            for file in subfolder.iterdir():
                if file.is_file() and file.stem == filename:
                    matching_files.append((file, subfolder_name))
    
    return matching_files

def copy_files_with_suffix(files: List[Tuple[Path, str]], output_dir: Path, filename: str, logger) -> int:
    """
    Copy files to output directory organized by filename, with subfolder name as suffix
    
    Args:
        files: List of (file_path, subfolder_name) tuples
        output_dir: Base output directory path
        filename: Original filename (used as folder name)
        logger: Logger instance
    
    Returns:
        Number of files copied
    """
    # Create filename-specific folder within output directory
    filename_dir = output_dir / filename
    filename_dir.mkdir(exist_ok=True, parents=True)
    
    copied_count = 0
    
    for file_path, subfolder_name in files:
        # Create new filename with subfolder suffix
        original_stem = file_path.stem
        original_suffix = file_path.suffix
        new_filename = f"{original_stem}_{subfolder_name}{original_suffix}"
        
        # Destination path within the filename-specific folder
        dest_path = filename_dir / new_filename
        
        # Copy file
        try:
            shutil.copy2(file_path, dest_path)
            logger.info(f"Copied: {file_path} -> {dest_path}")
            copied_count += 1
        except Exception as e:
            logger.error(f"Failed to copy {file_path}: {str(e)}")
    
    return copied_count

def download_results(base_results_path: str, filename: str, split: str, output_dir: str = "results to download"):
    """
    Main function to download specific results files
    
    Args:
        base_results_path: Base path to results directory
        filename: Filename to search for (without extension)
        split: Dataset split (train/test/valid)
        output_dir: Output directory name
    """
    logger = setup_logging()
    
    # Setup paths
    base_path = Path(base_results_path)
    output_path = Path(output_dir)
    
    logger.info(f"Searching for files matching '{filename}' in split '{split}'")
    logger.info(f"Base results path: {base_path}")
    logger.info(f"Output directory: {output_path}")
    
    if not base_path.exists():
        logger.error(f"Base results path does not exist: {base_path}")
        return
    
    try:
        # Find all matching files
        matching_files = find_matching_files(base_path, filename, split)
        
        if not matching_files:
            logger.warning(f"No files found matching '{filename}' in split '{split}'")
            return
        
        logger.info(f"Found {len(matching_files)} matching files:")
        for file_path, subfolder_name in matching_files:
            logger.info(f"  {subfolder_name}/{file_path.name}")
        
        # Copy files with suffix to filename-specific folder
        copied_count = copy_files_with_suffix(matching_files, output_path, filename, logger)
        
        logger.info(f"Successfully copied {copied_count} files to '{output_path / filename}'")
        
        # List the files in the filename-specific output directory
        filename_dir = output_path / filename
        if filename_dir.exists():
            output_files = [f.name for f in filename_dir.iterdir() if f.is_file()]
            logger.info(f"Files in '{filename}' directory:")
            for file in sorted(output_files):
                logger.info(f"  {file}")
    
    except Exception as e:
        logger.error(f"Error during download: {str(e)}")
        raise

def batch_download_results(base_results_path: str, filenames: List[str], split: str, output_dir: str = "results to download"):
    """
    Download multiple files at once
    
    Args:
        base_results_path: Base path to results directory
        filenames: List of filenames to search for
        split: Dataset split (train/test/valid)
        output_dir: Output directory name
    """
    logger = setup_logging()
    
    # Setup paths
    base_path = Path(base_results_path)
    output_path = Path(output_dir)
    
    logger.info(f"Batch downloading {len(filenames)} files from split '{split}'")
    logger.info(f"Files: {filenames}")
    
    total_copied = 0
    
    for filename in filenames:
        try:
            matching_files = find_matching_files(base_path, filename, split)
            
            if matching_files:
                copied_count = copy_files_with_suffix(matching_files, output_path, filename, logger)
                total_copied += copied_count
                logger.info(f"Copied {copied_count} files for '{filename}' to folder '{filename}'")
            else:
                logger.warning(f"No files found for '{filename}'")
        
        except Exception as e:
            logger.error(f"Error processing '{filename}': {str(e)}")
    
    logger.info(f"Batch download completed. Total files copied: {total_copied}")
    
    # List all created directories
    if output_path.exists():
        created_dirs = [d.name for d in output_path.iterdir() if d.is_dir()]
        logger.info(f"Created directories: {sorted(created_dirs)}")

def list_available_files(base_results_path: str, split: str, limit: int = 20):
    """
    List available files in the specified split to help with filename selection
    
    Args:
        base_results_path: Base path to results directory
        split: Dataset split (train/test/valid)
        limit: Maximum number of files to show per subdirectory
    """
    logger = setup_logging()
    
    base_path = Path(base_results_path)
    split_path = base_path / split
    
    if not split_path.exists():
        logger.error(f"Split directory not found: {split_path}")
        return
    
    logger.info(f"Available files in split '{split}':")
    
    for subfolder in sorted(split_path.iterdir()):
        if subfolder.is_dir():
            files = [f.stem for f in subfolder.iterdir() if f.is_file()]
            files = sorted(set(files))  # Remove duplicates and sort
            
            logger.info(f"\n{subfolder.name}:")
            if len(files) > limit:
                logger.info(f"  {files[:limit]} ... (showing first {limit} of {len(files)} files)")
            else:
                logger.info(f"  {files}")

def main():
    parser = argparse.ArgumentParser(description='Download specific results files organized in folders by filename')
    parser.add_argument(
        '--base_path',
        type=str,
        default='/home/badri/Hard Disk Project/neuralnets/depth_fusion/three_branch/model/inference/circularkernel_2_5_real_and_synthetic/',
        help='Base path to results directory'
    )
    parser.add_argument(
        '--filename',
        type=str,
        help='Filename to search for (without extension), e.g., "1_sample_a". Creates a folder with this name.'
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'test', 'val'],
        help='Dataset split to search in'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results to download/circularkernel_2_5_real_and_synthetic',
        help='Base output directory name. Individual filename folders will be created inside this. (default: "results to download")'
    )
    parser.add_argument(
        '--batch',
        type=str,
        nargs='+',
        help='Download multiple files at once, each in its own folder, e.g., --batch file1 file2 file3'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available files in the specified split'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=20,
        help='Limit number of files shown when listing (default: 20)'
    )
    
    args = parser.parse_args()
    
    if args.list:
        if not args.split:
            print("Error: --split is required when using --list")
            return
        list_available_files(args.base_path, args.split, args.limit)
    elif args.batch:
        if not args.split:
            print("Error: --split is required when using --batch")
            return
        batch_download_results(args.base_path, args.batch, args.split, args.output_dir)
    elif args.filename and args.split:
        download_results(args.base_path, args.filename, args.split, args.output_dir)
    else:
        print("Error: Either provide --filename and --split, use --batch with --split, or use --list with --split")
        parser.print_help()

# Example usage and testing
if __name__ == "__main__":
    # Uncomment these for direct testing
    # Example 1: Download single file (creates folder "1_sample_a")
    # download_results(
    #     base_results_path="/home/badri/Hard Disk Project/neuralnets/depth_fusion/three_branch/model/results/squeezeunet_fusion_branches_2_5/predictions/results/",
    #     filename="1_sample_a",
    #     split="train"
    # )
    
    # Example 2: Batch download (creates folders for each filename)
    # batch_download_results(
    #     base_results_path="/home/badri/Hard Disk Project/neuralnets/depth_fusion/three_branch/model/results/squeezeunet_fusion_branches_2_5/predictions/results/",
    #     filenames=["1_sample_a", "1_sample_b", "2_sample_c"],
    #     split="valid"
    # )
    
    main()
