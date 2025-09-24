import os
import pandas as pd
from collections import defaultdict
import zipfile
import tempfile

def combine_matching_csvs_from_zips(root_folder, output_path=None):
    """
    Combine CSV files from zipped output folders into a single output_combined folder.
    
    Args:
        root_folder (str): Path to the root folder containing zip files
        output_path (str, optional): Specific path for output_combined folder. 
                                   If None, creates in root_folder.
    """
    
    # Set output directory
    if output_path is None:
        output_dir = os.path.join(root_folder, "output_combined")
    else:
        output_dir = output_path
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output will be saved to: {output_dir}")
    
    # Dictionary to group files by language pair and CSV name across ALL zip files
    file_groups = defaultdict(list)
    
    # Find all zip files in the root folder
    zip_files = [f for f in os.listdir(root_folder) if f.endswith('.zip')]
    
    if not zip_files:
        print("No zip files found in the root folder!")
        return output_dir
    
    print(f"Found {len(zip_files)} zip files to process: {zip_files}")
    
    # Create a temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        # Process each zip file and collect all CSV files
        for zip_file in zip_files:
            zip_path = os.path.join(root_folder, zip_file)
            print(f"\nProcessing zip file: {zip_file}")
            
            try:
                # Extract zip to temporary directory
                extract_path = os.path.join(temp_dir, os.path.splitext(zip_file)[0])
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                print(f"  ✓ Extracted to temporary location")
                
                # Walk through the extracted content to find CSV files
                for foldername, subfolders, filenames in os.walk(extract_path):
                    # Skip the root of the extracted content, look for language pair folders
                    if foldername == extract_path:
                        continue
                        
                    # Get the language pair name (the immediate subfolder)
                    lang_pair = os.path.basename(foldername)
                    
                    for filename in filenames:
                        if filename.endswith('.csv'):
                            full_path = os.path.join(foldername, filename)
                            file_groups[(lang_pair, filename)].append((full_path, zip_file))
                            print(f"  - Found: {lang_pair}/{filename} from {zip_file}")
                            
            except Exception as e:
                print(f"  ✗ Error processing {zip_file}: {e}")
        
        # Now combine all matching CSV files across different zip files
        print(f"\n{'='*60}")
        print("COMBINING CSV FILES ACROSS ALL ZIP FILES")
        print(f"{'='*60}")
        
        total_combined = 0
        total_copied = 0
        
        for (lang_pair, csv_name), file_info in file_groups.items():
            # Create language pair subfolder in output_combined
            lang_pair_output_dir = os.path.join(output_dir, lang_pair)
            os.makedirs(lang_pair_output_dir, exist_ok=True)
            output_file_path = os.path.join(lang_pair_output_dir, csv_name)
            
            if len(file_info) > 1:
                # COMBINE files from multiple zip files
                print(f"\n🔗 Combining {len(file_info)} instances of {lang_pair}/{csv_name}:")
                
                dataframes = []
                for file_path, zip_source in file_info:
                    try:
                        df = pd.read_csv(file_path)
                        # Add source column to track which zip file this came from
                        df['source_zip_file'] = os.path.splitext(zip_source)[0]
                        dataframes.append(df)
                        print(f"  - Added: {os.path.splitext(zip_source)[0]} ({len(df)} rows)")
                    except Exception as e:
                        print(f"  - Error reading {file_path}: {e}")
                
                if dataframes:
                    # Combine all dataframes
                    combined_df = pd.concat(dataframes, ignore_index=True)
                    combined_df.to_csv(output_file_path, index=False)
                    
                    total_rows = len(combined_df)
                    total_sources = len(set([src for _, src in file_info]))
                    print(f"  ✅ COMBINED: {output_file_path}")
                    print(f"     Total rows: {total_rows}, Sources: {total_sources}")
                    total_combined += 1
                else:
                    print(f"  ❌ Failed to combine {lang_pair}/{csv_name}")
                    
            else:
                # Only one file found for this language pair/CSV combination
                file_path, zip_source = file_info[0]
                try:
                    df = pd.read_csv(file_path)
                    df['source_zip_file'] = os.path.splitext(zip_source)[0]
                    df.to_csv(output_file_path, index=False)
                    print(f"📋 Copied single file: {lang_pair}/{csv_name} from {os.path.splitext(zip_source)[0]}")
                    total_copied += 1
                except Exception as e:
                    print(f"❌ Error copying {file_path}: {e}")
    
    print(f"\n{'='*60}")
    print("COMBINATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total files combined: {total_combined}")
    print(f"Total files copied (single instance): {total_copied}")
    print(f"Combined data available in: {output_dir}")
    print(f"{'='*60}")
    
    return output_dir

# Example usage
if __name__ == "__main__":
    # Specify your paths here
    root_path = "/home/owusus/Documents/GitHub/nsanku/output"  # Folder containing the zip files
    
    combine_matching_csvs_from_zips(root_path, "/home/owusus/Documents/GitHub/nsanku/output_combined")
    
