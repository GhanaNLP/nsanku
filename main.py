import os
import pandas as pd
import importlib.util
from pathlib import Path
import re
import sys
import json
import random

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from reporting import generate_report

def setup_api_key():
    """Check for .env file and create it if missing with user-provided API key"""
    env_file = '.env'
    if not os.path.exists(env_file):
        print("No .env file found.")
        print("Please enter your NVIDIA API key obtained from https://build.nvidia.com/")
        api_key = input("API Key: ").strip()
        
        with open(env_file, 'w') as f:
            f.write(f'NVIDIA_BUILD_API_KEY={api_key}\n')
        
        print(".env file created with your API key.")
        return api_key
    else:
        # Load existing API key from .env file
        nvidia_key = None
        
        with open(env_file, 'r') as f:
            for line in f:
                if line.startswith('NVIDIA_BUILD_API_KEY='):
                    nvidia_key = line.strip().split('=', 1)[1]
        
        # If we have the NVIDIA key, return it
        if nvidia_key:
            return nvidia_key
        
        # If we get here, the .env file exists but doesn't contain the key
        print("Existing .env file found but no NVIDIA API key detected.")
        api_key = input("Please enter your NVIDIA API key: ").strip()
        
        with open(env_file, 'w') as f:
            f.write(f'NVIDIA_BUILD_API_KEY={api_key}\n')
        
        print("API key added to .env file.")
        return api_key

def load_recipes(recipes_dir="recipes"):
    recipes = {}
    for file in os.listdir(recipes_dir):
        if file.endswith(".py") and file != "__init__.py":
            module_name = file[:-3]
            spec = importlib.util.spec_from_file_location(
                module_name, os.path.join(recipes_dir, file)
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            recipes[module_name] = module
    return recipes

def extract_language_pair_from_filename(filename):
    """Extract language pair from filename in format source-target.csv"""
    pattern = r'^([a-zA-Z]+)-([a-zA-Z]+)\.csv$'
    match = re.match(pattern, filename)
    if match:
        return match.group(1), match.group(2)
    return None, None

def load_processing_state(state_file="processing_state.json"):
    """Load the processing state from a JSON file"""
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                return json.load(f)
        except:
            # If file is corrupted, create a new one
            print("State file corrupted, creating a new one")
            with open(state_file, 'w') as f:
                json.dump({}, f)
            return {}
    else:
        # Create an empty state file if it doesn't exist
        print("Creating new state file")
        with open(state_file, 'w') as f:
            json.dump({}, f)
        return {}

def save_processing_state(state, state_file="processing_state.json"):
    """Save the processing state to a JSON file"""
    try:
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"State saved with {len(state)} entries")
    except Exception as e:
        print(f"Error saving state: {str(e)}")

def sample_dataframe(df, sample_size=5):
    """Randomly sample rows from dataframe with proper shuffling"""
    if len(df) <= sample_size:
        return df.copy()
    
    # Create a copy and shuffle randomly
    shuffled_df = df.copy().sample(frac=1, random_state=None).reset_index(drop=True)
    
    # Take the first sample_size rows
    return shuffled_df.head(sample_size)

def process_csv(input_path, recipe_module, source_lang, target_lang, mode="full"):
    # Read the CSV and sample 5 rows randomly
    df = pd.read_csv(input_path)
    sampled_df = sample_dataframe(df, sample_size=5)
    
    print(f"Processing {len(sampled_df)} randomly sampled rows from {len(df)} total rows")
    
    # Process with the specified language codes
    if mode == "translation_only" and hasattr(recipe_module, 'translation_only'):
        processed_df = recipe_module.translation_only(sampled_df, source_lang=source_lang, target_lang=target_lang)
    elif mode == "similarity_only" and hasattr(recipe_module, 'similarity_only'):
        processed_df = recipe_module.similarity_only(sampled_df)
    else:
        processed_df = recipe_module.process_dataframe(sampled_df, source_lang=source_lang, target_lang=target_lang)
    
    return processed_df

def get_output_filename(input_filename, recipe_name):
    """Generate output filename with recipe prefix"""
    name, ext = os.path.splitext(input_filename)
    return f"{name}_{recipe_name}{ext}"

def run_full_process_automatic(input_dir, output_dir, recipes):
    """Run the full process automatically without user intervention"""
    print("="*60)
    print("Starting automatic end-to-end processing")
    print("="*60)
    
    # Load processing state
    state = load_processing_state()
    print(f"Loaded state with {len(state)} entries")
    
    # Process each CSV file in the input directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    print(f"Found {len(csv_files)} CSV files to process")
    
    for file in csv_files:
        # Extract language pair from filename
        source_lang, target_lang = extract_language_pair_from_filename(file)
        if not source_lang or not target_lang:
            print(f"Skipping {file}: filename should be in format 'source-target.csv'")
            continue
            
        input_path = os.path.join(input_dir, file)
        
        # Create output directory for this language pair
        lang_pair_dir = os.path.join(output_dir, f"{source_lang}-{target_lang}")
        os.makedirs(lang_pair_dir, exist_ok=True)
        
        for recipe_name, recipe_module in recipes.items():
            # Generate recipe-specific output filename
            output_filename = get_output_filename(file, recipe_name)
            output_path = os.path.join(lang_pair_dir, output_filename)
            
            # Check if this recipe has already processed this file
            state_key = f"{source_lang}-{target_lang}/{file}/{recipe_name}"
            if state.get(state_key, {}).get('similarity_completed', False):
                print(f"Skipping {recipe_name} for {file} ({source_lang}-{target_lang}) - already processed")
                continue
            
            print(f"\nProcessing {file} with recipe {recipe_name} for {source_lang}-{target_lang}")
            print(f"Input: {input_path}")
            print(f"Output: {output_path}")
            
            try:
                # Run the full process (translation + similarity)
                result_df = process_csv(input_path, recipe_module, source_lang, target_lang)
                result_df.to_csv(output_path, index=False)
                
                # Update state
                state[state_key] = {
                    'translation_completed': True,
                    'similarity_completed': True,
                    'rows_processed': len(result_df),
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                save_processing_state(state)
                
                print(f"✓ Completed {recipe_name} on {file} for {source_lang}-{target_lang}")
                print(f"  Processed {len(result_df)} rows")
                
            except Exception as e:
                print(f"✗ Error applying {recipe_name} to {file} for {source_lang}-{target_lang}: {str(e)}")
    
    print(f"\nFull process completed! Final state: {len(state)} entries")
    
    # Generate reports automatically
    print("\n" + "="*60)
    print("Generating reports...")
    print("="*60)
    generate_report(output_dir)
    
    print("\n" + "="*60)
    print("Automatic processing completed successfully!")
    print("="*60)

def main():
    # Setup API key first
    api_key = setup_api_key()
    os.environ['NVIDIA_BUILD_API_KEY'] = api_key
    
    # Define input and output directories
    input_dir = "input"
    output_dir = "output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load recipes
    recipes = load_recipes()
    print(f"Loaded {len(recipes)} recipes: {list(recipes.keys())}")
    
    # Run the full automatic process
    run_full_process_automatic(input_dir, output_dir, recipes)

if __name__ == "__main__":
    main()
