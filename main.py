import os
import pandas as pd
import importlib.util
from pathlib import Path
import re
import sys
import json

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from reporting import generate_report

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
            return {}
    return {}

def save_processing_state(state, state_file="processing_state.json"):
    """Save the processing state to a JSON file"""
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

def process_csv(input_path, recipe_module, source_lang, target_lang):
    df = pd.read_csv(input_path)
    
    # Process with the specified language codes
    processed_df = recipe_module.process_dataframe(df, source_lang=source_lang, target_lang=target_lang)
    
    return processed_df

def get_output_filename(input_filename, recipe_name):
    """Generate output filename with recipe prefix"""
    name, ext = os.path.splitext(input_filename)
    return f"{name}_{recipe_name}{ext}"

def run_translation(input_dir, output_dir, recipes, state):
    """Run translation and similarity comparison with reference text"""
    print("Running translation and similarity comparison...")
    
    # Process each CSV file in the input directory
    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
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
                if state.get(state_key, {}).get('completed', False):
                    print(f"Skipping {recipe_name} for {file} ({source_lang}-{target_lang}) - already processed")
                    continue
                
                print(f"Processing {input_path} with recipe {recipe_name} for {source_lang}-{target_lang}")
                print(f"Output will be saved to {output_path}")
                
                try:
                    result_df = process_csv(input_path, recipe_module, source_lang, target_lang)
                    result_df.to_csv(output_path, index=False)
                    
                    # Update state
                    state[state_key] = {'completed': True}
                    save_processing_state(state)
                    
                    print(f"Completed {recipe_name} on {file} for {source_lang}-{target_lang}")
                except Exception as e:
                    print(f"Error applying {recipe_name} to {file} for {source_lang}-{target_lang}: {str(e)}")

def display_menu():
    """Display the menu options"""
    print("\n" + "="*50)
    print("Translation Pipeline Menu")
    print("="*50)
    print("1. Run translation and similarity comparison")
    print("2. Generate reports only")
    print("3. Reset processing state")
    print("4. Exit")
    print("="*50)
    
    while True:
        try:
            choice = input("Please select an option (1-4): ")
            if choice in ["1", "2", "3", "4"]:
                return choice
            else:
                print("Invalid option. Please enter a number between 1 and 4.")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

def reset_processing_state(state_file="processing_state.json"):
    """Reset the processing state"""
    if os.path.exists(state_file):
        os.remove(state_file)
    print("Processing state has been reset.")

def main():
    # Define input and output directories
    input_dir = "input"
    output_dir = "output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load recipes
    recipes = load_recipes()
    
    # Load processing state
    state = load_processing_state()
    
    while True:
        choice = display_menu()
        
        if choice == "1":
            run_translation(input_dir, output_dir, recipes, state)
        elif choice == "2":
            generate_report(output_dir)
        elif choice == "3":
            reset_processing_state()
            state = {}  # Reset in-memory state
        elif choice == "4":
            print("Exiting...")
            break

if __name__ == "__main__":
    main()
