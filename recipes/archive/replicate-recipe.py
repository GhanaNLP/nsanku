import os
import re

def generate_model_files(template_file, models_file):
    # Read the template file
    with open(template_file, 'r') as f:
        template_content = f.read()
    
    # Read the models list
    with open(models_file, 'r') as f:
        models = [line.strip() for line in f if line.strip()]
    
    # Create output directory if it doesn't exist
    output_dir = "generated_models"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a file for each model
    for model in models:
        # Replace the model name in the template
        new_content = template_content.replace(
            '"bytedance/seed-oss-36b-instruct"',
            f'"{model}"'
        )
        
        # Create a safe filename from the model name
        safe_model_name = re.sub(r'[^a-zA-Z0-9]', '_', model)
        output_filename = f"nvidia_{safe_model_name}.py"
        output_path = os.path.join(output_dir, output_filename)
        
        # Write the new file
        with open(output_path, 'w') as f:
            f.write(new_content)
        
        print(f"Generated: {output_filename}")
    
    print(f"\nGenerated {len(models)} model files in the '{output_dir}' directory.")

if __name__ == "__main__":
    generate_model_files("template_nvidia-new.py", "models_nvidia.txt")
