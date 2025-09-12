import pandas as pd
import time
import os
import re
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from typing import List
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize NVIDIA openai client
api_key = os.getenv("NVIDIA_BUILD_API_KEY")
if not api_key:
    raise ValueError("NVIDIA_BUILD_API_KEY not found in environment variables")

# Initialize OpenAI client for NVIDIA API
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# Add utils to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from language_mapping import get_language_name, get_iso2_code, get_nllb_code

# Initialize similarity model
similarity_model_name = "sentence-transformers/all-mpnet-base-v2"
similarity_model = SentenceTransformer(similarity_model_name)

# Initialize NLLB model for backtranslation
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device} for NLLB backtranslation")

# Load NLLB model and tokenizer for backtranslation
backtranslation_model_name = "facebook/nllb-200-3.3B"
backtranslation_tokenizer = AutoTokenizer.from_pretrained(backtranslation_model_name)
backtranslation_model = AutoModelForSeq2SeqLM.from_pretrained(backtranslation_model_name).to(device)

def extract_text_from_brackets(text):
    """Extract text from square brackets, return empty string if not found"""
    match = re.search(r'\[(.*?)\]', text, flags=re.S)
    if match:
        return match.group(1).strip()
    return ""

def get_model_list():
    """Read model names from models.txt file in the recipes directory"""
    models_file = os.path.join(os.path.dirname(__file__), 'models_nvidia.txt')
    models = []
    
    try:
        with open(models_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    models.append(line)
    except FileNotFoundError:
        print(f"models.txt file not found in {os.path.dirname(__file__)}")
        print("No models to process - exiting")
        return []
    
    return models

def translate_text_with_nvidia(text, source_lang, target_lang, model_name, max_retries=5):
    """Translate text using NVIDIA Build API via OpenAI client with specified model"""
    source_lang_name = get_language_name(source_lang)
    target_lang_name = get_language_name(target_lang)

    for attempt in range(max_retries):
        try:
            # Call NVIDIA API
            completion = nvidia_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a translation assistant that translates text from {source_lang_name} to {target_lang_name}. "
                                   f"Return ONLY the translation inside square brackets."
                    },
                    {
                        "role": "user",
                        "content": f"Translate the following {source_lang_name} text into {target_lang_name}:\n\n{text}"
                    }
                ],
                temperature=0.3,
                top_p=0.95,
                max_tokens=2096,
                stream=False
            )

            # Simple approach - just return the response as is
            response_text = completion.choices[0].message.content
            
            if response_text is None:
                print(f"[{model_name}] Response is None on attempt {attempt+1}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return ""

            # Return the response directly without any parsing
            return response_text.strip()

        except Exception as e:
            print(f"[{model_name}] Error on attempt {attempt+1} for text '{text[:50]}...': {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return ""

    return ""

def backtranslate_with_nllb(texts: List[str], source_lang: str, target_lang: str) -> List[str]:
    """Backtranslate texts using NLLB-3B model"""
    # Convert language codes to NLLB format
    nllb_source = get_nllb_code(target_lang)  # Note: target_lang becomes source for backtranslation
    nllb_target = get_nllb_code(source_lang)  # Note: source_lang becomes target for backtranslation
    
    backtranslations = []
    
    for i, text in enumerate(texts):
        if not text:  # Skip empty texts
            backtranslations.append("")
            continue
            
        try:
            # Set source language
            backtranslation_tokenizer.src_lang = nllb_source
            
            # Encode the text
            inputs = backtranslation_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            
            # Generate translation
            generated_tokens = backtranslation_model.generate(
                **inputs,
                forced_bos_token_id=backtranslation_tokenizer.convert_tokens_to_ids(nllb_target),
                max_length=512
            )
            
            # Decode the translation
            backtranslation = backtranslation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            backtranslations.append(backtranslation)
            
            # Show progress
            print(f"Backtranslated {i+1}/{len(texts)}: {text[:50]}... → {backtranslation[:50]}...")
            
        except Exception as e:
            print(f"NLLB backtranslation failed for text '{text}': {str(e)}")
            backtranslations.append("")
            
        # Add a small delay to prevent overwhelming the system
        time.sleep(0.1)
    
    return backtranslations

def calculate_similarity(original, backtranslated):
    """Calculate cosine similarity between original and backtranslated text"""
    try:
        if not original or not backtranslated:
            return 0.0

        clean_backtranslated = extract_text_from_brackets(backtranslated)
        if not clean_backtranslated:
            clean_backtranslated = backtranslated

        embeddings = similarity_model.encode([original, clean_backtranslated])
        return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    except Exception as e:
        print(f"Error calculating similarity: {str(e)}")
        return 0.0

def process_dataframe(df, source_lang, target_lang):
    """Main processing function for multiple models"""
    # Get the list of models to test
    models = get_model_list()
    if not models:
        print("No models found to process")
        return pd.DataFrame()
    
    print(f"Found {len(models)} models to test: {', '.join(models)}")
    
    # Calculate delay between requests to achieve 38 requests per minute
    delay_between_requests = 60 / 38  # Approximately 1.58 seconds
    
    # Create a list to store results for all models
    all_results = []
    
    for model_name in models:
        print(f"\n{'='*80}")
        print(f"Processing with model: {model_name}")
        print(f"{'='*80}")
        
        try:
            # Test if model is accessible by making a simple API call
            test_response = nvidia_client.models.retrieve(model_name)
            if not hasattr(test_response, 'id'):
                print(f"Model {model_name} not found or inaccessible. Skipping.")
                continue
                
        except Exception as e:
            print(f"Error accessing model {model_name}: {str(e)}. Skipping.")
            continue
        
        result_df = df.copy()
        result_df['model'] = model_name  # Add model name column
        result_df['translated'] = ""
        result_df['backtranslated'] = ""
        result_df['similarity_score'] = 0.0

        # Forward translations with rate limiting
        translations = []
        total_texts = len(result_df)
        processing_failed = False
        
        for i, text in enumerate(result_df['text']):
            print(f"Translating {i+1}/{total_texts} with {model_name}: {text[:50]}...")
            translation = translate_text_with_nvidia(text, source_lang, target_lang, model_name)
            translations.append(translation)
            
            # Show translation result
            if translation:
                print(f"  → {translation[:50]}...")
            else:
                print("  → [Translation failed]")
                processing_failed = True
            
            # Rate limiting: wait before next request (except after the last one)
            if i < total_texts - 1:
                print(f"Waiting {delay_between_requests:.2f} seconds before next request...")
                time.sleep(delay_between_requests)

        # Skip this model if translation failed for all texts
        if processing_failed and all(not t for t in translations):
            print(f"Skipping model {model_name} due to complete translation failure")
            continue

        result_df['translated'] = translations

        # Backtranslations using NLLB-3B
        print(f"Starting backtranslation with NLLB-3.3B for model {model_name}...")
        backtranslations = backtranslate_with_nllb(result_df['translated'].tolist(), source_lang, target_lang)
        result_df['backtranslated'] = backtranslations

        # Calculate similarity
        print(f"Calculating similarity scores for model {model_name}...")
        result_df['similarity_score'] = result_df.apply(
            lambda row: calculate_similarity(row['text'], row['backtranslated']) if row['backtranslated'] else 0.0,
            axis=1
        )

        # Add to all results
        all_results.append(result_df)
        print(f"Completed processing for model: {model_name}")

    if not all_results:
        print("No models were successfully processed")
        return pd.DataFrame()
    
    # Combine all results into a single DataFrame
    combined_df = pd.concat(all_results, ignore_index=True)
    print(f"\nCompleted all models! Total rows: {len(combined_df)}")
    
    return combined_df
