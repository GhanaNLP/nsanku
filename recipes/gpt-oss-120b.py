import pandas as pd
import time
import os
import re
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from typing import List

# Load environment variables from .env file
load_dotenv()

# Initialize NVIDIA API client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_BUILD_API_KEY")
)

# Add utils to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from language_mapping import get_language_name, get_iso2_code, get_nllb_code

# Initialize similarity model
similarity_model_name = "sentence-transformers/all-mpnet-base-v2"
similarity_model = SentenceTransformer(similarity_model_name)

def translate_text_with_nvidia(text, source_lang, target_lang, max_retries=5):
    """Translate text using NVIDIA Build API via OpenAI client"""
    source_lang_name = get_language_name(source_lang)
    target_lang_name = get_language_name(target_lang)

    prompt = f"Translate the following {source_lang_name} text into {target_lang_name} and return ONLY the translation inside square brackets:\n\n{text}"

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                top_p=0.95,
                max_tokens=2024,
                reasoning_effort="low",
                stream=False
            )
            
            # Directly get the response content
            response_text = completion.choices[0].message.content
            
            # Extract text from brackets if present, otherwise use as-is
            match = re.search(r'\[(.*?)\]', response_text, flags=re.S)
            if match:
                return match.group(1).strip()
            return response_text.strip()
                
        except Exception as e:
            print(f"Attempt {attempt+1} failed for text '{text}': {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return ""

def calculate_similarity(translated, reference):
    """Calculate cosine similarity between translated text and reference text"""
    try:
        if not translated or not reference:
            return 0.0

        embeddings = similarity_model.encode([translated, reference])
        return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    except Exception as e:
        print(f"Error calculating similarity: {str(e)}")
        return 0.0

def translation_only(df, source_lang, target_lang):
    """Only perform translation without similarity calculation"""
    print(f"Translation: NVIDIA Build API")
    print(f"Rate limiting: 38 requests per minute (~1.58 seconds between requests)")

    result_df = df.copy()
    result_df['translated'] = ""

    # Calculate delay between requests to achieve 38 requests per minute
    delay_between_requests = 60 / 38  # Approximately 1.58 seconds

    # Translations with rate limiting
    total_texts = len(result_df)
    
    for i, row in result_df.iterrows():
        text = row['text']
        print(f"Translating {i+1}/{total_texts}: {text[:50]}...")
        
        translation = translate_text_with_nvidia(text, source_lang, target_lang)
        result_df.at[i, 'translated'] = translation
        
        # Show translation result
        if translation:
            print(f"  → {translation[:50]}...")
        else:
            print("  → [Translation failed]")
        
        # Rate limiting: wait before next request (except after the last one)
        if i < total_texts - 1:
            print(f"Waiting {delay_between_requests:.2f} seconds before next request...")
            time.sleep(delay_between_requests)

    print("Translation process completed!")
    return result_df

def similarity_only(df, batch_size=32):
    """Only calculate similarity between translated text and reference using batch processing"""
    print("Calculating similarity scores with batch processing...")
    
    result_df = df.copy()
    
    # Check if 'translated' column exists
    if 'translated' not in result_df.columns:
        print("Error: No 'translated' column found in the DataFrame")
        return result_df
    
    # Check if 'ref' column exists
    if 'ref' not in result_df.columns:
        print("Error: No 'ref' column found in the DataFrame")
        return result_df
    
    # Prepare data for batch processing
    translated_texts = result_df['translated'].fillna('').tolist()
    ref_texts = result_df['ref'].fillna('').tolist()
    
    # Calculate similarity in batches
    similarities = []
    total_batches = (len(translated_texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(translated_texts), batch_size):
        batch_translated = translated_texts[i:i+batch_size]
        batch_ref = ref_texts[i:i+batch_size]
        
        # Calculate embeddings for both batches
        embeddings_translated = similarity_model.encode(batch_translated, convert_to_tensor=True)
        embeddings_ref = similarity_model.encode(batch_ref, convert_to_tensor=True)
        
        # Calculate cosine similarities
        batch_similarities = util.pytorch_cos_sim(embeddings_translated, embeddings_ref)
        
        # Extract the diagonal (each translation compared to its corresponding reference)
        batch_diagonal = batch_similarities.diag().cpu().numpy()
        similarities.extend(batch_diagonal)
        
        # Show progress
        batch_num = i // batch_size + 1
        print(f"Processed batch {batch_num}/{total_batches}")
    
    result_df['similarity_score'] = similarities
    
    print("Similarity calculation completed!")
    return result_df

def process_dataframe(df, source_lang, target_lang):
    """Full processing function (translation + similarity)"""
    # First do translation
    df = translation_only(df, source_lang, target_lang)
    
    # Then calculate similarity
    df = similarity_only(df)
    
    return df
