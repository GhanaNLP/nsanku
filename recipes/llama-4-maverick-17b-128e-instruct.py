import pandas as pd
import time
import os
import re
from groq import Groq
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from typing import List

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Add utils to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from reporting import get_language_name

# Initialize similarity model
similarity_model_name = "sentence-transformers/all-mpnet-base-v2"
similarity_model = SentenceTransformer(similarity_model_name)

def translate_text_with_groq(text, source_lang, target_lang, max_retries=5):
    """Translate text using Groq API"""
    source_lang_name = get_language_name(source_lang)
    target_lang_name = get_language_name(target_lang)

    prompt = f"Translate the following {source_lang_name} text into {target_lang_name} and return ONLY the translation inside square brackets:\n\n{text}"

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",  # You can change this to other Groq models
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2024,
                top_p=0.95,
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
    print(f"Translation: Groq API")
    print(f"Model: meta-llama/llama-4-scout-17b-16e-instruct")

    result_df = df.copy()
    result_df['translated'] = ""

    # Groq has very fast inference, but we should still add a small delay to be respectful
    delay_between_requests = 2.0  # 100ms delay

    # Translations with rate limiting
    total_texts = len(result_df)
    
    for i, row in result_df.iterrows():
        text = row['text']
        print(f"Translating {i+1}/{total_texts}: {text[:50]}...")
        
        translation = translate_text_with_groq(text, source_lang, target_lang)
        result_df.at[i, 'translated'] = translation
        
        # Show translation result
        if translation:
            print(f"  → {translation[:50]}...")
        else:
            print("  → [Translation failed]")
        
        # Small delay to be respectful of the API
        if i < total_texts - 1:
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
