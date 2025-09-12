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
                model="deepseek-ai/deepseek-v3.1",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                top_p=0.95,
                max_tokens=2024,
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

def process_dataframe(df, source_lang, target_lang):
    """Main processing function"""
    print(f"Translation: NVIDIA Build API | Similarity: Compare with reference")
    print(f"Rate limiting: 38 requests per minute (~1.58 seconds between requests)")

    result_df = df.copy()
    result_df['translated'] = ""
    result_df['similarity_score'] = 0.0

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
            
            # Calculate similarity with reference text
            if 'ref' in row and pd.notna(row['ref']):
                similarity = calculate_similarity(translation, row['ref'])
                result_df.at[i, 'similarity_score'] = similarity
                print(f"  → Similarity with reference: {similarity:.4f}")
        else:
            print("  → [Translation failed]")
        
        # Rate limiting: wait before next request (except after the last one)
        if i < total_texts - 1:
            print(f"Waiting {delay_between_requests:.2f} seconds before next request...")
            time.sleep(delay_between_requests)

    print("Translation process completed!")
    return result_df
