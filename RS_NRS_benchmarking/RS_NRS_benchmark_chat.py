# -*- coding: utf-8 -*-
"""
Script Name: RS_NRS_benchmark_chat.py
Description: This script processes a dataset of README files, classifies them into 'Research' or 'Non-Research' categories
             using OpenAI's GPT-4 model, and saves the results. It includes error handling for API rate limits, connection issues,
             and retries when necessary. The script processes the dataset in batches to avoid hitting rate limits.

Author: B.M. Bruntink
Date: 2025-02-26

Dependencies:
    - openai: The official Python library for OpenAI's API.
    - pandas: A library for data manipulation and analysis, used for reading the dataset and saving the results.
    - nltk: A library used for natural language processing tasks (in this case, to download required resources).
    - re: Python's regular expression library, used for text preprocessing.
    - time: Used to implement delays and handle rate-limiting errors.
    - os: Used for environment variable handling (to securely store the API key).
    
Usage:
    python RS_NRS_benchmark_chat.py
    This will process the 'test_set.csv' input file, classify each README, and output the results to 'chatgpt_classified_results.csv'.
"""

import openai
import pandas as pd
import time
import re
import os
from openai import OpenAIError, RateLimitError, APIConnectionError
import nltk

# Download necessary NLTK resources
nltk.download("punkt")

# Retrieve the OpenAI API key from environment variables
api_key = os.getenv('OPENAI_API_KEY')

# Check if the API key is present
if not api_key:
    raise ValueError("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable.")

# Initialize the OpenAI client with the API key
client = openai.Client(api_key=api_key)

def preprocess_text(text):
    """
    Preprocesses the input text by keeping only alphabetic characters and normalizing whitespace.
    
    Args:
        text (str): The input text to preprocess.
        
    Returns:
        str: The cleaned text containing only alphabetic characters and normalized whitespace.
    """
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

def classify_with_chatgpt(readme_content, max_length=500):
    """
    Classifies a README file as either 'Research' or 'Non-Research' using OpenAI's GPT-4 model.
    
    Args:
        readme_content (str): The content of the README file to classify.
        max_length (int): The maximum length of the input text to be passed to the GPT model (default 500).
        
    Returns:
        str: The classification result, either 'Research', 'Non-Research', or 'Unknown' if classification fails.
    """
    preprocessed_content = preprocess_text(readme_content)
    truncated_content = preprocessed_content[:max_length] if isinstance(preprocessed_content, str) else ""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": 
                    "You are a classifier that identifies whether software README files describe research or non-research software. "
                    "Respond with only 'Research' or 'Non-Research'."},
                {"role": "user", "content": f"Classify the following README as 'Research' or 'Non-Research':\n\n{truncated_content}"}
            ],
            max_tokens=5,
            temperature=0
        )
        result = response.choices[0].message.content.strip()

        if result.lower() in ["research", "'research'", '"research"']:
            return "Research"
        elif result.lower() in ["non-research", "'non-research'", '"non-research"']:
            return "Non-Research"
        else:
            return "Unknown"
    except RateLimitError as e:
        print(f"Rate limit reached: {e}")
        delay = int(e.response.headers.get("Retry-After", 5))
        print(f"Retrying after {delay} seconds...")
        time.sleep(delay)
        return classify_with_chatgpt(readme_content, max_length)
    except APIConnectionError as e:
        print(f"API connection error: {e}")
        return None
    except OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def process_in_batches(df, batch_size, output_file):
    """
    Processes the dataset in manageable batches to avoid hitting API rate limits and save progress.
    
    Args:
        df (DataFrame): The pandas DataFrame containing the dataset.
        batch_size (int): The number of records to process in each batch.
        output_file (str): The path to save the output CSV file with the classified results.
        
    Returns:
        list: The list of results after processing all batches.
    """
    results = []
    total_batches = (len(df) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        print(f"Processing batch {batch_num + 1}/{total_batches}")
        batch = df.iloc[batch_num * batch_size: (batch_num + 1) * batch_size]

        for index, row in batch.iterrows():
            prediction = classify_with_chatgpt(row['Raw README Content'])
            results.append({
                "Raw README Content": row['Raw README Content'],
                "Label": row['Label'],
                "ChatGPT Prediction": prediction
            })

        pd.DataFrame(results).to_csv(output_file, index=False, encoding="utf-8")
        print(f"Saved progress to {output_file}")
        time.sleep(5)  # Brief delay to prevent rate limits

    return results

# Main script execution
if __name__ == "__main__":
    test_data_path = "test_set.csv"
    output_file = "chatgpt_classified_results.csv"
    batch_size = 5

    try:
        df = pd.read_csv(test_data_path)

        required_columns = ['Raw README Content', 'Label']  # ✅ Updated to match your CSV
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Expected: {required_columns}")

        print(f"Classifying {len(df)} README files using ChatGPT...")

        process_in_batches(df, batch_size, output_file)

        print(f"Classification complete. Results saved to {output_file}.")
    except FileNotFoundError:
        print(f"File '{test_data_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
