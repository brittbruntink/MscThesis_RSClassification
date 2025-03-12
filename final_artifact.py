#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: final_artifact.py
Description: This script classifies GitHub repositories as research or non-research software based on their README files and suggests an academic domain classification.
Author: B.M. Bruntink
Date: 2025-03-12

Dependencies:
    - joblib
    - requests
    - nltk
    - beautifulsoup4

Usage:
    python classify_software.py [GitHub README URL]
"""

import joblib
import re
import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

# Load stopwords
stop_words = set(stopwords.words("english"))

# Load models
MODEL_FILE = "best_model.pkl"
DOMAIN_MODEL_FILE = "domain_model.pkl"  
DOMAIN_VECTORIZER_FILE = "domain_vectorizer.pkl"  # Vectorizer for domain model

try:
    model = joblib.load(MODEL_FILE)
    domain_model = joblib.load(DOMAIN_MODEL_FILE)
    domain_vectorizer = joblib.load(DOMAIN_VECTORIZER_FILE)  # Load vectorizer
    print("✅ Successfully loaded all models and vectorizer.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

# Mapping abbreviated domain names to full terms
DOMAIN_MAPPING = {
    "CS": "Computer Science",
    "Civil": "Civil Engineering",
    "EE": "Electrical Engineering",
    "Mech": "Mechanical Engineering",
    "Medical": "Medical Sciences",
    "Psychology": "Psychology",
    "Bio": "Biochemistry",
    "Uncertain": "Uncertain Domain"
}

README_FILE_NAMES = ['README.md', 'readme.md', 'README.rst', 'readme.rst', 'README.txt', 'readme.txt', 'README.markdown', 'readme.markdown']
BRANCHES = ['main', 'master', 'develop', 'gh-pages']

def is_repository_link(url):
    """
    Check if the provided URL is a GitHub or GitLab link.

    Parameters:
        url (str): The URL to check.

    Returns:
        bool: True if the URL is a valid GitHub or GitLab repository link, False otherwise.
    """
    if not isinstance(url, str):
        return False
    repo_pattern = r'^(https?://)?(www\.)?(github\.com|gitlab\.com)/.*$'
    return bool(re.match(repo_pattern, url))

def construct_raw_readme_url(url):
    """
    Construct the raw README URL from a given repository URL.

    Parameters:
        url (str): The repository URL.

    Returns:
        str: The raw README URL or "Not Found" if the README couldn't be found.
    """
    cleaned_url = url.split('#')[0].split('?')[0]  # Clean the URL by removing fragments and query parameters
    match = re.match(r'^(https?://)?(www\.)?(github\.com|gitlab\.com)/([^/]+)/([^/]+)', cleaned_url)
    if match:
        platform = match.group(3)
        owner = match.group(4)
        repo = match.group(5)  # Correctly extract the repository name
        base_url = "raw.githubusercontent.com" if "github" in platform else "gitlab.com"
        
        # Check for different README file names and common branches
        for readme_file in README_FILE_NAMES:
            for branch in BRANCHES:
                raw_readme_url = f'https://{base_url}/{owner}/{repo}/{branch}/{readme_file}'
                if is_readme_url_valid(raw_readme_url):
                    return raw_readme_url
    return "Not Found"

def is_readme_url_valid(readme_url):
    """
    Check if the README URL is valid by sending a HEAD request.

    Parameters:
        readme_url (str): The raw README URL to check.

    Returns:
        bool: True if the URL is valid, False otherwise.
    """
    try:
        response = requests.head(readme_url, allow_redirects=True)
        return response.status_code == 200
    except Exception:
        return False

def fetch_readme_from_url(repo_url):
    """
    Fetches the README content from GitHub or GitLab repositories.

    Parameters:
        repo_url (str): The URL of the GitHub or GitLab repository.

    Returns:
        tuple: A tuple containing the README content (str) and the repository name (str).
        If the README can't be fetched, returns (None, None).
    """
    try:
        raw_readme_url = construct_raw_readme_url(repo_url)
        if raw_readme_url == "Not Found":
            return None, None
        
        response = requests.get(raw_readme_url)
        
        if response.status_code == 200:
            repo_name = repo_url.split('/')[4]  # Correctly extract repository name (second segment after 'github.com')
            return response.text, repo_name
        else:
            print(f"❌ Failed to fetch README. Status code: {response.status_code}")
            return None, None
    except Exception as e:
        print(f"❌ Error fetching README: {e}")
        return None, None

def preprocess_text_with_html_removal(text):
    """
    Preprocess the text before classification: Lowercasing, Removing HTML, URLs, and Stopword Removal.

    Parameters:
        text (str): The raw text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags using BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()  # Get plain text, without HTML tags
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)  # Remove URLs
    
    # Remove non-alphanumeric characters (keep only letters and numbers)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()  
    
    tokens = word_tokenize(text)  # Tokenize the text
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords

    return " ".join(tokens)

def classify_readme(clean_text):
    """
    Classifies the already preprocessed README text.

    Parameters:
        clean_text (str): The preprocessed README text.

    Returns:
        str: "Research Software" or "Non-Research Software".
    """
    if not isinstance(clean_text, str):
        raise ValueError("❌ Error: Input text must be a string!")

    try:
        prediction = model.predict([clean_text])
        return "Research Software" if prediction[0] == "Research" else "Non-Research Software"
    except Exception as e:
        print(f"❌ Classification error: {e}")
        return "Error"

def classify_domain(clean_text):
    """
    Classifies the academic domain of research software and maps to full term.

    Parameters:
        clean_text (str): The preprocessed README text.

    Returns:
        str: The suggested academic domain (full term).
    """
    if not isinstance(clean_text, str):
        raise ValueError("❌ Error: Input text must be a string!")

    try:
        # 🔹 Transform text into vectorized format before classification
        vectorized_text = domain_vectorizer.transform([clean_text])  
        
        predicted_domain_abbr = domain_model.predict(vectorized_text)[0]
        
        # Map the abbreviation to the full domain term
        full_domain = DOMAIN_MAPPING.get(predicted_domain_abbr, "Unknown Domain")
        return full_domain
    except Exception as e:
        print(f"❌ Domain classification error: {e}")
        return "Error"

# Example usage
if __name__ == "__main__":
    print("Welcome to the Research Software Classifier Tool!")
    print("This tool allows you to classify an entire GitHub repository as research software or non-research software based on its README file.")
    print("If the repository is classified as research software, it will also suggest an academic domain classification.")
    print("\nTo use this tool, please provide the URL to the GitHub repository's README file.")

    readme_url = input("Please enter the GitHub repository README URL: ")
    
    readme_text, repo_name = fetch_readme_from_url(readme_url)

    if readme_text and repo_name:
        cleaned_text = preprocess_text_with_html_removal(readme_text)

        # Classify whether the repository is research or non-research
        software_classification = classify_readme(cleaned_text)
        print(f"\n✅ Repository '{repo_name}' is classified as {software_classification}.")
        
        # If classified as research software, suggest the academic domain
        if software_classification == "Research Software":
            domain_classification = classify_domain(cleaned_text)
            print(f"✅ Suggested Academic Domain for '{repo_name}': {domain_classification}")
            print("(This is a suggestion based on the classification model, not a definitive classification.)")
        else:
            print(f"⚠️ Repository '{repo_name}' is classified as Non-Research Software, so no domain classification will be applied.")

    else:
        print("❌ Error: Could not fetch README content.")

