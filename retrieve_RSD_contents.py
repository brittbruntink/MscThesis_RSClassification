# -*- coding: utf-8 -*-
"""
Script Name: retrieve_NRS_contents.py
Description: This script retrieves and cleans README contents from GitHub and GitLab repositories
             by fetching the raw content from the provided URLs, cleaning it (removing HTML tags,
             and normalizing whitespace), and filtering based on a minimum word count. The script
             processes a dataset of repositories, reports statistics on successfully fetched
             README contents, and saves the cleaned content to an output CSV file.

Author: B.M. Bruntink
Date: 2025-02-20

Dependencies:
    - requests: A library for making HTTP requests to GitHub and GitLab to fetch README contents.
    - pandas: A library for handling and saving datasets in CSV format.
    - nltk: A library used to download the necessary NLTK resources for text processing.
    - BeautifulSoup (from bs4): A library used to parse and clean HTML content from README files.

Usage:
    python retrieve_NRS_contents.py
    This will process the 'repository_readme_links.csv' input file, clean and filter the README contents, 
    and save the results to 'rs_contents.csv'.
"""

import os
import requests
import pandas as pd
import random
import re
import time
from bs4 import BeautifulSoup
import nltk

# Download necessary NLTK resources
nltk.download('punkt')

# Minimum word threshold for README content
MIN_WORD_COUNT = 50  

# Retrieve the GitHub token from the environment variable
GITHUB_ACCESS_TOKEN = os.getenv('GITHUB_ACCESS_TOKEN')

# Check if the token was retrieved successfully
if GITHUB_ACCESS_TOKEN is None:
    print("❌ Error: GitHub access token is missing. Please set the 'GITHUB_ACCESS_TOKEN' environment variable.")
    exit()

# GitHub API configuration
GITHUB_API_URL = "https://api.github.com"
HEADERS = {
    "Authorization": f"Bearer {GITHUB_ACCESS_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Load the RS dataset and determine target sample size
RS_CSV_FILE = "rs_contents.csv"
try:
    rs_df = pd.read_csv(RS_CSV_FILE)
    valid_rs_count = rs_df[rs_df['Cleaned README Content'].notnull() & (rs_df['Cleaned README Content'].str.strip() != '')].shape[0]

    TARGET_SAMPLE_SIZE = valid_rs_count  # Ensure exact balance with RS dataset
    if TARGET_SAMPLE_SIZE == 0:
        print(f"❌ Error: No valid README content found in {RS_CSV_FILE}. Exiting.")
        exit()
    
    print(f"✅ Target sample size set to {TARGET_SAMPLE_SIZE} based on successful README retrievals in RS dataset.")
except FileNotFoundError:
    print(f"❌ Error: {RS_CSV_FILE} not found. Please check the file path.")
    exit()
except KeyError:
    print("❌ Error: Column 'Cleaned README Content' not found in the CSV file. Please check the column names.")
    exit()

def get_random_repositories(batch_size=100):
    """
    Fetch random repositories from GitHub API.

    Args:
        batch_size (int): The number of repositories to fetch in one batch.

    Returns:
        list: A list of dictionaries containing 'Brand Name' and 'Get Started URL' for each repository.
    """
    collected_repos = set()  # Use a set to prevent duplicates
    while len(collected_repos) < batch_size:
        since_id = random.randint(1, 10000000)
        response = requests.get(f"{GITHUB_API_URL}/repositories", headers=HEADERS, params={"since": since_id})
        if response.status_code == 200:
            repositories = response.json()
            for repo in repositories:
                repo_name = repo.get("name")
                repo_url = repo.get("html_url")
                if repo_url not in collected_repos:  # Ensure unique URLs
                    collected_repos.add(repo_url)
        else:
            print(f"❌ Error fetching repositories: {response.status_code}")
        time.sleep(1)  # Avoid API rate limits

    return [{"Brand Name": url.split('/')[-1], "Get Started URL": url} for url in collected_repos]

def fetch_readme(repo_url):
    """
    Fetch the raw README content from a GitHub repository.

    Args:
        repo_url (str): The URL of the repository.

    Returns:
        tuple: A tuple containing the raw URL and content of the README file, or (None, None) if an error occurs.
    """
    try:
        owner, repo = repo_url.rstrip('/').split('/')[-2:]
        readme_url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/readme"
        response = requests.get(readme_url, headers=HEADERS)

        if response.status_code == 200:
            readme_data = response.json()
            raw_url = readme_data.get("download_url")
            if raw_url:
                raw_response = requests.get(raw_url, timeout=10)
                if raw_response.status_code == 200:
                    content = raw_response.text.strip()
                    if not content or "404: Not Found" in content:
                        return None, None
                    return raw_url, content
        return None, None
    except Exception as e:
        print(f"❌ Error fetching README for {repo_url}: {e}")
        return None, None

def clean_readme_content(content):
    """
    Clean the README content by removing HTML and normalizing whitespace.

    Args:
        content (str): The raw content of the README file.

    Returns:
        str: The cleaned content with HTML tags removed and whitespace normalized.
    """
    try:
        if not content:
            return None
        soup = BeautifulSoup(content, "html.parser")
        text = soup.get_text()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        print(f"❌ Error cleaning content: {e}")
        return None

def process_readmes(collected_data):
    """
    Process and clean the README files for collected repositories.

    Args:
        collected_data (list): A list of repositories with 'Brand Name' and 'Get Started URL'.

    Returns:
        list: A list of repositories with cleaned README content.
    """
    valid_repos = []
    seen_readmes = set()  # Store already seen README content to prevent duplicates

    while len(valid_repos) < TARGET_SAMPLE_SIZE:
        for repo in collected_data[:]:  # Use slice to avoid modifying list during iteration
            readme_link, raw_readme = fetch_readme(repo["Get Started URL"])
            if raw_readme:
                cleaned_content = clean_readme_content(raw_readme)

                # Check for duplicate README content and ensure minimum word count
                if cleaned_content and len(cleaned_content.split()) >= MIN_WORD_COUNT and cleaned_content not in seen_readmes:
                    repo["README Link"] = readme_link
                    repo["Raw README Content"] = raw_readme
                    repo["Cleaned README Content"] = cleaned_content
                    valid_repos.append(repo)
                    seen_readmes.add(cleaned_content)  # Track this README to avoid duplicates
                    print(f"✅ Processed README for {repo['Brand Name']} (Total: {len(valid_repos)}/{TARGET_SAMPLE_SIZE})")

            collected_data.remove(repo)

            if len(valid_repos) >= TARGET_SAMPLE_SIZE:
                break

        # **Ensuring exact dataset parity**: Keep fetching more repositories if needed
        if len(valid_repos) < TARGET_SAMPLE_SIZE:
            print(f"⚠️ Retrieved {len(valid_repos)} NRS repositories. Fetching more...")
            collected_data.extend(get_random_repositories(batch_size=50))  # Fetch additional repositories

    return valid_repos[:TARGET_SAMPLE_SIZE]  # Ensure exact match

def main():
    """
    Main function to collect, process, and save non-research software repository data.
    """
    output_file = "random_non_research_software_contents.csv"
    print("📂 Starting collection of random non-research software repositories...")

    repositories = get_random_repositories(batch_size=200)  # Start with a large batch
    processed_data = process_readmes(repositories)
    df = pd.DataFrame(processed_data)

    # **Remove repositories where Cleaned README Content is missing or empty**
    df = df.dropna(subset=["Cleaned README Content"])
    df = df[df["Cleaned README Content"].str.strip() != ""]

    # Ensure dataset size matches research dataset
    if len(df) != TARGET_SAMPLE_SIZE:
        print(f"⚠️ Dataset mismatch! Expected {TARGET_SAMPLE_SIZE}, but got {len(df)}. Fetching more to correct...")
        additional_repos = get_random_repositories(batch_size=50)
        more_processed_data = process_readmes(additional_repos)
        df = pd.concat([df, pd.DataFrame(more_processed_data)]).head(TARGET_SAMPLE_SIZE)

    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"✅ Successfully saved non-research software data to '{output_file}'.")
    print(f"✅ Final dataset size: {len(df)} repositories (after ensuring exact balance).")

if __name__ == "__main__":
    main()

# Script Name: retrieve_NRS_contents.py
# Date: 2025-02-20
# Author: B.M. Bruntink
