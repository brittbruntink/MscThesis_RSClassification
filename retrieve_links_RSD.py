#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: retrieve_links_RSD.py
Description: This script retrieves repository metadata from the Research Software Directory (RSD) API, infers the corresponding README URLs,
             and merges manual README links into the dataset. It also identifies repositories that still need manual review, sorting the manual README file 
             so that repos without links are at the bottom.

Author: B.M. Bruntink
Date: 2025-01-23

Dependencies:
    - requests: A library for making HTTP requests to the RSD API.
    - pandas: A library for data manipulation and analysis, used to handle and save the dataset.

Usage:
    python retrieve_links_RSD.py
    This will fetch data from the RSD API, process the repositories to infer README links, and generate a CSV file with the repository details.

"""
import requests
import pandas as pd
import re

# Research Software Directory API URL
RSD_API_URL = "https://research-software-directory.org/api/v1/software"

# Expanded list of README file names, including .rst and others
README_FILE_NAMES = ['README.md', 'readme.md', 'README.rst', 'readme.rst', 'README.txt', 'README', 'readme', 'README.MD', 'README.markdown']

# Function to check if a URL is a GitHub, GitLab, or GitHub Pages link
def is_repository_link(url):
    """Check if the provided URL is a GitHub, GitLab, or GitHub Pages link."""
    if not isinstance(url, str):
        return False
    repo_pattern = r'^(https?://)?(www\.)?(github\.com|gitlab\.com|.*\.github\.io)/.*$'
    return bool(re.match(repo_pattern, url))

# Function to fetch repository data from RSD API
def fetch_rsd_data():
    """Fetch data from the RSD API without any filters."""
    try:
        response = requests.get(RSD_API_URL)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to retrieve data. Status: {response.status_code}, Message: {response.text}")
            return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# Function to infer GitHub repository from github.io URL
def infer_github_repo_from_io(url):
    """Infer the GitHub repository URL from a github.io URL."""
    match = re.match(r'^(https?://)?(www\.)?(?P<subdomain>.*)\.github\.io/(?P<repo>[^/]+)', url)
    if match:
        subdomain = match.group('subdomain')
        repo = match.group('repo')
        return f'https://github.com/{subdomain}/{repo}'
    return None

# Function to construct the raw README URL
def construct_raw_readme_url(url):
    """Construct the raw README URL for the given repository URL."""
    inferred_repo = infer_github_repo_from_io(url) if 'github.io' in url else url
    if inferred_repo:
        cleaned_url = inferred_repo.split('#')[0].split('?')[0]
        match = re.match(r'^(https?://)?(www\.)?(github\.com|gitlab\.com)/([^/]+)/([^/]+)', cleaned_url)
        if match:
            platform = match.group(3)
            owner = match.group(4)
            repo = match.group(5)
            base_url = "raw.githubusercontent.com" if "github" in platform else "gitlab.com"
            
            for readme_file in README_FILE_NAMES:
                for branch in ['main', 'master', 'develop', 'gh-pages']:
                    raw_readme_url = f'https://{base_url}/{owner}/{repo}/{branch}/{readme_file}'
                    if is_readme_url_valid(raw_readme_url):
                        return raw_readme_url
    return "Not Found"

# Function to check if the raw README URL is valid
def is_readme_url_valid(readme_url):
    """Check if the raw README URL is valid."""
    try:
        response = requests.head(readme_url, allow_redirects=True)
        return response.status_code == 200
    except Exception:
        return False

# Function to check and merge manual readme file while preserving valid manual links
def check_and_merge_manual_readme_links(dataset, manual_file_path='manual_readme_file.csv'):
    """Check and merge manual README file while preserving valid manual links."""
    try:
        manual_df = pd.read_csv(manual_file_path)
    except FileNotFoundError:
        print(f"File '{manual_file_path}' not found. Please provide manual README links.")
        return dataset

    if 'Brand Name' not in manual_df.columns or 'Get Started URL' not in manual_df.columns or 'README Link' not in manual_df.columns:
        print("Manual README file does not have the required columns: 'Brand Name', 'Get Started URL', or 'README Link'.")
        return dataset

    merged_list = []
    for item in dataset:
        match = manual_df[(manual_df['Brand Name'] == item['Brand Name']) & (manual_df['Get Started URL'] == item['Get Started URL'])]

        # Preserve existing valid links and only update if "Not Found"
        if item['README Link'] == "Not Found" and not match.empty:
            manual_readme = match.iloc[0]['README Link']
            if manual_readme != "Not Found" and isinstance(manual_readme, str) and manual_readme.strip():
                item['README Link'] = manual_readme

        merged_list.append(item)

    return merged_list

# Main function to run the entire process
def main():
    """Run the process of retrieving repository metadata, inferring README URLs, and merging manual links."""
    rsd_data = fetch_rsd_data()
    if not rsd_data:
        print("No data retrieved from RSD API.")
        return

    dataset = []
    for repo in rsd_data:
        brand_name = repo.get('brand_name', None)
        get_started_url = repo.get('get_started_url', None)
        readme_link = construct_raw_readme_url(get_started_url) if is_repository_link(get_started_url) else "Not Found"
        dataset.append({"Brand Name": brand_name, "Get Started URL": get_started_url, "README Link": readme_link})

    dataset = check_and_merge_manual_readme_links(dataset)
    df = pd.DataFrame(dataset)
    df.sort_values(by=['README Link'], ascending=False, inplace=True)
    df.to_csv('repository_readme_links.csv', index=False, encoding="utf-8")

    print(f"Processed {len(df)} repositories.")
    print("Please manually review the repositories at the bottom with missing README links.")

if __name__ == "__main__":
    main()
