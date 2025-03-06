import joblib
import re
import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

# Load stopwords
stop_words = set(stopwords.words("english"))

# Load ONLY the trained model (do NOT load vectorizer separately)
MODEL_FILE = "best_model.pkl"

try:
    model = joblib.load(MODEL_FILE)
    print("✅ Successfully loaded the model.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

def transform_github_readme_link(original_link):
    """
    Transforms a GitHub README URL to its raw format.
    """
    pattern = r'https://github.com/([^/]+)/([^/]+)/blob/([^/]+)/README.md'
    match = re.match(pattern, original_link)
    
    if match:
        owner, repo, branch = match.groups()
        raw_link = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/README.md"
        return raw_link
    else:
        raise ValueError("Invalid GitHub README link format. Please provide a valid URL.")

def fetch_readme_from_github(repo_url):
    """
    Fetches the README content from the GitHub repository.
    """
    try:
        raw_readme_url = transform_github_readme_link(repo_url)
        response = requests.get(raw_readme_url)

        if response.status_code == 200:
            return response.text  # Return the raw README content
        else:
            print(f"❌ Failed to fetch README. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error fetching README: {e}")
        return None

def preprocess_text(text):
    """
    Preprocess the text before classification:
    - Convert text to lowercase
    - Remove stopwords
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()  # Convert text to lowercase
    tokens = word_tokenize(text)  # Tokenize the text
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords

    return " ".join(tokens)  # Return as a string

def classify_readme(clean_text):
    """
    Classifies the already preprocessed README text.
    """
    if not isinstance(clean_text, str):
        raise ValueError("❌ Error: Input text must be a string!")

    # 🔹 If the model is a pipeline (contains TfidfVectorizer), call predict directly
    try:
        prediction = model.predict([clean_text])
        return "Research Software" if prediction[0] == "Research" else "Non-Research Software"

    except Exception as e:
        print(f"❌ Classification error: {e}")
        return "Error"

# Example usage
if __name__ == "__main__":
    readme_url = input("Please enter the GitHub repository README URL: ")
    
    # Step 1: Fetch README content from GitHub
    readme_text = fetch_readme_from_github(readme_url)

    if readme_text:
        # Step 2: Preprocess the README text separately
        cleaned_text = preprocess_text(readme_text)
        print("\n✅ Preprocessed README text:")
        print(cleaned_text[:500])  # Print first 500 characters for verification

        # Step 3: Classify using the trained model
        classification = classify_readme(cleaned_text)
        print(f"\n✅ Classification: {classification}")
    else:
        print("❌ Error: Could not fetch README content.")
