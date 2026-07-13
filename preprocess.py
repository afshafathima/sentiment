import pandas as pd
import re
from bs4 import BeautifulSoup

# Load dataset
df = pd.read_csv("data/IMDB Dataset.csv")

# Function to clean text
def clean_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Apply cleaning
df["review"] = df["review"].apply(clean_text)

# Convert labels to numbers
df["label"] = df["sentiment"].map({
    "negative": 0,
    "positive": 1
})

# Save cleaned dataset
df.to_csv("data/cleaned_imdb.csv", index=False)

print(" Dataset cleaned successfully!")
print(df.head())