#!/usr/bin/env python3
"""
Script to download necessary NLTK data for Limnos.
This ensures that all required NLTK resources are available for text processing.
"""

import os
import nltk
import sys

def setup_nltk_data():
    """Download necessary NLTK data packages."""
    print("Setting up NLTK data...")
    
    # Set NLTK data path to the project's nltk_data directory
    nltk_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nltk_data')
    os.environ['NLTK_DATA'] = nltk_data_dir
    print(f"NLTK data will be stored in: {nltk_data_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # List of required NLTK packages
    required_packages = [
        'punkt',           # Tokenizer models
        'stopwords',       # Common stopwords
        'wordnet',         # Lexical database
        'averaged_perceptron_tagger',  # POS tagger
        'vader_lexicon',   # Sentiment analysis
    ]
    
    # Download each package
    for package in required_packages:
        try:
            print(f"Downloading {package}...")
            nltk.download(package, download_dir=nltk_data_dir)
        except Exception as e:
            print(f"Error downloading {package}: {e}")
            continue
    
    print("NLTK data setup complete!")

if __name__ == "__main__":
    setup_nltk_data()
