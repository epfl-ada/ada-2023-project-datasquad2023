import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# nltk.download('stopwords')
# nltk.download('wordnet')

def clean_text(text_data: str) -> list:
    """
    Clean and preprocess text data.

    Parameters:
    - text_data (str): Input text data.

    Returns:
    - text_data_clean (list): List of cleaned and preprocessed words.
    """
    # Convert to lowercase
    text_data = text_data.lower()

    # Remove all non-English-letter characters
    text_data = re.sub(r'[^a-z]', ' ', text_data)

    # Create a list of words
    text_data = text_data.split()

    # Lemmatize the words
    wl = WordNetLemmatizer()
    text_data_lem = [wl.lemmatize(word) for word in text_data if not word in set(stopwords.words('english'))]

    # Remove single letters
    text_data_clean = [word for word in text_data_lem if len(word)>1]

    return text_data_clean

def get_common_words(words: list) -> pd.DataFrame:
    """
    Get the most common words and their occurrences from a list of words.

    Parameters:
    - words (list): A list of words.

    Returns:
    - words_common (DataFrame): A DataFrame containing the top 100 most common words and their occurrences.
    """
    # Count the occurrences of each word and create a DataFrame
    words_unique = np.unique(words,return_counts=True)
    words_unique = pd.DataFrame({'word':words_unique[0],'occurance':words_unique[1]})
    
    # Sort the DataFrame by occurrence in descending order and select the top 100
    words_common = words_unique.sort_values(by='occurance',ascending=False).iloc[:100]
    
    return words_common