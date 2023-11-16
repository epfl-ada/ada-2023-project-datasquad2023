import numpy as np
import pandas as pd
import re
from typing import Optional, List
import networkx as nx
import statsmodels.formula.api as smf
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
    Get the most 100 common words and their occurrences from a list of words.

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

def classify_product(title: str, keywords: dict) -> Optional[str]:
    """
    Classifies a product based on the presence of keywords in its title.

    Parameters:
    - title (str): The title of the product to be classified.
    - keywords (dict): A dictionary where keys represent product categories, and values are lists of keywords associated with each category.

    Returns:
    - str or np.nan: The category of the product with the highest keyword match. Returns np.nan if no keywords are found.
    """
    
    count = {}
    for keyword in keywords:
        count[keyword] = sum([word in title for word in keywords[keyword]])
    if not all(value == 0 for value in count.values()):
        return max(count, key=count.get)
    else:
        return np.nan
    
def balance_data(df: pd.DataFrame,
                 treat_column: str,
                 continuous_features: List[str] = [],
                 categorical_features: List[str] = []) -> List[int]:
    """
    Balances a dataset based on propensity scores for treatment and control groups.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the original data.
    - treat_column (str): The column indicating treatment assignment (1 for treatment, 0 for control).
    - continuous_features (List[str]): List of names of continuous features to be standardized.
    - categorical_features (List[str]): List of names of categorical features for logistic regression.

    Returns:
    - List[int]: List of indices of matched instances in the original DataFrame.
    """
    
    # Copy the df to avoid modifying the original dataframe
    data = df[[treat_column] + continuous_features + categorical_features]
    
    # Standardize the continuous features
    for column in continuous_features:
        data[column] = (data[column] - data[column].mean())/data[column].std()

    continuous_formula = ' + '.join(continuous_features)
    categorical_formula = ' + '.join([f'C({col})' for col in categorical_features])
    
    if (len(categorical_formula) != 0) & (len(continuous_formula) != 0):
        formula = f"{treat_column} ~ {continuous_formula} + {categorical_formula}"
    elif len(continuous_formula) != 0:
        formula = f"{treat_column} ~ {continuous_formula}"
    else:
        formula = f"{treat_column} ~ {categorical_formula}"

    mod = smf.logit(formula=formula, data=data)

    res = mod.fit()

    # Extract the estimated propensity scores
    data['Propensity_score'] = res.predict()

    # Calculate similarity
    def get_similarity(propensity_score1, propensity_score2):
        '''Calculate similarity for instances with given propensity scores'''
        return 1-np.abs(propensity_score1-propensity_score2)

    # Balance the dataset 
    treatment_df = data[data[treat_column] == 1]
    control_df = data[data[treat_column] == 0]

    # Create Graph
    G = nx.Graph()

    for control_id, control_row in control_df.iterrows():
        for treatment_id, treatment_row in treatment_df.iterrows():

            similarity = get_similarity(control_row['Propensity_score'], treatment_row['Propensity_score'])
            G.add_weighted_edges_from([(control_id, treatment_id, similarity)])

    matching = nx.max_weight_matching(G)
    matched = [i[0] for i in list(matching)] + [i[1] for i in list(matching)]
    
    return matched