import numpy as np
import pandas as pd
import re
from typing import Optional, List
import networkx as nx
import statsmodels.formula.api as smf
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import gensim
from gensim import corpora, models
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
    text_data = nltk.word_tokenize(text_data)

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
    assert (len(continuous_features) != 0) | (len(categorical_features) != 0), 'no feature passed to be matched on'
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

def score_per_day(df:pd.DataFrame, item: str, release_date: pd.Timestamp) -> pd.DataFrame:
    """
    Given a product name, fetches the df for videos with titles containing the name of the product
    that were uploaded within 6 months of the product's release date and computes the total view count,
    total likes total dislikes and number of videos per day.
    :param item: product name
    :param df: dataframe of videos with titles containing the name of the product
    :return: dataframe of total view count, total likes total dislikes and number of videos per day
    """

    # filter out titles that do not contain the product name
    df = df[df["title"].str.lower().str.contains(item)]
    
    # convert 'view_count' to int
    df["view_count"] = df["view_count"].astype(int)

    # convert 'upload_date' to datetime
    df["upload_date"] = pd.to_datetime(df["upload_date"])
    df["upload_date"] = df["upload_date"].dt.date

    # keep only rows that were uploaded within 3 months of the product's release date
    df = df[(df["upload_date"] >= release_date - datetime.timedelta(days=90)) & (df["upload_date"] <= release_date + datetime.timedelta(days=90))]

    # keep only 'view_count', 'upload_date', 'likes', 'dislikes', 'title'
    df = df[["view_count", "upload_date", "like_count", "dislike_count", "title"]]

    # group by 'upload_date'
    df = df.groupby("upload_date")

    # compute total view count, total like_count, total dislike_count and number of videos per day
    df = df.agg({"view_count": "sum", "like_count": "sum", "dislike_count": "sum", "title": "count"})

    return df

def get_sum_views(df:pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the sum of number of views per upload date from a DataFrame.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing view count and upload date data.

    Returns:
    - pandas.DataFrame: A DataFrame with the sum of view counts per upload date.

    The function extracts the 'view_count' and 'upload_date' columns from the input DataFrame,
    converts the 'upload_date' to datetime format, and drops rows with missing or invalid values.
    It then calculates the sum of view counts for each upload date and returns the result.
    """
    df_view = df[["view_count", "upload_date"]].copy()
    df_view["upload_date"] = pd.to_datetime(df_view["upload_date"])
    df_view = df_view.dropna(subset = ["view_count", "upload_date"])
    df_view["view_count"] = df_view["view_count"].astype("int64")

    # keep only 'view_count' and 'upload_date'
    df_view = df_view.groupby("upload_date")[['view_count']].sum()
    
    return df_view

def get_lda_topics(df:pd.DataFrame, num_topics:int, num_words:int) -> None:
    """
    Extracts topics using Latent Dirichlet Allocation (LDA) from a DataFrame of text data.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing text data.
    - num_topics (int): The number of topics to extract using LDA.
    - num_words (int): The number of words to display for each topic.

    Returns:
    - gensim.models.ldamodel.LdaModel: The trained LDA model.

    The function tokenizes, preprocesses, and lemmatizes the text data in the DataFrame.
    It then builds a dictionary and corpus for LDA modeling and trains the LDA model.
    Finally, it prints the top words for each topic.
    """
    # tokenize words
    tokenizer = RegexpTokenizer(r'\w+')
    df["tokens"] = df["title"].apply(tokenizer.tokenize)

    # convert to lowercase
    df["tokens"] = df["tokens"].apply(lambda x: [word.lower() for word in x])
    
    # remove stop words
    stop_words = stopwords.words('english')
    df["tokens"] = df["tokens"].apply(lambda x: [word for word in x if word not in stop_words])
    
    # lemmatize words
    lemmatizer = WordNetLemmatizer()
    df["tokens"] = df["tokens"].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    
    # remove words that appear only once
    all_tokens = sum(df["tokens"], [])
    tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
    df["tokens"] = df["tokens"].apply(lambda x: [word for word in x if word not in tokens_once])

    # remove the token 'iphone'
    df["tokens"] = df["tokens"].apply(lambda x: [word for word in x if word != "iphone"])

    # remove one character tokens
    df["tokens"] = df["tokens"].apply(lambda x: [word for word in x if len(word) > 2])
    
    # create dictionary and corpus
    dictionary = corpora.Dictionary(df["tokens"])
    corpus = [dictionary.doc2bow(text) for text in df["tokens"]]
    
    # create LDA model
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)
    
    # print topics
    topics = lda.show_topics(num_topics=3, num_words=10, formatted=False)
    for topic_id, words in topics:
        print(f"Topic {topic_id + 1}: {', '.join(word[0] for word in words)}")

    # visualize topics
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary)

def get_tf_idf_score(tf_idf_scores: dict, words: List[str]):
    """
    Given a dictionary of tf-idf scores and a list of words, compute the average tf-idf score for the list of words.
    :param tf_idf_scores: dictionary of tf-idf scores
    :param words: list of words
    :return: average tf-idf score for the list of words
    """
    score = 0
    for word in words:
        score += tf_idf_scores.get(word, 0)

    return score / len(words)

def tf_idf(tokens: List[str]):
    """
    Given a list of tokens, compute the tf-idf coefficient for each token.
    :param tokens: list of titles split into tokens
    :return: dictionary of tf-idf coefficients
    """
    # create dictionary
    dictionary = corpora.Dictionary(tokens)
    
    # create corpus
    corpus = [dictionary.doc2bow(text) for text in tokens]
    
    # create tf-idf model
    tfidf = models.TfidfModel(corpus)
    
    # get tf-idf coefficients
    tfidf_weights = {}
    for doc in corpus:
        for id, weight in tfidf[doc]:
            tfidf_weights[dictionary[id]] = weight
    
    return tfidf_weights