import pandas as pd
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plotly.figure_factory as ff
import seaborn as sns

import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import ttest_ind
from scipy import stats

from IPython.display import Image, display
from tqdm import tqdm
from typing import Optional, List
import datetime
import re

import pyLDAvis.gensim
import gensim
from gensim import corpora, models

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize

# nltk.download('stopwords')
# nltk.download('wordnet')
def ols(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Perform Ordinary Least Squares (OLS) regression analysis.

    Parameters:
    - X (pandas.DataFrame): The independent variable(s) matrix.
    - y (pandas.Series): The dependent variable vector.

    Returns:
    None -> prints the summary statistics.
    """
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print_model = model.summary()
    print(print_model)

def get_correlations(df_vd_tech):
    '''
    df_vd_tech: dataframe to use for computing the correlations
    
    returns: pearson correlation between duration and # of likes as well as duration and # of dislikes
    '''
    # filter only the tech review videos by tech review channels
    df_vd_tech["upload_date"] = pd.to_datetime(df_vd_tech["upload_date"])
    df_vd_tech["upload_year"] = df_vd_tech['upload_date'].dt.year

    # correlation between the duration of the video and number of likes and dislikes
    df_vd_tech_filtered = df_vd_tech.copy()
    # just to avoid division by 0
    df_vd_tech_filtered["view_count"] = df_vd_tech_filtered["view_count"].replace({0:1})
    #correlation between like count and duration
    df_vd_tech_filtered["like_count_ratio"] = df_vd_tech_filtered['like_count']/df_vd_tech_filtered["view_count"]
    df_vd_tech_filtered["dislike_count_ratio"] = df_vd_tech_filtered['dislike_count']/df_vd_tech_filtered["view_count"]
    pearson_like_duration_corrs = []
    pearson_dislike_duration_corrs = []

    years = np.unique(df_vd_tech_filtered["upload_year"])
    for year in years:
        #compute pearson correlation
        df = df_vd_tech_filtered[df_vd_tech_filtered["upload_year"]==year].dropna()
        corr_like = stats.pearsonr(df["duration"], df['like_count_ratio'])
        corr_dislike = stats.pearsonr(df["duration"], df['dislike_count_ratio'])
        pearson_like_duration_corrs.append(corr_like)
        pearson_dislike_duration_corrs.append(corr_dislike)

    df_corrs_pearson_like = pd.DataFrame(pearson_like_duration_corrs, columns=["pearson_corr", "p_value"], index=years)
    df_corrs_pearson_dislike = pd.DataFrame(pearson_dislike_duration_corrs, columns=["pearson_corr", "p_value"], index=years) 
    
    return df_corrs_pearson_like, df_corrs_pearson_dislike

def plot_correlations(df_corrs_pearson_like, df_corrs_pearson_dislike):
    '''
    df_corrs_pearson_like: dataframe containing pearson correltions with their p-values between likes and duration per year
    df_corrs_pearson_dislike: dataframe containing pearson correltions with their p-values between dislikes and duration per year

    plots the pearson correlations as a bar plot
    '''
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    colors = sns.color_palette("colorblind",2)

    # Create a bar plot for Pearson correlation coefficients
    bar_like = axs[0].bar(df_corrs_pearson_like.index, df_corrs_pearson_like["pearson_corr"], color=colors[0], alpha=0.7)
    bar_dislike = axs[1].bar(df_corrs_pearson_dislike.index, df_corrs_pearson_dislike["pearson_corr"], color=colors[0], alpha=0.7)

    # Highlight bars where p-values are less than 0.05
    for bar, p_value in zip(bar_like, df_corrs_pearson_like["p_value"]):
        if p_value < 0.05:
            bar.set_color(colors[1])
    for bar, p_value in zip(bar_dislike, df_corrs_pearson_dislike["p_value"]):
        if p_value < 0.05:
            bar.set_color(colors[1])

    # Set labels and title
    axs[0].set_title('# Likes vs Duration')
    axs[1].set_title('# Dislikes vs Duration')

    # Display a legend indicating significance
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], markersize=10, label='p < 0.05')],
               loc='upper right')
    plt.suptitle('Pearson Correlation Coefficients per Year')
    plt.tight_layout()
    plt.show()

def compute_ratios_and_print_duration(df_vd_tech):
    '''
    computes the like to dislike ratio and prints the duration
    
    df_vd_tech: dataframe to use for the statistics computation
    
    retuns the dataframe containing the ratio
    '''
    
    # get the likes to dislikes ratio
    df_vd_tech_ratio = df_vd_tech[['channel_id', 'upload_year','like_count','dislike_count','display_id','duration']]
    df_vd_tech_ratio['dislike_count'] = df_vd_tech_ratio['dislike_count'].replace({0:1})
    df_vd_tech_ratio['dislike_count'].fillna(1,inplace=True)
    df_vd_tech_ratio['like_dislike_ratio'] = df_vd_tech_ratio['like_count'] / df_vd_tech_ratio['dislike_count']
    df_vd_tech_ratio.like_dislike_ratio.fillna(0,inplace=True)

    #group videos per channel and compute the average duration and like to dislike ratio (per channel). 
    #Sort by the latter and get the top 100 and bottom 100 like to dislike ratios and compute the average of 
    #their duration averages (macro average). Note that by computing average per channel we give every channel the same weight.
    df_vd_tech_sorted = df_vd_tech_ratio.groupby("channel_id")[["duration", "like_dislike_ratio"]].mean().sort_values(by='like_dislike_ratio', ascending=False)
    print("The average duration of a video from the top 10% of channels (in terms of like to dislike ratio) is {:.0f} seconds and the average duration of a video from the bottom 10% of channels is {:.0f}.".format(\
    df_vd_tech_ratio.iloc[0:(int)(0.1*len(df_vd_tech_ratio))].duration.mean(), df_vd_tech_ratio.iloc[(int)(0.9*len(df_vd_tech_ratio)):].duration.mean()))
    print(stats.ttest_ind(df_vd_tech_ratio.iloc[0:(int)(0.1*len(df_vd_tech_ratio))].duration, df_vd_tech_ratio.iloc[(int)(0.9*len(df_vd_tech_ratio)):].duration))

    return df_vd_tech_ratio

def plot_helper(df_vd_tech):
    '''
    helper function to plot various statistics
    
    df_vd_tech: dataframe to use for the statistics computation
    '''
    # just to avoid division by 0
    df_vd_tech['dislike_count'] = df_vd_tech['dislike_count'].replace({0:1})

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 6), sharex=True)
    colors = sns.color_palette("colorblind",1)

    sns.pointplot(x=df_vd_tech['upload_year'],
                  y=df_vd_tech['like_count'],
                  errwidth=1.5,
                  color=colors[0],
                  ax=axs[0][0])

    sns.pointplot(x=df_vd_tech['upload_year'],
                  y=df_vd_tech['dislike_count'],
                  errwidth=1.5,
                  color=colors[0],
                  ax=axs[0][1])
    
    sns.pointplot(x=df_vd_tech['upload_year'],
                  y=df_vd_tech['like_count'] / df_vd_tech['dislike_count'],
                  errwidth=1.5,
                  color=colors[0],
                  ax=axs[1][0])

    sns.pointplot(x=df_vd_tech['upload_year'],
                  y=df_vd_tech['duration'],
                  errwidth=1.5,
                  color=colors[0],
                  ax=axs[1][1])

    axs[0][0].set(title='Average number of likes per year', xlabel=None)
    axs[0][0].grid(axis='y', linestyle='--')
    axs[0][0].tick_params(axis='x', rotation=45)
    axs[0][0].set_xlabel('years')
    
    axs[0][1].set(title='Average number of dislikes per year', xlabel=None)
    axs[0][1].grid(axis='y', linestyle='--')
    axs[0][1].tick_params(axis='x', rotation=45)
    axs[0][1].set_xlabel('years')
    
    axs[1][0].set(title='Average ratio of like to dislike per video per year', xlabel=None)
    axs[1][0].grid(axis='y', linestyle='--')
    axs[1][0].tick_params(axis='x', rotation=45)
    axs[1][0].set_xlabel('years')
    axs[1][0].set_ylabel("like-dislike ratio")
    
    axs[1][1].set(title='Average video duration per year', xlabel=None)
    axs[1][1].grid(axis='y', linestyle='--')
    axs[1][1].tick_params(axis='x', rotation=45)
    axs[1][1].set_xlabel('years')

    plt.tight_layout()
    plt.show()

def plot_moving_avg(df_vd_tech_ratio, window, short):
    '''
    df_vd_tech_ratio: dataframe to be used to compute the moving avg
    window: window of the moving average
    short: either False or True for either long or short videos respectively
    '''
    # define color
    colors = sns.color_palette("colorblind",1)

    ## sort videos by duration
    df_vd_tech_ratio.sort_values(by='duration',inplace=True)

    # plot moving average of likes to dislikes ratio versus duration
    plt.figure(figsize=(6,3))
    y = df_vd_tech_ratio["like_dislike_ratio"].rolling(window).mean().values
    x = df_vd_tech_ratio['duration']
    #if short videos only plot from 0 to 20 minutes else from 20 minutes to the longest duration
    if(short):
        plt.xlim(0,max(df_vd_tech_ratio['duration']))
    else:
        plt.xlim(1200, max(df_vd_tech_ratio['duration']))
        plt.xscale('log')
        plt.xticks([1200, 10000, 100000])
    ax = sns.lineplot(x=x, y=y, color=colors[0])
    ax.set(title='Moving average of likes-to-dislikes ratio over the duration of video', xlabel='Duration (seconds)', ylabel='#likes / #dislikes')
    plt.show()

def plot_statistics_across_duration_intervals(df_vd_tech_ratio, threshold):
    '''
    df_vd_tech_ratio: dataframe to use to compute the statistics
    threshold: int taking 2 possible values: if 2004 take all videos and if 2010 take videos from 2011 to 2019
    '''
    df_vd_tech_ratio['duration_intervals'] = pd.cut(df_vd_tech_ratio['duration'], bins=10)

    fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(12, 12))
    colors = sns.color_palette("colorblind",4)
    fig.tight_layout()
    fig.suptitle("mean of like_dislike_ratio, like_count, dislike_count and # of videos for each duration interval for videos uploaded after 2010", y=1)

    number_of_videos_per_interval = df_vd_tech_ratio[df_vd_tech_ratio['upload_year']>threshold].groupby('duration_intervals', as_index=False).apply(lambda x: len(x)).rename(columns={None: 'number_of_videos'})
    sns.barplot(x='duration_intervals', y='like_dislike_ratio', data=df_vd_tech_ratio[df_vd_tech_ratio['upload_year']>threshold], ax=axs[0], estimator ='mean', color=colors[0])
    sns.barplot(x='duration_intervals', y='like_count', data=df_vd_tech_ratio[df_vd_tech_ratio['upload_year']>threshold], ax=axs[1], color=colors[1])
    sns.barplot(x='duration_intervals', y='dislike_count', data=df_vd_tech_ratio[df_vd_tech_ratio['upload_year']>threshold], ax=axs[2], color=colors[2])
    sns.barplot(x='duration_intervals', y='number_of_videos', data=number_of_videos_per_interval, ax=axs[3], color=colors[3])

    #plot the like and dislike count on the same log scale
    axs[0].set(xlabel=None)

    axs[1].set(xlabel=None)
    axs[1].set_yscale('log')

    axs[2].set(xlabel=None)
    axs[2].set_yscale('log')

    axs[3].set(xlabel="Duration intervals")
    axs[3].tick_params(axis='x', rotation=20)

    plt.tight_layout()
    plt.show()

def find_in_name_and_tags(df: pd.DataFrame, item: str, release_date: pd.Timestamp) -> pd.DataFrame:
    """
    Filter a DataFrame based on a specified product in the title or tags column,
    within a three-month window around a given release date.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing video information.
    - item (str): The product to search for in the title or tags.
    - release_date (pd.Timestamp): The release date around which to filter the DataFrame.

    Returns:
    - pandas.DataFrame: A filtered DataFrame with an additional column indicating the presence
      (1) or absence (0) of the specified product in the title or tags.
    """
    df = df[df.title.notna()]
    df["upload_date"] = pd.to_datetime(df["upload_date"])

    df.loc[
        ((df["title"].str.lower().str.contains(item)) | (df["tags"].str.lower().str.contains(item, na=False))) &
        (df["upload_date"] > release_date - pd.DateOffset(months=3)) & 
        (df["upload_date"] < release_date + pd.DateOffset(months=3)),
        item] = 1 

    df[item].fillna(0,inplace=True)

    return df

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

def get_lda_topics(df:pd.DataFrame, num_topics:int, num_words:int):
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
    return vis