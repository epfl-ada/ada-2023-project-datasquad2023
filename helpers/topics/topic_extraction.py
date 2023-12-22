import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# text preprocessing
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet
from gensim import corpora, models
import gensim

# topic extraction
import pyLDAvis



class Topics():
    def __init__(self) -> None:
        pass

    def extract_topics(self, df: pd.DataFrame, num_topics: int, num_words: int) -> pd.DataFrame:

        # create dictionary and corpus
        dictionary = corpora.Dictionary(df["tokens"])
        corpus = [dictionary.doc2bow(text) for text in df["tokens"]]
    
        # create LDA model
        lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)
        
        # print topics
        print("LDA Topics:")
        for topic in lda.print_topics(num_words=num_words):
            print(topic)
        
        # visualize topics
        pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
        return vis

    def preprocess_title_for_lda(self, df: pd.DataFrame, words_to_remove: list) -> pd.DataFrame:
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
        df["tokens"] = df["tokens"].apply(lambda x: [word for word in x if word not in words_to_remove])

        # remove one character tokens
        df["tokens"] = df["tokens"].apply(lambda x: [word for word in x if len(word) > 2])

        return df