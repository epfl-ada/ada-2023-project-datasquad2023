import datetime
import pandas as pd
from tqdm import tqdm

import json
import os
import re

class DataExtractor():
    def __init__(self, config) -> None:
        self.config = config
        self.video_data = pd.read_csv(config['video_data_path'], sep='\t')
        self.channel_data = pd.read_csv(config['channel_data_path'], sep='\t')

    def process_df(self, df):
        df['upload_date'] = pd.to_datetime(df['upload_date'])
        return df
    
    def products_data(self, products: list, releases: list) -> dict:
        """
        Returns a dictionary of dataframes for each product
        Where each dataframe only contains videos about that product
        :param products: list of product names
        :param releases: list of release dates
        :return: dictionary of dataframes
        """
        df = self.video_data.copy()
        products_df = {p: [] for p in products}
        for product_name, release_date in zip(products, releases):
            # change type of title to str
            df['title'] = df['title'].astype(str)
            item_df = df[df['title'].str.contains(product_name, case=False)]

            item_df["upload_date"] = pd.to_datetime(df["upload_date"])
            item_df = item_df[(item_df['upload_date'] >= datetime.datetime.strptime(release_date, '%d-%m-%Y') - datetime.timedelta(days=180)) & (item_df['upload_date'] <= datetime.datetime.strptime(release_date, '%d-%m-%Y') + datetime.timedelta(days=180))]

            if len(item_df) > 0:
                products_df[product_name].append(item_df)
            
        for product_name in products:
            products_df[product_name] = pd.concat(products_df[product_name])

        return products_df
    
    def ratio_views(self, product_df):
        """
        Returns a dictionary of the ratio of views of each product
        to the total views of all videos uploaded on that day
        :param product_df: dictionary of dataframes
        :return: dictionary of ratios
        """
        youtube_views = self.video_data[['upload_date', 'view_count']].groupby('upload_date').sum().reset_index()
        youtube_views = youtube_views.rename(columns={'view_count': 'youtube_views'})
        product_views = product_df[['upload_date', 'view_count']].groupby('upload_date').sum().reset_index()
        product_views = product_views.rename(columns={'view_count': 'product_views'})

        # convert upload_date to string
        youtube_views['upload_date'] = youtube_views['upload_date'].astype(str)
        product_views['upload_date'] = product_views['upload_date'].astype(str)

        product_views = product_views.merge(youtube_views, on='upload_date')
        product_views = product_views.dropna()

        # compute the ratio of product views to youtube views
        product_views['view_ratio'] = product_views['product_views']/product_views['youtube_views']

        # now reconvert upload_date to datetime
        product_views['upload_date'] = pd.to_datetime(product_views['upload_date'])

        product_views = product_views.set_index('upload_date').to_dict()['view_ratio']
        return product_views
    
    def get_topic_info(self, df: pd.DataFrame, start: str, end: str, topic: list = None) -> dict:
        """
        Returns a dictionary of the average views and subs of videos
        about a specific topic from start to end
        :param df: dataframe of videos
        :param topic: list of keywords
        :param start: start date in format dd-mm-yyyy
        :param end: end date in format dd-mm-yyyy
        :return: dictionary of average views and subs
        """
        # only keep videos with the topic in the title
        df = df.copy()

        if topic:
            # keep only videos where one of the pre-release keywords is present in the title
            df = df[df['title'].str.contains('|'.join(topic), case=False)]
        # convert datetime to datetime object
        df['upload_date'] = pd.to_datetime(df['upload_date'])
        # keep only rows that have datetime between 180 days before the release date and the release date
        df = df[(df['upload_date'] > datetime.datetime.strptime(start, '%Y-%m-%d')) & (df['upload_date'] < datetime.datetime.strptime(end, '%Y-%m-%d'))]

        # convert start and end to dates
        start = datetime.datetime.strptime(start, '%Y-%m-%d')
        end = datetime.datetime.strptime(end, '%Y-%m-%d')

        # get the date equivalent of the start and end dates
        start_date = start.date()
        end_date = end.date()

        product_channels = df['channel_id'].unique()
        channels_df = self.channel_data.copy()

        # keep only channels that published pre-release videos
        channels_df = channels_df[channels_df['channel'].isin(product_channels)]
        # convert datetime to datetime object and date format
        channels_df['datetime'] = pd.to_datetime(channels_df['datetime'])
        # keep only rows that have datetime between 180 days before the release date and the release date
        channels_df = channels_df[(channels_df['datetime'] > start) & (channels_df['datetime'] < end)]
        channels_df['datetime'] = channels_df['datetime'].dt.date
        channels_df = channels_df.groupby('datetime').sum()

        channels_df['ratio_views'] = channels_df['delta_views'] / channels_df['views']
        channels_df['ratio_subs'] = channels_df['delta_subs'] / channels_df['subs']

        # make a moving average of the ratio of views and subs
        channels_df['ratio_views'] = channels_df['ratio_views'].rolling(window=10).mean()
        channels_df['ratio_subs'] = channels_df['ratio_subs'].rolling(window=10).mean()

        channels_df = channels_df[['ratio_views', 'ratio_subs']]
        channels_df = channels_df.reset_index()
        channels_df['datetime'] = pd.to_datetime(channels_df['datetime'])
        channels_df = channels_df[['datetime', 'ratio_views', 'ratio_subs']]
        # turn into dictionary
        topic_info = channels_df.set_index('datetime').to_dict()

        return topic_info
    

    def get_channels_product(self, product_name, release_date):
        """
        Given a dataframe with columns: channel_id, title and upload_date, get
        the channels that did not upload any video about the product before the
        release date of the product
        :param product_name: name of product
        :param release_date: release date of product
        :return: list of channel ids
        """
        df = self.video_data.copy()
        df = df[['channel_id', 'title', 'upload_date']]
        df = df[df['title'].str.contains(product_name, case=False)]
        df['upload_date'] = pd.to_datetime(df['upload_date'])
        df = df[df['upload_date'] < datetime.datetime.strptime(release_date, '%d-%m-%Y')]
        df = df.drop_duplicates(subset=['channel_id'])

        # filter from video_data the channl_ids in df
        channels = self.video_data[~self.video_data['channel_id'].isin(df['channel_id'])]

        return channels
    
