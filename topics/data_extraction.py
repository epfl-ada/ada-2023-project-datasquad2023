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
        df = df[['title', 'channel_id', 'upload_date']]
        df['title'] = df['title'].astype(str)
        if topic:
            df = df[df['title'].str.contains('|'.join(topic), case=False)]

        # if there is a column called channel, rename it to channel_id
        if 'channel' in self.channel_data.columns:
            self.channel_data = self.channel_data.rename(columns={'channel': 'channel_id'})
        df = pd.merge(self.channel_data, df, how='inner', on='channel_id')
        df = df[df['channel_id'].notnull()]

        # keep only rows where upload date is between start and end
        df['upload_date'] = pd.to_datetime(df['upload_date'])
        df = df[(df['upload_date'] >= datetime.datetime.strptime(start, '%d-%m-%Y')) & (df['upload_date'] <= datetime.datetime.strptime(end, '%d-%m-%Y'))]



        # remove rows where 'upload_date' < 'datetime'
        df = df[df['upload_date'] >= df['datetime']]
        df = df[['datetime', 'delta_views', 'delta_subs', 'views', 'subs']]

        # convert datetime to date
        df['datetime'] = pd.to_datetime(df['datetime']).dt.date
        # group by datetime and channel_id and take the sum of all the columns

        # remove rows having a nan in any of the columns
        df = df.dropna()

        # take ratio of delta_views to views
        df['ratio_views'] = df['delta_views'] / df['views']

        # take ratio of delta_subs to subs
        df['ratio_subs'] = df['delta_subs'] / df['subs']

        # group by datetime and take the average of delta_views and delta_subs
        df = df.groupby('datetime').mean().reset_index()

        # take a moving average of delta_views and delta_subs and subs
        df['ratio_views'] = df['ratio_views'].rolling(window=10).mean()
        df['ratio_subs'] = df['ratio_subs'].rolling(window=10).mean()
        df['subs'] = df['subs'].rolling(window=10).mean()

        # convert df to dictionary
        df = df.set_index('datetime').to_dict()

        return df
    

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
        df['title'] = df['title'].astype(str)
        df = df[df['title'].str.contains(product_name, case=False)]
        df['upload_date'] = pd.to_datetime(df['upload_date'])
        df = df[df['upload_date'] < datetime.datetime.strptime(release_date, '%d-%m-%Y')]
        df = df.drop_duplicates(subset=['channel_id'])

        # filter from video_data the channl_ids in df
        channels = self.video_data[~self.video_data['channel_id'].isin(df['channel_id'])]

        return channels
    

    
    

    
    
    # once i get the channels that talked about a specific topic
    # I need to split them into big, medium and small channels
    # Then I need to keep only videos that were uploaded past the specific date
    # Then I take the average of delta_views and delta_subs from that date upwards
    # Then I take a moving average of that and plot it against time

