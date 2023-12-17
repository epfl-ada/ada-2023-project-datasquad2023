import datetime
import pandas as pd
from tqdm import tqdm

import json
import os
import re

class DataExtractor():
    def __init__(self, config) -> None:
        self.config = config
        self.video_data = pd.read_csv(config['video_data_path'])
        self.channel_data = pd.read_csv(config['channel_data_path'])

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
        product_views = product_df[['upload_date', 'view_count']].groupby('upload_date').sum().reset_index()
        product_views['view_count'] = product_views['view_count'] / youtube_views['view_count']

        product_views = product_views.set_index('upload_date').to_dict()['view_count']
        return product_views
    
    

    
    
    # once i get the channels that talked about a specific topic
    # I need to split them into big, medium and small channels
    # Then I need to keep only videos that were uploaded past the specific date
    # Then I take the average of delta_views and delta_subs from that date upwards
    # Then I take a moving average of that and plot it against time

