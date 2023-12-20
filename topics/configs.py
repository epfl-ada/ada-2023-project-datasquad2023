

configs = {
    'data_extraction': {
        'video_data_path': 'data/yt_tech_channels_metadata.tsv.gz',
        'channel_data_path': 'data/df_timeseries_en.tsv',
        'useless_cols': ["description", "crawl_date", "display_id", "tags"],
        'categories': ['Science & Technology', 'Education', 'Entertainment']
    }
}

def get_config(config_name):
    return configs[config_name]