

configs = {
    'data_extraction': {
        'data_path': 'data/yt_tech_channels_metadata.tsv',
        'useless_cols': ["description", "crawl_date", "display_id", "tags"],
        'categories': ['Science & Technology', 'Education', 'Entertainment']
    }
}

def get_config(config_name):
    return configs[config_name]