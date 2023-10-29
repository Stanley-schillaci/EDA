import numpy as np
import pandas as pd

from src.preprocessing import preprocess_data, preprocess_metadata, processing, save_files


def load_reviews_csv(review_path: str) -> pd.DataFrame:
    df = pd.read_csv(review_path, sep=',', dtype={'overall': np.float16,
                                                            'verified': np.bool8,
                                                            'reviewerID': pd.StringDtype(),
                                                            'asin': pd.StringDtype(),
                                                            'reviewText': pd.StringDtype(),
                                                            'summary': pd.StringDtype(),
                                                            'unixReviewTime': pd.StringDtype(),
                                                            'vote': np.float32}, low_memory=False)
    df['unixReviewTime'] = pd.to_datetime(df['unixReviewTime'], format='%Y-%m-%d')
    return df


def load_metadata_json(metadata_path: str) -> pd.DataFrame:
    df = pd.read_json(metadata_path, lines=True, dtype={'description': pd.StringDtype(),
                                                                  'title': pd.StringDtype(),
                                                                  'also_buy': np.object_,
                                                                  'brand': pd.StringDtype(),
                                                                  'rank': np.float32,
                                                                  'also_view': np.object_,
                                                                  'price': np.float32,
                                                                  'asin': pd.StringDtype(),
                                                                  'details': pd.StringDtype()})
    return df

# Only one time
# Usage : make_preprocess_data('Digital_Music')
def make_preprocess_data(file_name: str):
    review_path = "./data/" + file_name + ".json"
    metadata_path = "./data/meta_" + file_name + ".json"

    review = preprocess_data(review_path)
    metadata = preprocess_metadata(metadata_path)

    review, metadata = processing(review, metadata)

    save_files(review, metadata, './data_cleaned/reviews_cleaned_' + file_name + '.csv', './data_cleaned/metadata_cleaned_' + file_name + '.json')

# Usage : review, metadata = load_data('Digital_Music')
def load_data(file_name: str) -> (pd.DataFrame, pd.DataFrame):
    review_path = "./data_cleaned/reviews_cleaned_" + file_name + ".csv"
    metadata_path = "./data_cleaned/metadata_cleaned_" + file_name + ".json"

    review = load_reviews_csv(review_path)
    metadata = load_metadata_json(metadata_path)

    return review, metadata