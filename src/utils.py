import numpy as np
import pandas as pd


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