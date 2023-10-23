import json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import lxml

def load_data(path: str) -> pd.DataFrame:
    data = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass
    return pd.DataFrame.from_dict(data)

def load_metadata(path: str) -> pd.DataFrame:
    data = []
    with open(path, 'r') as f:
        for l in f:
            data.append(json.loads(l.strip()))
    return pd.DataFrame.from_dict(data)


def drop_col(df: pd.DataFrame, col: [str]) -> pd.DataFrame:
    df.drop(col, axis=1, inplace=True)
    return df

def drop_duplicate(df: pd.DataFrame, subset: [str]) -> pd.DataFrame:
    df.drop_duplicates(subset=subset, keep='first', inplace=True)
    return df

def drop_null(df: pd.DataFrame, subset: [str]) -> pd.DataFrame:
    df.dropna(subset=subset, how='any', inplace=True)
    return df

def nullValues(x : any) -> any:
    if isinstance(x, str) and x == '':
        return pd.NA
    elif isinstance(x, list) and len(x) == 0:
        return pd.NA
    elif isinstance(x, dict) and len(x) == 0:
        return pd.NA
    else:
        return x
    
def cleanText(raw_text):
    '''
    Convert a raw review to a cleaned review
    '''
    if isinstance(raw_text, str):  
        if raw_text == '':
            return pd.NA
        if "<" in raw_text or ">" in raw_text:
            soup = BeautifulSoup(raw_text, 'lxml')
            raw_text = soup.get_text()  # remove html
            soup.decompose()
        letters_only = re.sub("[^a-zA-Z]", " ", raw_text)  # remove non-character
        words = letters_only.lower().split() # convert to lower case 
        return " ".join(words)
    else:
        return pd.NA
    
def cleanRank(raw_rank):
    if isinstance(raw_rank, list):
        if len(raw_rank) == 0:
            return np.nan
        raw_rank = raw_rank[0]
    try:
        return re.search(pattern=r'(\d{1,3}(?:,\d{3})*) in', string=raw_rank).group(1).replace(',','')
    except:
        return np.nan
    
def cleanPrice(raw_price: str):
    if not isinstance(raw_price, str):
        return np.nan
    if raw_price == "" or raw_price[0] != '$':
        return np.nan
    try:
        return re.search(r'(\d{1,3}(?:,\d{3})*(?:.\d{1,10}))', raw_price).group(1).replace(',','')
    except:
        return np.nan
    
def cleanDescription(raw_description):
    if not isinstance(raw_description, list):
        return raw_description
    return ' '.join(raw_description).strip()

def preprocess_data(file_path: str) -> pd.DataFrame:
    data = load_data(file_path)
    data = drop_col(data, 
                    [
                        'reviewTime', 
                        'style', 
                        'reviewerName', 
                        'image'
                    ])
    data = data.map(nullValues)
    data = drop_duplicate(data, ['reviewerID', 'summary'])
    data['unixReviewTime'] = pd.to_datetime(data['unixReviewTime'], unit='s')
    data['reviewText'] = data['reviewText'].apply(cleanText)
    data['vote'] = data['vote'].apply(lambda x: x.replace(',', '') if isinstance(x, str) else x)
    data['vote'] = data['vote'].astype(np.float32)
    data[['reviewerID', 'asin', 'reviewText', 'summary']] = data[['reviewerID', 'asin', 'reviewText', 'summary']].astype(pd.StringDtype())

    return data

def preprocess_metadata(file_path: str) -> pd.DataFrame:
    metadata = load_metadata(file_path)
    metadata = drop_col(metadata,
                        [
                            'category', 
                            'fit', 
                            'tech1', 
                            'tech2', 
                            'feature', 
                            'date', 
                            'similar_item', 
                            'main_cat', 
                            'imageURL', 
                            'imageURLHighRes'
                        ])
    
    
    metadata = metadata.map(nullValues)

    metadata['rank'] = metadata['rank'].apply(cleanRank)
    metadata['rank'] = metadata['rank'].astype(np.float32)

    metadata['price'] = metadata['price'].apply(cleanPrice)
    metadata['price'] = metadata['price'].astype(np.float32)

    metadata['description'] = metadata['description'].apply(cleanDescription)
    metadata['description'] = metadata['description'].astype(pd.StringDtype())

    metadata[['description', 'title', 'brand', 'details', 'asin']] = metadata[['description', 'title', 'brand', 'details', 'asin']].astype(pd.StringDtype())

    return metadata

def processing(review: pd.DataFrame, metadata: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    metadata = drop_null(metadata, ['description', 'title'])

    review = pd.merge(review, metadata, on='asin', how='left')
    review = drop_null(review, ['description', 'title'])
    review = drop_col(review, metadata.columns.to_list())

    return review, metadata

def save_files(review: pd.DataFrame,
               metadata: pd.DataFrame, 
               review_path: str = 'cleaned_data/reviews_cleaned.csv',
               metadata_path: str = 'cleaned_data/metadata_cleaned.json' ) -> None:
    
    review.to_csv(review_path, index=False)
    metadata.to_json(metadata_path, orient='records', lines=True)


# Example of usage
def main():
    review_path = "./data/All_Beauty_tiny.json"
    metadata_path = "./data/meta_All_Beauty_tiny.json"

    review = preprocess_data(review_path)
    metadata = preprocess_metadata(metadata_path)

    review, metadata = processing(review, metadata)

    save_files(review, metadata, './data_cleaned/reviews_cleaned.csv', './data_cleaned/metadata_cleaned.json')

if __name__ == "__main__":
    main()