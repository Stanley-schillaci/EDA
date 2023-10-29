import pandas as pd

from src.utils import load_data, make_preprocess_data

# raw data should be in ./data
# cleaned data should be in ./data_cleaned

def main():
    # Use only one time
    #make_preprocess_data('Digital_Music')

    review, metadata = load_data('Digital_Music')

    # show some stats
    print("review")
    print(review.dtypes)

    print("metadata")
    print(metadata.dtypes)

if __name__ == '__main__':
    main()