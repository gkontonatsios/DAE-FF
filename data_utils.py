import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from stemmer import StemTokenizer


def normalise(X):
    """
    Normalise feature values of X between 0 and 1

    :param X: input feature vectors
    """
    X = X.astype("float32")
    X_normalised = X.reshape((len(X), np.prod(X.shape[1:])))

    return X_normalised


def load_bow_vectors_and_labels(input_data_file, max_features=10000, min_df=10):
    """
    Loads data from input_data_file and returns a matrix corresponding to the
    bag-of-words representation of the abstracts (i.e. X) and a list of labels (i.e. Y)

    :param input_data_file: a TSV file with the following two columns:
            column 1: abstract of the citation, column 2: classification label
    :param max_features: bag-of-words space consists of consider the top max_features
            ordered by word frequency across the corpus.
    :param min_df: Ignore words that have a document frequency lower than min_df
    :return: X, y

    """

    # load tsv file into a Pandas data frame
    df = pd.read_csv(input_data_file, delimiter="\t")

    # get abstracts column into a list
    raw_abstracts = list(df["abstracts"])
    # get labels column into a list
    y = list(df["labels"])

    # Convert into bag-of-words. Filter out words based on max_feautres and min_df
    count_vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        stop_words="english",
        tokenizer=StemTokenizer(num_docs=len(raw_abstracts)),
    )
    X = count_vectorizer.fit_transform(raw_abstracts)
    y = np.asarray(y)
    return X, y
