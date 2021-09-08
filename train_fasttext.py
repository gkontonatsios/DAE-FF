import datetime
from operator import itemgetter

import re
from collections import OrderedDict

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
import os

import fasttext
from typing import List, Tuple
import argparse

from evaluation import compute_wss
import random


def undersample(X: np.array, y: np.array) -> Tuple[np.array, np.array]:
    exclude_label = 0
    include_label = 1
    include_count = len(y[y == include_label])
    exclude_count = len(y[y == exclude_label])

    smaller = include_count if include_count < exclude_count else exclude_count
    print(include_count, exclude_count, smaller)

    incl_indexes = np.where(y == include_label)[0]
    excl_indexes = np.where(y == exclude_label)[0]

    print(incl_indexes.shape, excl_indexes.shape)

    undersampled_indexes = list(random.sample(incl_indexes.tolist(), smaller))
    undersampled_indexes.extend(list(random.sample(excl_indexes.tolist(), smaller)))

    print(len(undersampled_indexes))
    return X[undersampled_indexes], y[undersampled_indexes]


def write_temp_fasttext_train_file(
    X: List[str], y: List[str], outfile="../data/train-tmp.data"
):
    with open(outfile, "w") as fp:
        for X_row, y_row in zip(X, y):
            fp.write(f"__label__{y_row} {X_row}\n")


def train_fasttext(
    X: List[str], y: List[str], lr=1.0, epoch=40, wordNgrams=7, dim=200, loss="hs", undersampling=False,
) -> fasttext.FastText._FastText:
    VECTORS_FILEPATH: str = "../../data/embeddings/BioWordVec_PubMed_MIMICIII_d200.vec.bin"
    TRAIN_FILEPATH: str = "data/train.data"

    if undersampling:
        print('before', X.shape, y.shape)
        X, y= undersample(X=X, y=y)
        print('after', X.shape, y.shape)

    write_temp_fasttext_train_file(X=X, y=y, outfile=TRAIN_FILEPATH)

    model = fasttext.train_supervised(
        input=TRAIN_FILEPATH,
        lr=lr,
        epoch=epoch,
        wordNgrams=wordNgrams,
        dim=dim,
        loss=loss,
        pretrainedVectors=VECTORS_FILEPATH,
    )

    return model


def train_and_evaluate_fasttext(
    input_data_file,
    num_dae_epochs=50,
    num_ff_epochs=100,
    drop_out=0.7,
    dae_minibatch=32,
    ff_minibatch=128,
):
    """
    Trains and evaluates (i.e. 10x2 cross validation) a three-branch model architecture
     which co-ordinates $6$ fully connected layers:
            a) 3 parallel layers which are initialised by three Denoising Autoencoders,
            b) a wide fully connected layer of $3072$ units (concatenation of 3 parallel
                layers) and
            c) two layers of $1024$ units.

    :param input_data_file: a TSV file with the following two columns:
        column 1: abstract of the citation, column 2: classification label
    :param num_dae_epochs: number of training epochs used for the Denoising AutoEncoders
    :param num_ff_epochs: number of training epochs used for the Feed Forward neural
            network
    :param drop_out: drop out reguralisation
    :param dae_minibatch: size of minibatch used for the Denoising AutoEncoders
    :param ff_minibatch: size of minibatch used for the Feed Forward neural network
    """
    df = pd.read_csv(input_data_file, delimiter="\t")

    # get abstracts column into a list
    X = list(df["abstracts"])
    X = [re.sub(r"[\W]+", " ", elem) for elem in X]
    X = [re.sub(r"[\n\r\t ]+", " ", elem) for elem in X]
    X = [elem.lower() for elem in X]
    X = np.asarray(X)

    # get labels column into a list
    y = list(df["labels"])
    y = np.asarray(y)

    wss_95_all_folds = []
    wss_100_all_folds = []
    precision = []
    recall = []

    results_dict = {}

    seeds = [60, 55, 98, 27, 36, 44, 72, 67, 3, 42]

    # perform stratified $10\times2$ cross-validation
    # use same seeds across all baselines
    for seed in seeds:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=(1 - 0.5), random_state=seed)

        for train_indexes, test_indexes in sss.split(X, y):
            # split into training and test subsets
            X_train, X_test = (
                X[train_indexes],
                X[test_indexes],
            )

            # split labels into training and test subsets
            y_train, y_test = y[train_indexes], y[test_indexes]

            model = train_fasttext(X=X_train, y=y_train)

            y_pred = [model.predict(row) for row in X_test]

            predictions = []
            distances = []

            for row in y_pred:
                prediction_int = int(row[0][0][9:])
                predictions.append(prediction_int)
                if prediction_int == 1:
                    distances.append(row[1][0])
                else:
                    distances.append(1 - row[1][0])

            # test_df["pred"] = [model.predict(x)[0][0][9:] for x in test_df["content"].tolist()]
            # test_df["pred_conf"] = [model.predict(x)[1][0] for x in test_df["content"].tolist()]

            f1 = f1_score(y_test, y_pred=predictions)
            precision.append(precision_score(y_test, y_pred=predictions))
            recall.append(recall_score(y_test, y_pred=predictions))

            print(f"prec={precision}, recall={recall} f1={f1}")

            # predictions = linear_svc.predict(X_test)
            #
            # # get distances between test documents and the SVM hyperplane.
            # distances = linear_svc.decision_function(X_test)
            test_indexes_with_distances = {}
            for index, prediction in enumerate(predictions):
                # print(index, prediction, distances[index])
                test_indexes_with_distances[index] = distances[index]

            # order documents in a descending order of their distance to the SVM hyperplane
            test_indexes_with_distances = OrderedDict(
                sorted(
                    test_indexes_with_distances.items(), key=itemgetter(1), reverse=True
                )
            )

            # evaluate ranking in terms of work saved over 95% and 100% recall
            wss_95, wss_100 = compute_wss(
                indexes_with_predicted_distances=test_indexes_with_distances,
                y_test=y_test,
            )


            wss_95_all_folds.append(wss_95)
            wss_100_all_folds.append(wss_100)
        print("Average WSS@95:", np.asarray(wss_95_all_folds).mean())
        print("Average WSS@100:", np.asarray(wss_100_all_folds).mean())

    results_dict["wss_95"] = np.asarray(wss_95_all_folds).mean()
    results_dict["wss_100"] = np.asarray(wss_100_all_folds).mean()
    results_dict["precision"] = np.asarray(precision).mean()
    results_dict["recall"] = np.asarray(recall).mean()

    return np.asarray(wss_95_all_folds).mean(), np.asarray(wss_100_all_folds).mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infile", default="data/processed/AtypicalAntipsychotics.tsv", type=str
    )
    parser.add_argument(
        "--results_file", default="data/fasttext-results_summary.tsv", type=str
    )

    args = parser.parse_args()

    wss95, wss100 = train_and_evaluate_fasttext(
        input_data_file=args.infile,
        num_dae_epochs=150,
        num_ff_epochs=100,
        drop_out=0.7,
        dae_minibatch=32,
        ff_minibatch=128,
    )

    result_dict = {
        args.infile: {"wss95": wss95, "wss100": wss100, "date": datetime.datetime.now()}
    }

    if os.path.isfile(args.results_file):
        df = pd.read_csv(args.results_file, sep="\t")
        df = df.append(
            pd.DataFrame.from_dict(result_dict).transpose().reset_index(),
            ignore_index=True,
        )
    else:
        df = pd.DataFrame.from_dict(result_dict).transpose().reset_index()

    df.drop_duplicates().to_csv(args.results_file, sep="\t", index=False)


#fasttext embeddings
# https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.vec.bin

