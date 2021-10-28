import datetime
import os
import pandas as pd
import random

random.seed(1)
import numpy as np

np.random.seed(1)
# import tensorflow as tf
# tf.random.set_random_seed(1)

from train_and_test import train_and_evaluate_dae_ff

# from train_fasttext import train_and_evaluate_fasttext

if __name__ == "__main__":

    # results_file = "data/fasttext-abstract-time.tsv"
    results_file = "data/dae-title-time.tsv"
    result_dict = {}
    # input_dir = "../CitationScreeningReplicability/data/processed/"
    input_dir = "data/processed/"

    for infile in os.listdir(input_dir)[6:]:
        print(infile)
        if infile not in ["SkeletalMuscleRelaxants.tsv", "proton_beam.tsv"]:
            continue

        if infile[-3:] != "tsv":
            continue
        single_file_dict = train_and_evaluate_dae_ff(
            input_data_file=f"{input_dir}/{infile}",
            num_dae_epochs=150,
            num_ff_epochs=100,
            drop_out=0.7,
            dae_minibatch=32,
            ff_minibatch=128,
        )
        single_file_dict["date"] = datetime.datetime.now()
        single_file_dict["pretrained"] = "no"
        single_file_dict["model"] = "dae"
        # single_dict = {
        #     "wss95": wss95,
        #     "wss100": wss100,
        #     "date": datetime.datetime.now(),
        #     "pretrained": "no",
        #     "model": "no undersampling, spacy tokenizer",
        # }

        result_dict[infile] = single_file_dict

        if os.path.isfile(results_file):
            df = pd.read_csv(results_file, sep="\t")
            df = df.drop_duplicates()
            df = df.append(
                pd.DataFrame.from_dict(result_dict).transpose().reset_index(),
                ignore_index=True,
            )
        else:
            df = pd.DataFrame.from_dict(result_dict).transpose().reset_index()

        df.to_csv(results_file, sep="\t", index=False)
