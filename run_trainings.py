import datetime
import os
import pandas as pd

# from train_and_test import train_and_evaluate_dae_ff
from train_fasttext import train_and_evaluate_fasttext

if __name__ == "__main__":

    results_file = "data/fasttext-results_summary.tsv"
    result_dict = {}
    input_dir = "data/processed/"

    for infile in os.listdir(input_dir):
        print(infile)
        wss95, wss100 = train_and_evaluate_fasttext(
            input_data_file=f"{input_dir}/{infile}",
            num_dae_epochs=150,
            num_ff_epochs=100,
            drop_out=0.7,
            dae_minibatch=32,
            ff_minibatch=128,
        )

        single_dict = {
            "wss95": wss95,
            "wss100": wss100,
            "date": datetime.datetime.now(),
            "pretrained": "BioWordVec_PubMed_MIMICIII_d200",
            "model": "no undersampling",
        }

        result_dict[infile] = single_dict

        if os.path.isfile(results_file):
            df = pd.read_csv(results_file, sep="\t")
            df = df.append(
                pd.DataFrame.from_dict(result_dict).transpose().reset_index(),
                ignore_index=True,
            )
        else:
            df = pd.DataFrame.from_dict(result_dict).transpose().reset_index()

        df.to_csv(results_file, sep="\t", index=False)
