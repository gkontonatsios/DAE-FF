import os
from typing import Dict

from tqdm import tqdm
import pandas as pd
from corpus_similarity import Similarity

cs = Similarity(language="eng")


def dataset_statistics(df) -> Dict:
    statistics = {}
    exclusions = df[df["labels"].astype(int) == 0]
    inclusions = df[df["labels"].astype(int) == 1]

    len_df = len(df)
    statistics["len"] = len_df
    statistics["#exclusions"] = len(exclusions)
    statistics["%exclusions"] = len(exclusions) / len_df
    statistics["#inclusions"] = len(inclusions)
    statistics["%inclusions"] = len(inclusions) / len_df

    statistics["max_WSS@95%"] = len(exclusions) / len_df - 0.05

    statistics["avg_char_num"] = df["abstracts"].str.len().mean()
    statistics["avg_char_num_inclusions"] = inclusions["abstracts"].str.len().mean()
    statistics["avg_char_num_exclusions"] = exclusions["abstracts"].str.len().mean()

    statistics["avg_word_num"] = df["abstracts"].str.split().str.len().mean()
    statistics["avg_word_num_inclusions"] = (
        inclusions["abstracts"].str.split().str.len().mean()
    )
    statistics["avg_word_num_exclusions"] = (
        exclusions["abstracts"].str.split().str.len().mean()
    )

    exclusion_text = exclusions["abstracts"].tolist()
    inclusion_text = inclusions["abstracts"].tolist()

    statistics["exclusions_similarity"] = cs.calculate(exclusion_text, exclusion_text)
    statistics["inclusions_similarity"] = cs.calculate(inclusion_text, inclusion_text)
    statistics["intra_similarity"] = cs.calculate(exclusion_text, inclusion_text)

    return statistics


if __name__ == "__main__":
    data_dir = "../../data/processed/"

    total_statistics = {}
    for file in tqdm(os.listdir(data_dir)):
        dataset_name = file.split(".")[0]

        df = pd.read_csv(f"{data_dir}/{file}", sep="\t")
        total_statistics[dataset_name] = dataset_statistics(df=df)

    results = pd.DataFrame.from_dict(total_statistics).transpose()
    results.sort_index().to_csv("../../data/statistics.tsv", sep="\t")
