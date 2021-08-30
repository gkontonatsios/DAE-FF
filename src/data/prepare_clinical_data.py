import os
import re

import pandas as pd


def prepare_clinical_datasets(
    dataset_name,
    labels_file,
    in_path,
    text_column: str = "abstracts",
    labels_column: str = "labels",
):
    abstract_dir = f"{in_path}/Abstracts/"
    title_dir = f"{in_path}/Titles/"
    output_file = f"../../sample_data/{dataset_name}.tsv"

    dataset = {}
    with open(labels_file) as fp:
        for content in fp.readlines():
            paper = {}
            doc_id, label = content.split(" ")[:2]
            if label == "-1":
                paper[labels_column] = 0
            else:
                paper[labels_column] = label

            dataset[doc_id] = paper

    for abstract_file in os.listdir(abstract_dir):
        if (
            not os.path.isdir(f"{abstract_dir}/{abstract_file}")
            and abstract_file in dataset
        ):
            doc_id = abstract_file.split("/")[-1]
            # print(f"{abstract_dir}/{abstract_file}")
            with open(f"{abstract_dir}/{abstract_file}") as fp:
                abstract = fp.readline()

                # print(abstract)

            with open(f"{title_dir}/{abstract_file}") as fp:
                title = fp.readline()

            # dataset[doc_id]['abstract_only'] = abstract
            # dataset[doc_id]['title'] = title

            dataset[doc_id][text_column] = f"{title.lower()}. {abstract.lower()}"
            # dataset[doc_id]['abstracts'] = re.sub(r"\s+", r" ",re.sub(r"(\W+)", r" \1 ",dataset[doc_id]['abstracts']))

    df = pd.DataFrame.from_dict(dataset).transpose()
    print(df[labels_column].astype(int).describe())

    print(len(df[df[labels_column].astype(int) == 1]) / len(df))
    df.to_csv(output_file, sep="\t", index=False)


if __name__ == "__main__":

    datasets = [
        {"dataset_name": "proton_beam", "labels_filename": "proton_labeled_features"},
        {
            "dataset_name": "micro_nutrients",
            "labels_filename": "micro_labeled_features_only",
        },
        {"dataset_name": "copd", "labels_filename": "copd_labeled_terms_only"},
    ]

    for dataset_dict in datasets:
        in_path = f"../../../citation-screening/modeling/curious_snake/data/{dataset_dict['dataset_name']}"

        labels_file = f'{in_path}/{dataset_dict["labels_filename"]}'

        prepare_clinical_datasets(
            dataset_name=dataset_dict["dataset_name"],
            labels_file=labels_file,
            in_path=in_path,
        )
