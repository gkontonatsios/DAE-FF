import os

import pandas as pd


def prepare_clinical_datasets(
    dataset_name: str,
    labels_file: str,
    in_path: str,
    text_column: str = "abstracts",
    labels_column: str = "labels",
) -> None:
    """
    github.com/bwallace/citation-screening/tree/master/modeling/curious_snake/data

    :param dataset_name:
    :param labels_file:
    :param in_path: location of a file containing clinical data
    :param text_column:
    :param labels_column:
    :return:
    """
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
            with open(f"{abstract_dir}/{abstract_file}") as fp:
                abstract = fp.readline()

            with open(f"{title_dir}/{abstract_file}") as fp:
                title = fp.readline()

            dataset[doc_id][text_column] = f"{title.lower()}. {abstract.lower()}"

    df = pd.DataFrame.from_dict(dataset).transpose()

    print(f" {len(df[df[labels_column].astype(int) == 1]) / len(df)}")
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
