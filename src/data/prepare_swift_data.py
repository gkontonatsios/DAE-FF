import pandas as pd
from Bio import Entrez
from Bio.Entrez import efetch
from tqdm import tqdm

Entrez.email = "Your.Name@example.org"


def get_from_pubmed(df, text_column: str = 'abstracts', labels_column: str = 'labels'):
    pubmed_id_column = 'PMID'

    df[labels_column] = 1
    df.loc[df["Status"] == "Excluded", labels_column] = 0

    df[text_column] = ""
    for pubmed_id in tqdm(df[pubmed_id_column].tolist()):
        try:
            handle = efetch(
                db="pubmed", id=pubmed_id, retmode="text", rettype="abstract"
            )
            df.loc[
                df[pubmed_id_column] == pubmed_id, text_column
            ] = handle.read()

        except Exception as E:
            print(E, pubmed_id)

            df.loc[df[pubmed_id_column] == pubmed_id, text_column] = "empty"

    return df[[text_column, labels_column]]


def prepare_fluoride_dataset(df: pd.DataFrame, text_column: str = 'abstracts', labels_column: str = 'labels'):
    """This function creates a dataframe in a shape that is later on accepted by the DAE model.

    :param df:
    :param text_column: contains data from Title and Abstract columns
    :param labels_column: contains numerized data from 'Included' column
    :return:
    """
    df[text_column] = df['Title'].fillna("") + '.' + df['Abstract'].fillna("")

    df[labels_column] = 1
    df.loc[df['Included'] == 'EXCLUDED', labels_column] = 0

    return df[[text_column, labels_column]]


if __name__ == '__main__':
    input_folder = "../../data/raw/SWIFT/"
    output_folder = "../../data/processed/"

    datasets = {
        "Fluoride.tsv": prepare_fluoride_dataset,
        "BPA.tsv": get_from_pubmed,
        "Transgenerational.tsv": get_from_pubmed,
        "PFOS-PFOA.tsv": get_from_pubmed,
    }

    for filename, parsing_function in datasets.items():
        print(filename)

        input_data = f"{input_folder}/{filename}"
        output_data = f"{output_folder}/{filename}"
        df = pd.read_csv(input_data, sep='\t')
        df = parsing_function(df)
        df.to_csv(output_data, sep='\t', index=False)
