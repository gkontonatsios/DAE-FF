import numpy as np


def rank_of_last_pos_doc_at_x_recall(
    indexes_with_predicted_distances, y_test, recall_threshold
):
    """

    :return: Position of the last positive document in the ranked list so that recall
            performance is equal to recall_threshold
    """

    # find total number of positives in dataset
    unique, counts = np.unique(y_test, return_counts=True)
    total_number_of_positives = dict(zip(unique, counts))[1]

    tp = 0
    for rank, index in enumerate(indexes_with_predicted_distances):
        # if document at position = index is positive (i.e., equal to 1.0) then
        # increment the number of true positives
        if y_test[index] == 1.0:
            tp += 1
        recall = tp / total_number_of_positives
        if recall >= recall_threshold:
            last_pos_doc = rank + 1
            return last_pos_doc


def compute_wss(indexes_with_predicted_distances, y_test):
    # number of documents
    N = len(y_test)

    # position of last positive document so that recall is 100%
    last_rel_doc = rank_of_last_pos_doc_at_x_recall(
        indexes_with_predicted_distances=indexes_with_predicted_distances,
        y_test=y_test,
        recall_threshold=1.0,
    )
    # position of last positive document so that recall is 95%
    last_rel_doc_95 = rank_of_last_pos_doc_at_x_recall(
        indexes_with_predicted_distances=indexes_with_predicted_distances,
        y_test=y_test,
        recall_threshold=0.95,
    )

    wss_100 = float(N - last_rel_doc) / float(N)
    wss_95 = (float(N - last_rel_doc_95) / float(N)) - 0.05

    return wss_95, wss_100
