from sklearn.svm import LinearSVC
from collections import OrderedDict
from operator import itemgetter
from evaluation import compute_wss

def prioritise_and_evaluate(X_train,
                            y_train,
                            X_test,
                            y_test):
    """
    Trains an L2-regularised linear SVM classifier.
    Documents in the test subset, i.e. X_test, are ranked according to the signed-margin distance between the document feature vectors and the SVM hyperplane.
    :param X_train: Training documents
    :param y_train: Training labels
    :param X_test: Test documents
    :param y_test: Test labels
    :return: work saved over sampling at 95%recall (wss_95) and 100%recall (wss_100)
    """
    # define SVM Classifier
    linear_svc = LinearSVC(loss='squared_hinge', penalty='l2',
                           dual=False, tol=1e-3, class_weight='balanced', C=0.000001)
    # train SVM classifier
    linear_svc.fit(X_train, y_train)

    # get predictions of test documents
    predictions = linear_svc.predict(X_test)

    # get distances between test documents and the SVM hyperplane.
    distances = linear_svc.decision_function(X_test)
    test_indexes_with_distances = {}
    for index, prediction in enumerate(predictions):
        test_indexes_with_distances[index] = distances[index]

    # order documents in a descending order of their distance to the SVM hyperplane
    test_indexes_with_distances = OrderedDict(
        sorted(test_indexes_with_distances.items(), key=itemgetter(1), reverse=True))


    # evaluate ranking in terms of work saved over 95% and 100% recall
    wss_95, wss_100 = compute_wss(indexes_with_predicted_distances=test_indexes_with_distances,
                                  y_test=y_test)
    return wss_95, wss_100