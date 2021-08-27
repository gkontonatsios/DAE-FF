import keras
import numpy as np
from keras.optimizers import SGD
from sklearn.model_selection import StratifiedShuffleSplit

from classification import prioritise_and_evaluate
from dae_model import DAE
from data_utils import load_bow_vectors_and_labels, normalise
from ff_model import FF

import argparse


def train_and_evaluate_dae_ff(
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

    # Load data from input_data_file into a bag-of-words format
    X, y = load_bow_vectors_and_labels(
        input_data_file=input_data_file, max_features=10000, min_df=10
    )

    X = X.toarray().astype("float32")
    X = normalise(X)

    num_words = int(X.shape[1])

    # Corrupt input X using additive Gaussian noise of a standard deviation $\sigma=0.5$
    X_noisy = X + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=X.shape)

    # =======================Train first DAE Component=======================
    dae_1 = DAE(
        num_words=num_words, encoder_layers=[1024, 512, 256], decoder_layers=[512, 1024]
    ).model

    dae_1.compile(optimizer="adadelta", loss="binary_crossentropy")

    dae_1.fit(
        X_noisy,
        X,
        epochs=int(num_dae_epochs),
        batch_size=int(dae_minibatch),
        shuffle=True,
    )

    X_denoised_first = dae_1.predict(X)
    # =======================End First DAE Component=======================

    # =======================Second DAE Component=======================
    dae_2 = DAE(
        num_words=num_words, encoder_layers=[2048, 512, 256], decoder_layers=[512, 2048]
    ).model

    dae_2.compile(optimizer="adadelta", loss="binary_crossentropy")

    X_noisy = X + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=X.shape)

    dae_2.fit(
        X_noisy,
        X,
        epochs=int(num_dae_epochs),
        batch_size=int(dae_minibatch),
        shuffle=True,
    )

    X_denoised_second = dae_2.predict(X)
    # =======================Second DAE Component=======================

    # =======================Third DAE Component=======================
    dae_3 = DAE(
        num_words=num_words, encoder_layers=[3072, 512, 256], decoder_layers=[512, 3072]
    ).model

    dae_3.compile(optimizer="adadelta", loss="binary_crossentropy")

    X_noisy = X + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=X.shape)

    dae_3.fit(
        X_noisy,
        X,
        epochs=int(num_dae_epochs),
        batch_size=int(dae_minibatch),
        shuffle=True,
    )

    X_denoised_third = dae_3.predict(X)
    # =======================Third DAE Component=======================

    # =======================10x2 cross validation=======================
    # For each fold:
    #   1) Split dataset into training and test subsets
    #   2) Train feedforward neural network on training subset
    #   3) Extract document embeddings using the learned weight matrix of the wide fully
    #       connected layer
    #   4) Use document embeddings to train an SVM classifier
    #   5) Prioritise documents according to the signed-margin distance between the
    #       document feature vector and the SVM hyperplane.
    #   6) Compute WSS@95% recall and WSS@100% recall.

    wss_95_all_folds = []
    wss_100_all_folds = []

    seeds = [60, 55, 98, 27, 36, 44, 72, 67, 3, 42]

    # perform stratified $10\times2$ cross-validation
    # use same seeds across all baselines
    for seed in seeds:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=(1 - 0.5), random_state=seed)

        for train_indexes, test_indexes in sss.split(X, y):
            # split into training and test subsets (first DAE projection)
            X_denoised_first_training, X_denoised_first_test = (
                X_denoised_first[train_indexes],
                X_denoised_first[test_indexes],
            )

            # split into training and test subsets (second DAE projection)
            X_denoised_second_training, X_denoised_second_test = (
                X_denoised_second[train_indexes],
                X_denoised_second[test_indexes],
            )

            # split into training and test subsets (third DAE projection)
            X_denoised_third_training, X_denoised_third_test = (
                X_denoised_third[train_indexes],
                X_denoised_third[test_indexes],
            )

            # split labels into training and test subsets
            y_train, y_test = y[train_indexes], y[test_indexes]

            # convert labels into one hot encoding
            num_classes = int(np.max(y_train) + 1)
            y_train_one_hot = keras.utils.to_categorical(y_train, num_classes)
            y_test_one_hot = keras.utils.to_categorical(y_test, num_classes)

            # normalise trainin/test subsets
            X_denoised_first_training = normalise(X_denoised_first_training)
            X_denoised_first_test = normalise(X_denoised_first_test)

            X_denoised_second_training = normalise(X_denoised_second_training)
            X_denoised_second_test = normalise(X_denoised_second_test)

            X_denoised_third_training = normalise(X_denoised_third_training)
            X_denoised_third_test = normalise(X_denoised_third_test)

            num_features_dae_1 = int(X_denoised_first_training.shape[1])
            num_features_dae_2 = int(X_denoised_second_training.shape[1])
            num_features_dae_3 = int(X_denoised_third_training.shape[1])

            # create a three-branch feed forward model
            ff = FF(
                num_features_dae_1=num_features_dae_1,
                num_features_dae_2=num_features_dae_2,
                num_features_dae_3=num_features_dae_3,
                drop_out=drop_out,
                num_classes=num_classes,
            )

            ff_model = ff.model

            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

            ff_model.compile(
                loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"]
            )

            # train feed forward model
            history = ff_model.fit(
                [
                    X_denoised_first_training,
                    X_denoised_second_training,
                    X_denoised_third_training,
                ],
                y_train_one_hot,
                batch_size=int(ff_minibatch),
                epochs=int(num_ff_epochs),
                verbose=1,
            )

            # weights of branch 1
            w_b1 = ff.branch1.layers[0].get_weights()[0]
            # weights of branch 2
            w_b2 = ff.branch2.layers[0].get_weights()[0]
            # weights of branch 3
            w_b3 = ff.branch3.layers[0].get_weights()[0]

            # weights of main branch (concatenation of branch 1, 2 and 3)
            w_main = ff_model.layers[1].get_weights()[0]

            # get (training) doc embeddings of branch 1
            branch_1_training_doc_embeddings = np.dot(
                np.asarray(X_denoised_first_training), w_b1
            )
            # get (training) doc embeddings of branch 2
            branch_2_training_doc_embeddings = np.dot(
                np.asarray(X_denoised_second_training), w_b2
            )
            # get (training) doc embeddings of branch 3
            branch_3_training_doc_embeddings = np.dot(
                np.asarray(X_denoised_third_training), w_b3
            )

            # get (test) doc embeddings of branch 1
            branch_1_test_doc_embeddings = np.dot(
                np.asarray(X_denoised_first_test), w_b1
            )
            # get (test) doc embeddings of branch 2
            branch_2_test_doc_embeddings = np.dot(
                np.asarray(X_denoised_second_test), w_b2
            )
            # get (test) doc embeddings of branch 3
            branch_3_test_doc_embeddings = np.dot(
                np.asarray(X_denoised_third_test), w_b3
            )

            # get main doc embeddings
            x_train_projected = np.dot(
                np.hstack(
                    (
                        branch_1_training_doc_embeddings,
                        branch_2_training_doc_embeddings,
                        branch_3_training_doc_embeddings,
                    )
                ),
                w_main,
            )
            x_test_projected = np.dot(
                np.hstack(
                    (
                        branch_1_test_doc_embeddings,
                        branch_2_test_doc_embeddings,
                        branch_3_test_doc_embeddings,
                    )
                ),
                w_main,
            )

            # normalise data
            x_train_projected = x_train_projected.reshape(
                (len(x_train_projected), np.prod(x_train_projected.shape[1:]))
            )
            x_test_projected = x_test_projected.reshape(
                (len(x_test_projected), np.prod(x_test_projected.shape[1:]))
            )

            x_train_projected = x_train_projected.astype("float32")
            x_test_projected = x_test_projected.astype("float32")

            # evaluate model according to wss@95%recall and wss@100%recall
            wss_95, wss_100 = prioritise_and_evaluate(
                X_train=x_train_projected,
                y_train=y_train,
                X_test=x_test_projected,
                y_test=y_test,
            )

            wss_95_all_folds.append(wss_95)
            wss_100_all_folds.append(wss_100)
        print("Average WSS@95:", np.asarray(wss_95_all_folds).mean())
        print("Average WSS@100:", np.asarray(wss_100_all_folds).mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", default="sample_data/bpa.tsv", type=str)

    args = parser.parse_args()

    train_and_evaluate_dae_ff(
        input_data_file=args.infile,
        num_dae_epochs=150,
        num_ff_epochs=100,
        drop_out=0.7,
        dae_minibatch=32,
        ff_minibatch=128,
    )
