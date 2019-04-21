import numpy as np
from data_utils import load_bow_vectors_and_labels, normalise
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import StratifiedShuffleSplit
import keras
from keras.models import Sequential
from keras.layers import Dropout, Activation
from keras.layers import Merge
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta
from classification import prioritise_and_evaluate

from dae_model import DAE


def evaluate_on_10_runs_on_all_datasets(input_data_file,
                                        num_dae_epochs=50,
                                        num_ff_epochs=100,
                                        drop_out=0.7,
                                        dae_minibatch=32,
                                        ff_minibatch=128):
    """
    Trains and evaluates (i.e. 10x2 cross validation) a three-branch model architecture
     which co-ordinates $6$ fully connected layers:
            a) 3 parallel layers which are initialised by three Denoising Autoencoders,
            b) a wide fully connected layer of $3072$ units (concatenation of 3 parallel layers) and
            c) two layers of $1024$ units.

    :param input_data_file: a TSV file with the following two columns: column 1: abstract of the citation, column 2: classification label
    :param num_dae_epochs: number of training epochs used for the Denoising AutoEncoders
    :param num_ff_epochs: number of training epochs used for the Feed Forward neural network
    :param drop_out: drop out reguralisation
    :param dae_minibatch: size of minibatch used for the Denoising AutoEncoders
    :param ff_minibatch: size of minibatch used for the Feed Forward neural network
    """

    # Load data from input_data_file into a bag-of-words format
    X, y = load_bow_vectors_and_labels(input_data_file=input_data_file,
                                       max_features=10000,
                                       min_df=10)

    X = X.toarray().astype('float32')
    X = normalise(X)

    num_words = int(X.shape[1])

    # Corrupt input X using additive Gaussian noise of a standard deviation $\sigma=0.5$
    X_noisy = X + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=X.shape)


    # =======================Train first DAE Component=======================
    dae_1 = DAE(num_words=num_words,
                encoder_layers=[1024, 512, 256],
                decoder_layers=[512, 1024]).model

    dae_1.compile(optimizer='adadelta', loss='binary_crossentropy')

    dae_1.fit(X_noisy, X,
                    epochs=int(num_dae_epochs),
                    batch_size=int(dae_minibatch),
                    shuffle=True)

    X_denoised = dae_1.predict(X)
    # =======================End First DAE Component=======================

    # =======================Second DAE Component=======================
    dae_2 = DAE(num_words=num_words,
                encoder_layers=[2048, 512, 256],
                decoder_layers=[512, 2048]).model

    dae_2.compile(optimizer='adadelta', loss='binary_crossentropy')

    X_noisy = X + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=X.shape)

    dae_2.fit(X_noisy, X,
                    epochs=int(num_dae_epochs),
                    batch_size=int(dae_minibatch),
                    shuffle=True)

    X_denoised_second = dae_2.predict(X)
    # =======================Second DAE Component=======================

    # =======================Third DAE Component=======================
    dae_3 = DAE(num_words=num_words,
                encoder_layers=[3072, 512, 256],
                decoder_layers=[512, 3072]).model

    dae_3.compile(optimizer='adadelta', loss='binary_crossentropy')

    X_noisy = X + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=X.shape)

    dae_3.fit(X_noisy, X,
                    epochs=int(num_dae_epochs),
                    batch_size=int(dae_minibatch),
                    shuffle=True)

    X_denoised_third = dae_3.predict(X)
    # =======================Third DAE Component=======================



    # =======================10x2 cross validation=======================
    # For each fold:
    #   1) Split dataset into training and test subsets
    #   2) Train feedforward neural network on training subset
    #   3) Extract document embeddings using the learned weight matrix of the wide fully connected layer
    #   4) Use document embeddings to train an SVM classifier
    #   5) Prioritise documents according to the signed-margin distance between the document feature vector and the SVM hyperplane.
    #   6) Compute WSS@95% recall and WSS@100% recall.

    wss_95_all_folds = []
    wss_100_all_folds = []

    seeds = [60, 55, 98, 27, 36, 44, 72, 67, 3, 42]

    for seed in seeds:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=(1 - 0.5), random_state=seed)
        for train_indexes, test_indexes in sss.split(X, y):
            X_train, X_test = X[train_indexes], X[test_indexes]
            y_train, y_test = y[train_indexes], y[test_indexes]

        X_bow_training, X_test_bow = X[train_indexes], X[test_indexes]

        X_denoised_training, X_denoised_test = X_denoised[train_indexes], X_denoised[
            test_indexes]

        X_denoised_second_training, X_denoised_second_test = X_denoised_second[train_indexes], \
                                                        X_denoised_second[test_indexes]

        X_denoised_third_training, X_denoised_third_test = X_denoised_third[train_indexes], \
                                                      X_denoised_third[test_indexes]

        X_denoised_training = normalise(X_denoised_training)
        X_denoised_test = normalise(X_denoised_test)

        X_denoised_second_training = normalise(X_denoised_second_training)
        X_denoised_second_test = normalise(X_denoised_second_test)

        X_denoised_third_training = normalise(X_denoised_third_training)
        X_denoised_third_test = normalise(X_denoised_third_test)


        y_train, y_test = y[train_indexes], y[test_indexes]

        max_words_bow = int(X_bow_training.shape[1])
        max_words_autoencoder = int(X_denoised_training.shape[1])
        max_words_autoencoder_second = int(X_denoised_second_training.shape[1])
        max_words_autoencoder_third = int(X_denoised_third_training.shape[1])


        num_classes = int(np.max(y_train) + 1)
        y_train_one_hot = keras.utils.to_categorical(y_train, num_classes)
        y_test_one_hot = keras.utils.to_categorical(y_test, num_classes)

        print('Building model...')
        model = Sequential()

        branch1 = Sequential()
        branch1.add(Dense(1024, input_shape=(max_words_autoencoder,)))
        branch1.add(Activation('relu'))
        branch1.add(Dropout(float(drop_out)))

        branch2 = Sequential()
        branch2.add(Dense(1024, input_shape=(max_words_autoencoder_second,)))
        branch2.add(Activation('relu'))
        branch2.add(Dropout(float(drop_out)))

        branch3 = Sequential()
        branch3.add(Dense(1024, input_shape=(max_words_autoencoder_third,)))
        branch3.add(Activation('relu'))
        branch3.add(Dropout(float(drop_out)))



        model.add(Merge([branch1, branch2, branch3], mode='concat'))

        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(float(drop_out)))

        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(float(drop_out)))

        model.add(Dense(num_classes))
        model.add(Activation('softmax'))


        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        history = model.fit([X_denoised_training, X_denoised_second_training, X_denoised_third_training], y_train_one_hot,
                            batch_size=int(ff_minibatch),
                            epochs=int(num_ff_epochs),
                            verbose=1)

        g_branch_1 = branch1.layers[0].get_config()
        h_branch_1 = branch1.layers[0].get_weights()[0]
        bias_1 = branch1.layers[0].get_weights()[1]

        g_branch_2 = branch2.layers[0].get_config()
        h_branch_2 = branch2.layers[0].get_weights()[0]
        bias_2 = branch2.layers[0].get_weights()[1]

        g_branch_3 = branch3.layers[0].get_config()
        h_branch_3 = branch3.layers[0].get_weights()[0]
        bias_3 = branch3.layers[0].get_weights()[1]



        g = model.layers[1].get_config()
        h = model.layers[1].get_weights()[0]
        bias_concat = model.layers[1].get_weights()[1]

        branch_1_training_doc_embeddings = np.dot(np.asarray(X_denoised_training), h_branch_1)
        branch_2_training_doc_embeddings = np.dot(np.asarray(X_denoised_second_training), h_branch_2)
        branch_3_training_doc_embeddings = np.dot(np.asarray(X_denoised_third_training), h_branch_3)

        branch_1_test_doc_embeddings = np.dot(np.asarray(X_denoised_test), h_branch_1)
        branch_2_test_doc_embeddings = np.dot(np.asarray(X_denoised_second_test), h_branch_2)
        branch_3_test_doc_embeddings = np.dot(np.asarray(X_denoised_third_test), h_branch_3)


        x_train_projected = np.dot(np.hstack(
            (branch_1_training_doc_embeddings, branch_2_training_doc_embeddings, branch_3_training_doc_embeddings)), h)
        x_test_projected = np.dot(
            np.hstack((branch_1_test_doc_embeddings, branch_2_test_doc_embeddings, branch_3_test_doc_embeddings)), h)

        x_train_projected = x_train_projected.reshape(
            (len(x_train_projected), np.prod(x_train_projected.shape[1:])))
        x_test_projected = x_test_projected.reshape(
            (len(x_test_projected), np.prod(x_test_projected.shape[1:])))

        x_train_projected = x_train_projected.astype('float32')
        x_test_projected = x_test_projected.astype('float32')

        wss_95, wss_100 = prioritise_and_evaluate(X_train=x_train_projected,
                                                  y_train = y_train,
                                                  X_test=x_test_projected,
                                                  y_test=y_test)




        wss_95_all_folds.append(wss_95)
        wss_100_all_folds.append(wss_100)
    print('Average WSS@95:', np.asarray(wss_95_all_folds).mean())
    print('Average WSS@100:', np.asarray(wss_100_all_folds).mean())


if __name__ == '__main__':
    evaluate_on_10_runs_on_all_datasets(input_data_file='sample_data/temp/BPA.raw.temp_tsv',
                                        num_dae_epochs=150,
                                        num_ff_epochs=100,
                                        drop_out=0.7,
                                        dae_minibatch=32,
                                        ff_minibatch=128)