from keras.layers import Dense
from keras.layers import Dropout, Activation
from keras.layers import Merge
from keras.models import Sequential


class FF(object):
    def __init__(
        self,
        num_features_dae_1,
        num_features_dae_2,
        num_features_dae_3,
        drop_out,
        num_classes,
    ):
        self.num_features_dae_1 = num_features_dae_1
        self.num_features_dae_2 = num_features_dae_2
        self.num_features_dae_3 = num_features_dae_3
        self.drop_out = drop_out
        self.num_classes = num_classes

        self.model = Sequential()

        # first branch
        self.branch1 = Sequential()
        self.branch1.add(Dense(1024, input_shape=(self.num_features_dae_1,)))
        self.branch1.add(Activation("relu"))
        self.branch1.add(Dropout(float(self.drop_out)))

        # second branch
        self.branch2 = Sequential()
        self.branch2.add(Dense(1024, input_shape=(self.num_features_dae_2,)))
        self.branch2.add(Activation("relu"))
        self.branch2.add(Dropout(float(self.drop_out)))

        # third branch
        self.branch3 = Sequential()
        self.branch3.add(Dense(1024, input_shape=(self.num_features_dae_3,)))
        self.branch3.add(Activation("relu"))
        self.branch3.add(Dropout(float(self.drop_out)))

        # merge branches into main branch
        self.model.add(Merge([self.branch1, self.branch2, self.branch3], mode="concat"))

        # additional dense layers
        self.model.add(Dense(1024))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(float(self.drop_out)))

        self.model.add(Dense(1024))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(float(self.drop_out)))

        self.model.add(Dense(self.num_classes))
        self.model.add(Activation("softmax"))
