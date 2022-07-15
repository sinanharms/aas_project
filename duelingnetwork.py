import tensorflow as tf
from keras import Model
from keras.layers import Conv2D, Flatten, Dense


class DuelingNetwork(Model):
    """
    Dueling network as described by Wang et al.
    """
    def __init__(self, input_dims, n_actions: int):
        super(DuelingNetwork, self).__init__()

        # three convolutional layers
        self.conv1 = Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation="relu", name="conv_layer_1")
        self.conv2 = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation="relu", name="conv_layer_2")
        self.conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu", name="conv_layer_3")
        # fully connected layer
        self.fc = Dense(units=512, activation="relu", name="fully_connected_layer")
        # advantage and value layers
        self.advantage = Dense(n_actions, activation="relu", name="advantage_layer")
        self.value = Dense(1, activation="relu", name="value_layer")

    def call(self, inputs):
        conv1_out = self.conv1(inputs)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv3_out = Flatten()(conv3_out)
        # advantage stream
        fc_a = self.fc(conv3_out)
        advantage = self.advantage(fc_a)
        # value stream
        fc_v = self.fc(conv3_out)
        value = self.value(fc_v)
        # Q-value formula
        output = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        return output


class SimpleDuelingNetwork(Model):
    """
    A simplified Version of a Dueling Network omitting the image processing
    """
    def __init__(self, n_actions: int):
        super().__init__()
        # layer initialization
        self.fc1 = Dense(units=128, activation="relu", name="fully_connected_layer1")
        self.fc2 = Dense(units=128, activation="relu", name="fully_connected_layer2")
        self.fc3 = Dense(units=64, activation="relu", name="fully_connected_layer3")
        self.advantage = Dense(units=n_actions, name="advantage_layer")
        self.value = Dense(units=1, name="value_layer")

    def call(self, inputs):
        dense1_out = self.fc1(inputs)
        dense2_out = self.fc2(dense1_out)
        dense3_out = self.fc3(dense2_out)
        # advantage stream
        advantage = self.advantage(dense3_out)
        # value stream
        value = self.value(dense3_out)
        # Q-value formula
        output = value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keepdims=True))
        return output

