import os
import json
import hashlib
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import mean_squared_error
import tensorflow.keras.backend as K
from tools import Observation


class AlphaZeroModel(object):
    def __init__(self, observation: Observation):
        self.observation: Observation = observation
        self.cnn_filter_num: int = 128
        self.cnn_filter_size: int = 3
        self.l2_reg: float = 1e-4
        self.res_layer_num: int = 2
        self.value_fc_size: int = 256
        # noinspection PyTypeChecker
        self.model: Model = None

    def build(self):
        in_x = x = layers.Input(self.observation.input_shape())

        # (batch, channels, height, width)
        x = layers.Conv2D(
            filters=self.cnn_filter_num,
            kernel_size=self.cnn_filter_size,
            padding="same",
            data_format="channels_first",
            kernel_regularizer=regularizers.l2(self.l2_reg)
        )(x)
        x = layers.BatchNormalization(axis=1)(x)
        x = layers.Activation("relu")(x)

        for _ in range(self.res_layer_num):
            x = self._build_residual_block(x)

        res_out = x
        # for policy output
        x = layers.Conv2D(
            filters=2,
            kernel_size=1,
            data_format="channels_first",
            kernel_regularizer=regularizers.l2(self.l2_reg)
        )(res_out)
        x = layers.BatchNormalization(axis=1)(x)
        x = layers.Activation("relu")(x)
        x = layers.Flatten()(x)
        # no output for 'pass'
        policy_out = layers.Dense(
            self.observation.nb_actions(),
            kernel_regularizer=regularizers.l2(self.l2_reg),
            activation="softmax",
            name="policy_out"
        )(x)

        # for value output
        x = layers.Conv2D(
            filters=1,
            kernel_size=1,
            data_format="channels_first",
            kernel_regularizer=regularizers.l2(self.l2_reg)
        )(res_out)
        x = layers.BatchNormalization(axis=1)(x)
        x = layers.Activation("relu")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(
            units=self.value_fc_size,
            kernel_regularizer=regularizers.l2(self.l2_reg),
            activation="relu"
        )(x)
        value_out = layers.Dense(
            units=1,
            kernel_regularizer=regularizers.l2(self.l2_reg),
            activation="tanh",
            name="value_out"
        )(x)

        self.model = Model(in_x, [policy_out, value_out], name="AlphaZeroModel")
        self.model.summary()
        plot_model(self.model, "AlphaZeroModel.png", True, True)

    def _build_residual_block(self, x):
        in_x = x
        x = layers.Conv2D(
            filters=self.cnn_filter_num,
            kernel_size=self.cnn_filter_size,
            padding="same",
            data_format="channels_first",
            kernel_regularizer=regularizers.l2(self.l2_reg)
        )(x)
        x = layers.BatchNormalization(axis=1)(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(
            filters=self.cnn_filter_num,
            kernel_size=self.cnn_filter_size,
            padding="same",
            data_format="channels_first",
            kernel_regularizer=regularizers.l2(self.l2_reg)
        )(x)
        x = layers.BatchNormalization(axis=1)(x)
        x = layers.Add()([in_x, x])
        x = layers.Activation("relu")(x)
        return x

    def save(self, config_path: str, weight_path: str):
        with open(config_path, "wt") as f:
            json.dump(self.model.get_config(), f)
            self.model.save_weights(weight_path)

    def load(self, config_path: str, weight_path: str):
        if config_path is None or weight_path is None:
            print("New model without save.")
            self.build()
        elif not os.path.exists(config_path) or not os.path.exists(weight_path):
            path = os.path.dirname(config_path)
            if not os.path.exists(path):
                os.mkdir(path)
            path = os.path.dirname(weight_path)
            if not os.path.exists(path):
                os.mkdir(path)
                print("New model with save:", weight_path)
            self.build()
            self.save(config_path, weight_path)
        else:
            print("Load model:", weight_path)
            with open(config_path, "rt") as f:
                self.model = Model.from_config(json.load(f))
            self.model.load_weights(weight_path)

    @staticmethod
    def fetch_digest(weight_path):
        if os.path.exists(weight_path):
            m = hashlib.sha256()
            with open(weight_path, "rb") as f:
                m.update(f.read())
            return m.hexdigest()

    def predict(self, x):
        assert x.ndim in (len(self.observation.input_shape()), len(self.observation.input_shape()) + 1)
        assert x.shape == self.observation.input_shape() or x.shape[1:] == self.observation.input_shape()
        orig_x = x
        if x.ndim == len(self.observation.input_shape()):
            x = np.expand_dims(x, axis=0)
        policy, value = self.model.predict_on_batch(x)

        if orig_x.ndim == len(self.observation.input_shape()):
            return policy[0], value[0]
        else:
            return policy, value


def objective_function_for_policy(y_true, y_pred):
    # can use categorical_crossentropy??
    return K.sum(-y_true * K.log(y_pred + K.epsilon()), axis=-1)


def objective_function_for_value(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)
