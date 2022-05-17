import logging
import numpy as np
import os
from .StopTrainingClass import StopTrainingClass
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPool1D, Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from transformers import AutoConfig, AutoTokenizer, TFAutoModel
import traceback

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

class BertModel:
    def __init__(self, bert_model_name="amazon/bort"):
        self.model = None
        self.bert_configs = None
        self.bert_model = None
        self.bert_model_name = bert_model_name
        self.bert_tokenizer = None

    def build(self, shape, fine_tuning=False):
        try:
            input_ids_in = Input(shape=(int(shape),), name='input_token', dtype='int32')
            input_masks_in = Input(shape=(int(shape),), name='masked_token', dtype='int32')

            encoder, pooler = self.bert_model(input_ids_in, attention_mask=input_masks_in)
            X = Dense(117, activation="relu")(pooler)
            X = Dropout(self.bert_configs.hidden_dropout_prob)(X)
            X = Dense(
                3,
                kernel_initializer=TruncatedNormal(stddev=self.bert_configs.initializer_range),
                activation="softmax"
            )(X)

            self.model = Model(
                inputs=[input_ids_in, input_masks_in],
                outputs=[X]
            )

            if fine_tuning:
                for layer in self.model.layers[:3]:
                    layer.trainable = False

            optimizer = Adam(learning_rate=3e-5)
            loss_obj = CategoricalCrossentropy(from_logits=True)

            self.model.compile(optimizer=optimizer, loss=loss_obj, metrics=['accuracy'])

            LOGGER.info("{}".format(self.model.summary()))
        except Exception as e:
            stacktrace = traceback.format_exc()
            LOGGER.error("{}".format(stacktrace))

            raise e

    def fit(self, checkpoint_path, model_name, X_train, y_train, checkpoints=False, stop_training=False, epochs=10, batch_size=100):

        callbacks = []

        if checkpoints:
            if not os.path.exists(checkpoint_path):
                os.mkdir(checkpoint_path)

            checkpointName = os.path.join(checkpoint_path, model_name + ".ckpt")

            # Create a callback that saves the src's weights
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointName,
                                                             save_weights_only=True,
                                                             verbose=1)

            callbacks.append(cp_callback)

        if stop_training:
            stop_callback = StopTrainingClass()
            callbacks.append(stop_callback)

        history = self.model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=int(epochs),
            batch_size=batch_size,
            callbacks=callbacks
        )

        return history

    def get_config(self):
        return self.bert_configs

    def get_model(self):
        return self.model

    def retrieve_model(self, output_hidden_states=False):
        try:
            self.bert_configs = AutoConfig.from_pretrained(self.bert_model_name)
            self.bert_configs.output_hidden_states = output_hidden_states

            self.bert_model = TFAutoModel.from_pretrained(self.bert_model_name, config=self.bert_configs)

            self.bert_model = self.bert_model.bert

            return self.bert_model
        except Exception as e:
            stacktrace = traceback.format_exc()
            LOGGER.error("{}".format(stacktrace))

            raise e

    def retrieve_tokenizer(self):
        try:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)

            return self.bert_tokenizer
        except Exception as e:
            stacktrace = traceback.format_exc()
            LOGGER.error("{}".format(stacktrace))

            raise e

    def tokenize_sequence(self, max_seq_length, data):
        try:
            input_ids = []
            input_masks = []
            input_segments = []

            bert_input = self.bert_tokenizer.encode_plus(
                data,
                add_special_tokens=True,
                max_length=max_seq_length,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=True
            )

            input_ids.append(bert_input['input_ids'])
            input_masks.append(bert_input['attention_mask'])
            input_segments.append(bert_input['token_type_ids'])

            return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32'), np.asarray(input_segments, dtype='int32')
        except Exception as e:
            stacktrace = traceback.format_exc()
            LOGGER.error("{}".format(stacktrace))

            raise e

    def tokenize_sequences(self, max_seq_length, data, labels):
        try:
            input_ids = []
            input_masks = []
            input_segments = []

            for sentence in data:
                bert_input = self.bert_tokenizer.encode_plus(
                    sentence,
                    add_special_tokens=True,
                    max_length=max_seq_length,
                    truncation=True,
                    padding='max_length',
                    return_attention_mask=True,
                    return_token_type_ids=True
                )

                input_ids.append(bert_input['input_ids'])
                input_masks.append(bert_input['attention_mask'])
                input_segments.append(bert_input['token_type_ids'])

            return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32'), np.asarray(
                input_segments, dtype='int32'), np.asarray(labels)
        except Exception as e:
            stacktrace = traceback.format_exc()
            LOGGER.error("{}".format(stacktrace))

            raise e