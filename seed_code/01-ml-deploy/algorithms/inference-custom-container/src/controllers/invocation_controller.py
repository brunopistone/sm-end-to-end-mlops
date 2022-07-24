import boto3
import logging
import os
import numpy as np
from services import TranslateService
import tensorflow as tf
import traceback
from utils import constants, utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_SHAPE = 144

comprehend_client = boto3.client("comprehend")

class InvocationController:
    def __init__(self):
        try:
            self.model = self.__model_fn()
            self.tokenizer = self.__tokenizer_fn()
            self.ts = TranslateService()

            X = self.input_handler("Questo è un test")
            prediction = self.predict_fn(X)
            self.output_handler(prediction)

        except Exception as e:
            stacktrace = traceback.format_exc()
            logger.error("{}".format(stacktrace))

            raise e

    def __detect_language(self, body):
        try:
            results = comprehend_client.detect_dominant_language(Text=body)

            max_result = max(results["Languages"], key=lambda x: x['Score'])

            return max_result["LanguageCode"]
        except Exception as e:
            stacktrace = traceback.format_exc()
            logger.error("{}".format(stacktrace))

            raise e

    def __encode(self, tokenizer, shape, body):
        input_ids, input_masks, input_segments = utils.tokenize_sequences(tokenizer, shape, [body])

        return input_ids, input_masks, input_segments

    def __model_fn(self):
        try:
            model = tf.keras.models.load_model(os.path.join(constants.MODEL_PATH, "saved_model", "1"))
            return model

        except Exception as e:
            stacktrace = traceback.format_exc()
            logger.error("{}".format(stacktrace))

            raise e

    def __tokenizer_fn(self):
        try:
            tokenizer = utils.load_pickle(constants.MODEL_PATH, "tokenizer")

            return tokenizer
        except Exception as e:
            stacktrace = traceback.format_exc()
            logger.error("{}".format(stacktrace))

            raise e

    def input_handler(self, body):
        try:
            logger.info("Handle request {}".format(body))
            start_lan = self.__detect_language(body)

            if start_lan != "en":
                body = self.ts.translate_string(body, start_lan, "en")

                logger.info("Translated sentence: {}".format(body))
            else:
                logger.info("Detected en language")

            body = utils.clean_text(body)

            logger.info("review_body: {}".format(body))

            input_ids, input_masks, input_segments = self.__encode(self.tokenizer, MODEL_SHAPE, body)

            return [input_ids, input_masks]
        except Exception as e:
            stacktrace = traceback.format_exc()
            logger.error("{}".format(stacktrace))

            raise e

    def output_handler(self, prediction):
        try:
            logger.info("prediction: {}".format(prediction))

            predicted_classes = []

            prediction_proba = [round(el, 3) for el in max(prediction)]

            predicted_classes.append(str(np.argmax(prediction)))
            predicted_classes.append(str(max(prediction_proba)))

            logger.info("Predictions: {}".format(predicted_classes))

            return ",".join(predicted_classes)
        except Exception as e:
            stacktrace = traceback.format_exc()
            logger.error("{}".format(stacktrace))

            raise e

    def predict_fn(self, X):
        try:
            prediction = self.model.predict(X)

            return prediction
        except Exception as e:
            stacktrace = traceback.format_exc()
            logger.error("{}".format(stacktrace))

            raise e
