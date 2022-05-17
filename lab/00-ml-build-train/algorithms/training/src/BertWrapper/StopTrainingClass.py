import logging
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StopTrainingClass(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        logger.info("Accuracy: {}".format(logs.get('accuracy')))
        logger.info("Loss: {}".format(logs.get('loss')))
        logger.info("Val Accuracy: {}".format(logs.get('val_accuracy')))

        if(logs.get('val_accuracy') is not None):
            if(logs.get('loss') <= 0.4 or logs.get('val_accuracy') >= 0.85 or logs.get('accuracy') >= 0.85):
                if (epoch + 1 > 4):
                    logger.info("Cancelling artefact; is good!")
                    self.model.stop_training = True
                else:
                    logger.info("Too low epochs for canceling!")
        else:
            if (logs.get('loss') <= 0.4 or logs.get('accuracy') >= 0.85):
                if (epoch + 1 > 4):
                    logger.info("Cancelling artefact; is good!")
                    self.model.stop_training = True
                else:
                    logger.info("Too low epochs for canceling!")
