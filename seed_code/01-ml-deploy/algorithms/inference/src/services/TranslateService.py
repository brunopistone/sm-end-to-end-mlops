import boto3
import logging
import traceback

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

class TranslateService:
    def __init__(self):
        self.client = boto3.client("translate")

    def translate_string(self, row, start_lan="it", end_lan="en"):
        try:
            LOGGER.info("Translating {} from {} to {}".format(row, start_lan, end_lan))
            print("Translating {} from {} to {}".format(row, start_lan, end_lan))

            response = self.client.translate_text(
                Text=row,
                SourceLanguageCode=start_lan,
                TargetLanguageCode=end_lan
            )

            return response["TranslatedText"]

        except Exception as e:
            stacktrace = traceback.format_exc()
            LOGGER.error("{}".format(stacktrace))
            print("{}".format(stacktrace))

            raise e
