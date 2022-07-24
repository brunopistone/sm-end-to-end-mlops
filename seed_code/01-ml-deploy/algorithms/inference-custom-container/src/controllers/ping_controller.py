import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PingController:
    def handle(self, data=None):
        return "Application Up and Running"