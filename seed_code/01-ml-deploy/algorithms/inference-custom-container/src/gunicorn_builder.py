import flask_builder as flask_builder
from gunicorn.app.base import BaseApplication

# Main application class
class GunicornApplication(BaseApplication):
    def init(self, parser, opts, args):
        pass

    def load_config(self):
        pass

    def __init__(self, port=None):
        super(GunicornApplication, self).__init__()
        self.app = flask_builder.create_app()
        if port:
            self.cfg.set("bind", ":"+str(port))

    def load(self):
        return self.app

def create_app(port=8080):
    return GunicornApplication(port=port)
