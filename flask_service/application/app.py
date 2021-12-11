from flask import Flask
from flask_restful import Api

from .api.acne_image_process import AcneController
from LDL.factory import get_model

ml_model = None

def create_app(cfg):
    app = Flask(__name__)
    api = Api(app)

    api.add_resource(AcneController, "/api/v1/acne_severity")

    setup_model(app, cfg)
    return app


def setup_model(app, cfg):
    global ml_model
    model = get_model(**cfg['model'])
    ml_model = model