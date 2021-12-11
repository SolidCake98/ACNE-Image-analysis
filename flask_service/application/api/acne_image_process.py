from flask import request
from flask_restful import Resource
from application.service.image_processing import transform_image


class AcneController(Resource):

    def post(self):
        image = request.files['img'].read()
        cls, cnt = transform_image(image)
        return {
            'severity' : cls.item(),
            'lesion num': cnt.item()}

