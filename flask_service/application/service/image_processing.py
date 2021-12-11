from torchvision import transforms
from PIL import Image
import io
import torch

import application.app 

def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45815152, 0.361242, 0.29348266],
                            std=[0.2814769, 0.226306, 0.20132513])
    ])

    image = Image.open(io.BytesIO(image_bytes))
    tensor =  my_transforms(image).unsqueeze(0)


    preds = application.app.ml_model(tensor)

    _, preds_cls = torch.max(preds[0] + preds[2], 1)
    _, preds_cnt = torch.max(preds[1], 1)

    return preds_cls, preds_cnt