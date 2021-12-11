from pytorch_lightning import callbacks
from .model.multi_model import LDLModel
from .dataset.dataset_processing import DatasetProcessing
from .model.backbones import get_backbone

from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


def get_data_loader(data_path, data_file, batch_size, num_workers):

    normalize = transforms.Normalize(mean=[0.45815152, 0.361242, 0.29348266],
                                    std=[0.2814769, 0.226306, 0.20132513])

    dset = DatasetProcessing(
            data_path, 
            data_file, 
            transform=transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
        )

    loader = DataLoader(dset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=True)
    return loader


def get_model(checkpoint, backbone):
    backbone = get_backbone(backbone, pretrined=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LDLModel(backbone, device)

    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def get_logger(name, project):
    return WandbLogger(name=name, project=project)


def get_trainer(backbone, dataset_t, dataet_v, logger, trainer):
    name = backbone
    backbone = get_backbone(backbone)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = LDLModel(backbone, device, 0.001)

    train_loader = get_data_loader(**dataset_t)
    test_loader = get_data_loader(**dataet_v)

    logger = get_logger(**logger)

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        dirpath="./checkpoints/swin_t/",
        filename= f"{name}-" + "{epoch:02d}-{valid_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    trainer = pl.Trainer(gpus=1, logger=logger, callbacks=[checkpoint_callback], **trainer)
    return trainer, model, train_loader, test_loader