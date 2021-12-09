import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from refactored_LDL.model.multi_model import LDLModel
from refactored_LDL.dataset.dataset_processing import DatasetProcessing

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    backbone = torchvision.models.resnet50()
    model = LDLModel(backbone, device)

    BATCH_SIZE = 32
    NUM_WORKERS = 4

    DATA_PATH = '../Classification/JPEGImages'
    TRAIN_FILE = '../Classification/NNEW_trainval_' + str(0) + '.txt'
    TEST_FILE = '../Classification/NNEW_test_' + str(0) + '.txt'

    normalize = transforms.Normalize(mean=[0.45815152, 0.361242, 0.29348266],
                                     std=[0.2814769, 0.226306, 0.20132513])

    dset_train = DatasetProcessing(
        DATA_PATH, 
        TRAIN_FILE, 
        transform=transforms.Compose([
                transforms.Scale((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    )

    dset_test = DatasetProcessing(
        DATA_PATH, 
        TEST_FILE, 
        transform=transforms.Compose([
                transforms.Scale((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    )

    train_loader = DataLoader(dset_train,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=True)


    test_loader = DataLoader(dset_test,
                            batch_size=BATCH_SIZE,

                            num_workers=NUM_WORKERS,
                            pin_memory=True)

    trainer = pl.Trainer(gpus=1, max_epochs=10, log_every_n_steps=36)
    trainer.fit(model, train_loader, test_loader)