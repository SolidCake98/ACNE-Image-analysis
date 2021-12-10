from torch import nn
import torch
import pytorch_lightning as pl
from refactored_LDL.losses.KLDivLoss import KLDivLoss
from refactored_LDL.dataset.LDLTransform import genLD
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torchmetrics import Accuracy, Precision, Specificity, MeanSquaredError, MeanAbsoluteError


class MultitaskModel(nn.Module):

    def __init__(self, backbone):
        super(MultitaskModel, self).__init__()

        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extraction = nn.Sequential(*layers)

        self.softamx = nn.Softmax(dim=-1)

        self.fc = nn.Linear(num_filters, 4)
        self.counting = nn.Linear(num_filters, 65)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), -1)

        cls = self.fc(x)
        cnt = self.counting(x)

        cls = self.softamx(cls)
        cnt = self.softamx(cnt)

        cnt2cls = torch.stack((
            torch.sum(cnt[:, :5], 1),
            torch.sum(cnt[:, 5:20], 1), 
            torch.sum(cnt[:, 20:50], 1),
            torch.sum(cnt[:, 50:], 1)), 
            dim=1)

        return [cls, cnt, cnt2cls]


class MultiTaskLossWrapper(nn.Module):

    def __init__(self, task_num) -> None:
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, preds, *args):
        kl_loss = KLDivLoss()
        losses = []
        for i in range(self.task_num):
            loss = kl_loss(preds[i], args[i])
            precision = torch.exp(-self.log_vars[i])
            loss = precision * loss + self.log_vars[i]
            losses.append(loss)
        return sum(losses)


class LDLModel(pl.LightningModule):

    def __init__(self, backbone, device):
        super().__init__()

        self.d = device
        self.LR = 0.001
        self.multitask_model = MultitaskModel(backbone)
        self.loss = MultiTaskLossWrapper(task_num=3)
        
        self.accuracy = Accuracy()
        self.precision = Precision()
        self.sensetivity = Specificity()
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError

    def forward(self, x):
        x = self.multitask_model(x)
        return x

    def configure_optimizers(self):
        params = []
        for key, value in dict(self.multitask_model.named_parameters()).items():
            if value.requires_grad:
                params += [{'params': [value], 'lr': self.LR * 1.0, 'weight_decay': 5e-4}]

        optimizer = torch.optim.SGD(params, momentum=0.9)
        lr_scheduler = StepLR(optimizer, 5, 0.5)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def training_step(self, train_batch, idx):
        x, y, l = train_batch   
        l = l.cpu().numpy()
        l -= 1
        ld = genLD(l, 3, 'klloss', 65)
        ld_4 = np.vstack((np.sum(ld[:, :5], 1), np.sum(ld[:, 5:20], 1), np.sum(ld[:, 20:50], 1), np.sum(ld[:, 50:], 1))).transpose()

        ld = torch.from_numpy(ld).to(device=self.d).float()
        ld_4 = torch.from_numpy(ld_4).to(device=self.d).float()

        preds = self.multitask_model(x)
        loss = self.loss(preds, ld_4, ld, ld_4)


        if self.trainer.is_last_batch:
            sch = self.lr_schedulers()
            sch.step()

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        
        mean_accuracy = self.accuracy(preds[0], l).compute()
        self.log('train_step_accuracy', mean_accuracy)

        return loss

    def training_epoch_end(self, outputs) -> None:
        return super().training_epoch_end(outputs)

    def validation_step(self, val_batch, batch_idx):
        x, y, l = val_batch
        preds = self.multitask_model(x)

        loss_f = torch.nn.CrossEntropyLoss()
        loss = loss_f(preds[2], y.long())
        self.log('valid_class_loss', loss, on_epoch=True, prog_bar=True)


        return loss

    def backward(self, loss, optimizer, idx):
        loss.backward()
