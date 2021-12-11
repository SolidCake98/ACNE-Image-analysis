from torch import nn
import torch
import pytorch_lightning as pl
from LDL.dataset.LDLTransform import genLD

import numpy as np
from torch.optim.lr_scheduler import StepLR
from torchmetrics import (Accuracy, 
                            Precision, 
                            Recall, 
                            Specificity, 
                            MeanSquaredError, 
                            MeanAbsoluteError)


class MultitaskModel(nn.Module):
    """
    Model, which solves two task (counting, classification) via Hard-parametr sharing
    Args:
    backbone: model for feature extraction
    """

    def __init__(self, backbone):
        super(MultitaskModel, self).__init__()

        num_filters = backbone.num_features
        self.feature_extraction = backbone

        self.softamx = nn.Softmax(dim=-1)

        self.fc = nn.Linear(num_filters, 4)
        self.counting = nn.Linear(num_filters, 65)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), -1)

        cls = self.fc(x)
        cnt = self.counting(x)

        cls = self.softamx(cls) + 1e-4
        cnt = self.softamx(cnt) + 1e-4

        cnt2cls = torch.stack((
            torch.sum(cnt[:, :5], 1),
            torch.sum(cnt[:, 5:20], 1), 
            torch.sum(cnt[:, 20:50], 1),
            torch.sum(cnt[:, 50:], 1)), 
            dim=1)

        return [cls, cnt, cnt2cls]


class MultiTaskLossWrapper(nn.Module):
    """
    Mutple loss wrapper with learning wiegths for each task
    (Results show, that this approach didn't work for given task)
    """
    def __init__(self, task_num) -> None:
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, cls, cnt, cnt2cls, *args):
        kl_loss1 = nn.KLDivLoss()
        kl_loss2 = nn.KLDivLoss()
        kl_loss3 = nn.KLDivLoss()

        loss_cls = kl_loss1(torch.log(cls), args[0]) * 4.0
        precision_cls = torch.exp(-self.log_vars[0])
        loss_cls = precision_cls * loss_cls + self.log_vars[0]

        loss_cnt = kl_loss2(torch.log(cnt), args[1]) * 65.0
        precision_cnt = torch.exp(-self.log_vars[1])
        loss_cnt = precision_cnt * loss_cnt + self.log_vars[1]

        loss_cnt2cls = kl_loss3(torch.log(cnt2cls), args[2]) * 4.0
        precision_cnt2cls = torch.exp(-self.log_vars[2])
        loss_cnt = precision_cnt2cls * loss_cnt2cls + self.log_vars[2]

        return loss_cls + loss_cnt + loss_cnt2cls


class MultiTaskLossWrapperThird(nn.Module):
    """
    Loss wrapper from the article
    """

    def __init__(self, task_num) -> None:
        super(MultiTaskLossWrapperThird, self).__init__()
        self.task_num = task_num

    def forward(self, cls, cnt, cnt2cls, *args):
        kl_loss1 = nn.KLDivLoss()
        kl_loss2 = nn.KLDivLoss()
        kl_loss3 = nn.KLDivLoss()

        loss_cls = kl_loss1(torch.log(cls), args[0]) * 4.0
        loss_cnt = kl_loss2(torch.log(cnt), args[1]) * 65.0
        loss_cnt2cls = kl_loss3(torch.log(cnt2cls), args[2]) * 4.0

        loss = (loss_cls + loss_cnt2cls) * 0.5 * 0.6 + loss_cnt * (1.0 - 0.6)
    
        return loss


class LDLModel(pl.LightningModule):
    """
    Label distribution learning model
    """

    def __init__(self, backbone, device, LR=0.001):
        super().__init__()

        self.d = device
        self.LR = LR
        self.multitask_model = MultitaskModel(backbone)
        self.loss = MultiTaskLossWrapperThird(task_num=3)
        
        self.train_mae =  MeanAbsoluteError()
        self.train_accuracy =  Accuracy()

        self.valid_mae =  MeanAbsoluteError()
        self.valid_mse =  MeanSquaredError()

        self.valid_accuracy = Accuracy()
        self.valid_precision = Precision()
        self.valid_specificity = Specificity()
        self.valid_sensetivity = Recall()


        self.train_metrics_counting = {
            'train_mae': self.train_mae
        }

        self.train_metrics_classes = {
            'train_accuracy':  self.train_accuracy,
        }

        self.valid_metrics_counting = {
            'valid_mae': self.valid_mae,
            'valid_mse': self.valid_mse,
        }

        self.valid_metrics_classes = {
            'valid_accuracy': self.valid_accuracy,
            'valid_precison': self.valid_precision,
            'valid_specificity': self.valid_specificity,
            'valid_sensetivity': self.valid_sensetivity
        }

    def forward(self, x):
        x = self.multitask_model(x)
        return x

    def metrics(self, name, loss, y, l, preds, metrics_cls, metrics_cnt, on_step=True):
        self.log(name, loss, on_epoch=True, prog_bar=True)

        _, preds_l = torch.max(preds[1], 1)

        for key, value in metrics_cls.items():
            v = value(preds[2], y)
            self.log(key, v, on_epoch=True, on_step=on_step)

        for key, value in metrics_cnt.items():
            v = value(preds_l, l)
            self.log(key, value, on_epoch=True, on_step=on_step)

    def configure_optimizers(self):
        params = []
        for key, value in dict(self.multitask_model.named_parameters()).items():
            if value.requires_grad:
                params += [{'params': [value], 'lr': self.LR * 1.0, 'weight_decay': 5e-4}]

        optimizer = torch.optim.SGD(params, momentum=0.9)
        lr_scheduler = StepLR(optimizer, 30, 0.5)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def training_step(self, train_batch, idx):
        x, y, l = train_batch   

        l = l.cpu().numpy()
        l -= 1
        ld = genLD(l, 1.5, 'klloss', 65)
        ld_4 = np.vstack((np.sum(ld[:, :5], 1), np.sum(ld[:, 5:20], 1), np.sum(ld[:, 20:50], 1), np.sum(ld[:, 50:], 1))).transpose()

        ld = torch.from_numpy(ld).to(device=self.d).float()
        ld_4 = torch.from_numpy(ld_4).to(device=self.d).float()

        preds = self.multitask_model(x)
        loss = self.loss(preds[0], preds[1], preds[2], ld_4, ld, ld_4)

        if self.trainer.is_last_batch:
            sch = self.lr_schedulers()
            sch.step()

        l = torch.from_numpy(l).to(device=self.device)

        self.metrics('train_loss', loss, y, l, preds, 
            self.train_metrics_classes, self.train_metrics_counting)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y, l = val_batch
        preds = self.multitask_model(x)

        loss_f = torch.nn.CrossEntropyLoss()
        loss = loss_f(preds[2], y.long())

        self.metrics('valid_loss', loss, y, l, preds, 
            self.valid_metrics_classes, self.valid_metrics_counting, on_step=False)

        return loss

    def backward(self, loss, optimizer, idx):
        loss.backward()
