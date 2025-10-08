import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
import torch
from torch import nn
from model import SeqBindModel
from pytorch_lightning import LightningModule
from torchmetrics import (
    MeanSquaredError, 
    SpearmanCorrCoef,
    PearsonCorrCoef,
    R2Score, 
    MeanAbsoluteError,
)
from collections import OrderedDict


class LitModel(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.lr = args.lr

        self.model = SeqBindModel(args)

        self.valid_metrics = nn.ModuleDict(
            OrderedDict(
                [
                    ("val_mse", MeanSquaredError()),
                    ("val_spearman_corr", SpearmanCorrCoef()),
                    ("val_pearson_corr", PearsonCorrCoef()),
                    ("val_r2", R2Score()),
                    ("val_mae", MeanAbsoluteError())
                ]
            )
        )
        self.test_metrics = nn.ModuleDict(
            OrderedDict(
                [
                    ("test_mse", MeanSquaredError()),
                    ("test_spearman_corr", SpearmanCorrCoef()),
                    ("test_pearson_corr", PearsonCorrCoef()),
                    ("test_r2", R2Score()),
                    ("test_mae", MeanAbsoluteError())
                ]
            )
        )
        if args.loss == "mse":
            self.criterion = nn.MSELoss()
        elif args.loss == "huber":
            self.criterion = nn.HuberLoss()
        else:
            raise ValueError("Invalid loss function")

        self.mlm_loss = nn.CrossEntropyLoss(ignore_index=-100)

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):  # TODO 修改输入
        labels = batch["labels"]

        outs, logits_w_ab, logits_w_ag, logits_m_ab, logits_m_ag = self.model(
            wt_ab_inputs_ids=batch["wt_ab_inputs_ids"],
            wt_ab_inputs_mask=batch["wt_ab_inputs_mask"],
            mut_ab_inputs_ids=batch["mut_ab_inputs_ids"],
            mt_ab_inputs_mask=batch["mt_ab_inputs_mask"],
            wt_ag_inputs_ids=batch["wt_ag_inputs_ids"],
            wt_ag_inputs_mask=batch["wt_ag_inputs_mask"],
            mut_ag_inputs_ids=batch["mut_ag_inputs_ids"],
            mt_ag_inputs_mask=batch["mt_ag_inputs_mask"],
        )

        id_loss = self.mlm_loss(
            logits_w_ab.view(-1, logits_w_ab.size(-1)),
            batch["wt_ab_inputs_ids"].view(-1)
        ) + self.mlm_loss(
            logits_w_ag.view(-1, logits_w_ag.size(-1)),
            batch["wt_ag_inputs_ids"].view(-1)
        ) + self.mlm_loss(
            logits_m_ab.view(-1, logits_m_ab.size(-1)),
            batch["mut_ab_inputs_ids"].view(-1)
        ) + self.mlm_loss(
            logits_m_ag.view(-1, logits_m_ag.size(-1)),
            batch["mut_ag_inputs_ids"].view(-1)
        )
        self.log("id_loss", id_loss.item(), on_step=True, on_epoch=True, prog_bar=True)

        loss = self.criterion(outs, labels)
        self.log("loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)

        return loss + id_loss

    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        outs, logits_w_ab, logits_w_ag, logits_m_ab, logits_m_ag = self.model(
            wt_ab_inputs_ids=batch["wt_ab_inputs_ids"],
            wt_ab_inputs_mask=batch["wt_ab_inputs_mask"],
            mut_ab_inputs_ids=batch["mut_ab_inputs_ids"],
            mt_ab_inputs_mask=batch["mt_ab_inputs_mask"],
            wt_ag_inputs_ids=batch["wt_ag_inputs_ids"],
            wt_ag_inputs_mask=batch["wt_ag_inputs_mask"],
            mut_ag_inputs_ids=batch["mut_ag_inputs_ids"],
            mt_ag_inputs_mask=batch["mt_ag_inputs_mask"],
        )

        id_loss = self.mlm_loss(
                logits_w_ab.view(-1, logits_w_ab.size(-1)),
                batch["wt_ab_inputs_ids"].view(-1)
            ) + self.mlm_loss(
                logits_w_ag.view(-1, logits_w_ag.size(-1)),
                batch["wt_ag_inputs_ids"].view(-1)
            ) + self.mlm_loss(
                logits_m_ab.view(-1, logits_m_ab.size(-1)),
                batch["mut_ab_inputs_ids"].view(-1)
            ) + self.mlm_loss(
                logits_m_ag.view(-1, logits_m_ag.size(-1)),
                batch["mut_ag_inputs_ids"].view(-1)
            )
        self.log("val_id_loss", id_loss.item(), on_step=True, on_epoch=True, prog_bar=True)

        w_ab_seq = torch.argmax(logits_w_ab, dim=-1)
        w_ag_seq = torch.argmax(logits_w_ag, dim=-1)
        m_ab_seq = torch.argmax(logits_m_ab, dim=-1)
        m_ag_seq = torch.argmax(logits_m_ag, dim=-1)

        id_acc = (
            (w_ab_seq == batch["wt_ab_inputs_ids"]).float().mean() +
            (w_ag_seq == batch["wt_ag_inputs_ids"]).float().mean() +
            (m_ab_seq == batch["mut_ab_inputs_ids"]).float().mean() +
            (m_ag_seq == batch["mut_ag_inputs_ids"]).float().mean()
        ) / 4.0
        self.log("val_id_acc", id_acc.item(), on_step=True, on_epoch=True, prog_bar=True)
        
        for name, metric in self.valid_metrics.items():
            metric(outs, labels)
            self.log(name, metric, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        labels = batch["labels"]
        outs, logits_w_ab, logits_w_ag, logits_m_ab, logits_m_ag = self.model(
            wt_ab_inputs_ids=batch["wt_ab_inputs_ids"],
            wt_ab_inputs_mask=batch["wt_ab_inputs_mask"],
            mut_ab_inputs_ids=batch["mut_ab_inputs_ids"],
            mt_ab_inputs_mask=batch["mt_ab_inputs_mask"],
            wt_ag_inputs_ids=batch["wt_ag_inputs_ids"],
            wt_ag_inputs_mask=batch["wt_ag_inputs_mask"],
            mut_ag_inputs_ids=batch["mut_ag_inputs_ids"],
            mt_ag_inputs_mask=batch["mt_ag_inputs_mask"],
        )

        id_loss = self.mlm_loss(
                logits_w_ab.view(-1, logits_w_ab.size(-1)),
                batch["wt_ab_inputs_ids"].view(-1)
            ) + self.mlm_loss(
                logits_w_ag.view(-1, logits_w_ag.size(-1)),
                batch["wt_ag_inputs_ids"].view(-1)
            ) + self.mlm_loss(
                logits_m_ab.view(-1, logits_m_ab.size(-1)),
                batch["mut_ab_inputs_ids"].view(-1)
            ) + self.mlm_loss(
                logits_m_ag.view(-1, logits_m_ag.size(-1)),
                batch["mut_ag_inputs_ids"].view(-1)
            )
        self.log("test_id_loss", id_loss.item(), on_step=True, on_epoch=True, prog_bar=True)

        w_ab_seq = torch.argmax(logits_w_ab, dim=-1)
        w_ag_seq = torch.argmax(logits_w_ag, dim=-1)
        m_ab_seq = torch.argmax(logits_m_ab, dim=-1)
        m_ag_seq = torch.argmax(logits_m_ag, dim=-1)

        id_acc = (
            (w_ab_seq == batch["wt_ab_inputs_ids"]).float().mean() +
            (w_ag_seq == batch["wt_ag_inputs_ids"]).float().mean() +
            (m_ab_seq == batch["mut_ab_inputs_ids"]).float().mean() +
            (m_ag_seq == batch["mut_ag_inputs_ids"]).float().mean()
        ) / 4.0
        self.log("test_id_acc", id_acc.item(), on_step=True, on_epoch=True, prog_bar=True)

        for name, metric in self.test_metrics.items():
            metric(outs, labels)
            self.log(name, metric, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=self.lr
        )
        return optimizer
