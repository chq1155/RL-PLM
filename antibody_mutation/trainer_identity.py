import torch, os, sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import seed_everything
from lit_model import LitModel
from dataset import WrapperDataset
from utils.test_callback import MetricsToFileCallback
from utils.data_utils import load_data
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_locate", type=str, default="./model/esm2_650m")

    # dataset
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_folder", type=str, default="./data/csv_AB645")
    parser.add_argument("--data_name", type=str, default="AB645")
    parser.add_argument("--max_length", type=int, default=None)

    # model
    parser.add_argument("--hidden_size", type=int, default=1280)
    parser.add_argument("--num_heads", type=int, default=4)  
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--out_dim", type=int, default=1)
    parser.add_argument("--freeze_backbone", action="store_true") 

    # Trainer
    parser.add_argument("--loss", type=str, default="mse")
    parser.add_argument("--monitor", type=str, default="val_pearson_corr")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=5.0)
    parser.add_argument("--gradient_clip_algorithm", type=str, default="value")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--rm_abnormal", type=bool, default=False)  # !
    args = parser.parse_args()

    seed_everything(args.seed)
    return args


def main():
    args = parse_args()
    # load data
    file_path = args.data_folder
    data_name = args.data_name
    train_data, val_data, test_data = load_data(
        data_name,
        file_path,
        val_ratio=args.val_ratio,
        random_seed=args.seed,
        rm_abnormal=args.rm_abnormal,
    )

    # get dataset and data_loader
    dataset = WrapperDataset(
        args=args,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
    )
    train_loader = dataset.get_train_loader()
    val_loader = dataset.get_val_loader()
    test_loader = dataset.get_test_loader()

    print("args: ", args)
    model_name = args.model_locate.split("/")[-1]
    monitor = args.monitor

    model = LitModel(args)
    mode = "max" if "corr" in monitor else "min"
    ckpt_name = f"{model_name}_{data_name}-{monitor}_lr-{args.lr}_loss-{args.loss}_batch-{args.batch_size}_patience-{args.patience}"
    model_checkpoint = ModelCheckpoint(
        (
            f"checkpoints_rmabnormal_testratio_{args.test_ratio}/{data_name}"
            if args.rm_abnormal
            else f"checkpoints_testratio_{args.test_ratio}/{data_name}"
        ),  # !
        monitor=monitor,
        mode=mode,
        filename=ckpt_name,
        verbose=True,
    )
    early_stop = EarlyStopping(
        monitor=monitor, mode=mode, patience=args.patience, verbose=True
    )

    if not args.rm_abnormal:
        metrics_folder = (
            f"results_testratio-{args.test_ratio}/{model_name}/{data_name}_{args.loss}_batch-{args.batch_size}_patience-{args.patience}"
        )
    else:
        metrics_folder = f"results_testratio-{args.test_ratio}_rmabnormal/{model_name}/{data_name}_{args.loss}_batch-{args.batch_size}_patience-{args.patience}"
    metrics2file = MetricsToFileCallback(
        folder=metrics_folder,
        data_name=data_name,
        monitor=monitor,
        model_path=args.model_locate,
        loss_fn=args.loss,
        lr=args.lr,
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        num_nodes=args.num_nodes,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        deterministic=True,
        # precision=args.precision,
        callbacks=[model_checkpoint, early_stop, metrics2file],
    )
    trainer.fit(model, train_loader, val_loader)
    model = LitModel.load_from_checkpoint(model_checkpoint.best_model_path)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
