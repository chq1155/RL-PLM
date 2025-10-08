import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class MetricsToFileCallback(Callback):
    def __init__(self, folder, data_name, monitor, model_path, loss_fn, lr, idt=30, test_ratio=0.2):
        model_name = model_path.split("/")[-1]
        self.folder = folder

        self.file_path_spearman = f"{data_name}_{model_name}_monitor-{monitor}_spearman_loss-{loss_fn}_lr-{lr}.txt"
        self.file_path_pearson = f"{data_name}_{model_name}_monitor-{monitor}_pearson_loss-{loss_fn}_lr-{lr}.txt"
        self.file_path_mse = (
            f"{data_name}_{model_name}_monitor-{monitor}_mse_loss-{loss_fn}_lr-{lr}.txt"
        )
        self.file_path_mae = (
            f"{data_name}_{model_name}_monitor-{monitor}_mae_loss-{loss_fn}_lr-{lr}.txt"
        )
        self.file_path_r2 = (
            f"{data_name}_{model_name}_monitor-{monitor}_R2_loss-{loss_fn}_lr-{lr}.txt"
        )

        self.data_name = data_name
        self.test_ratio = test_ratio
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.file_path_spearman = os.path.join(self.folder, self.file_path_spearman)
        self.file_path_pearson = os.path.join(self.folder, self.file_path_pearson)
        self.file_path_mse = os.path.join(self.folder, self.file_path_mse)
        self.file_path_r2 = os.path.join(self.folder, self.file_path_r2)
        self.file_path_mae = os.path.join(self.folder, self.file_path_mae)

    def on_test_epoch_end(self, trainer, pl_module):
        # spearman
        sprman = trainer.logged_metrics.get("test_spearman_corr", None)
        if sprman:
            with open(self.file_path_spearman, "a") as f:
                f.write(
                    f"protein_name={self.data_name}, test_ratio={self.test_ratio},  spearman={sprman}\n"
                )
        else:
            with open(self.file_path_spearman, "a") as f:
                f.write(
                    f"protein_name={self.data_name}, test_ratio={self.test_ratio}, spearman=NA\n"
                )
        # pearson
        pearson = trainer.logged_metrics.get("test_pearson_corr", None)
        if pearson:
            with open(self.file_path_pearson, "a") as f:
                f.write(
                    f"protein_name={self.data_name}, test_ratio={self.test_ratio},  pearson={pearson}\n"
                )
        else:
            with open(self.file_path_pearson, "a") as f:
                f.write(
                    f"protein_name={self.data_name}, test_ratio={self.test_ratio}, pearson=NA\n"
                )

        # mse
        mse = trainer.logged_metrics.get("test_mse", None)
        if mse:
            with open(self.file_path_mse, "a") as f:
                f.write(
                    f"protein_name={self.data_name}, test_ratio={self.test_ratio},  mse={mse}\n"
                )
        else:
            with open(self.file_path_mse, "a") as f:
                f.write(
                    f"protein_name={self.data_name}, test_ratio={self.test_ratio}, mse=NA\n"
                )
        # r2
        r2 = trainer.logged_metrics.get("test_r2", None)
        if r2:
            with open(self.file_path_r2, "a") as f:
                f.write(
                    f"protein_name={self.data_name}, test_ratio={self.test_ratio},  r2={r2}\n"
                )
        else:
            with open(self.file_path_r2, "a") as f:
                f.write(
                    f"protein_name={self.data_name}, test_ratio={self.test_ratio}, r2=NA\n"
                )
        # mae
        mae = trainer.logged_metrics.get("test_mae", None)
        if mae:
            with open(self.file_path_mae, "a") as f:
                f.write(
                    f"protein_name={self.data_name}, test_ratio={self.test_ratio},  mae={mae}\n"
                )
        else:
            with open(self.file_path_mae, "a") as f:
                f.write(
                    f"protein_name={self.data_name}, test_ratio={self.test_ratio}, mae=NA\n"
                )
