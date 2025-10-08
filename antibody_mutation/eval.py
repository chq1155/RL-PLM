import os
import torch
from lit_model import LitModel
from argparse import ArgumentParser, Namespace
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import seed_everything
from utils.data_utils import get_AB_pair_data, get_s1131_data
from dataset import SeqDataset
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
from typing import List, Optional, Sequence, Dict, Any
import logging
from pathlib import Path


def parse_args() -> Namespace:
    """
    Parse command-line arguments and set global seed.
    """
    parser = ArgumentParser()
    parser.add_argument("--model_locate", type=str, default="facebook/esm2_t33_650M_UR50D")

    parser.add_argument(
        "--ckpt_locate",
        type=str,
        default="./checkpoints_sigmul_rmabnormal/AB1101/esm2_t33_650M_UR50D_AB1101-val_pearson_corr_lr-0.0001_loss-mse.ckpt",
    )
    parser.add_argument(
        "--preds_path",
        type=str,
        default="./test_preds/AB1101",
    )
    # dataset
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--filt_path", type=str, default="./data/sigmul_data/AB1101_multiple.csv"
    )
    parser.add_argument("--max_length", type=int, default=None)

    # optional CDR info
    parser.add_argument("--cdr_info_path", type=str, default=None, help="Optional CSV path with CDR info")
    parser.add_argument(
        "--cdr_fragments",
        type=str,
        default="L3",
        help="Comma-separated CDR fragment columns to mask, e.g., 'H1,H2,H3,L1,L2,L3'",
    )

    # model
    parser.add_argument("--hidden_size", type=int, default=1280)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--out_dim", type=int, default=1)
    parser.add_argument("--freeze_backbone", action="store_true", help="Default is False unless specified")

    # Trainer
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    seed_everything(args.seed)
    return args


def create_sequence_masks(
    df: pd.DataFrame,
    sequences: Sequence[str],
    pdb_ids: Sequence[str],
    fragment_columns: Sequence[str],
) -> List[torch.Tensor]:
    """
    Build boolean masks for specified CDR fragments for each sequence.

    Args:
        df: DataFrame with columns ['PDB', 'H1', 'H2', 'H3', 'L1', 'L2', 'L3'] or subset.
        sequences: Antibody sequences (one per sample).
        pdb_ids: PDB IDs aligned with sequences.
        fragment_columns: Which CDR fragment columns to consider.

    Returns:
        List of 1D boolean tensors aligned with each input sequence.
    """
    if len(sequences) != len(pdb_ids):
        raise ValueError("sequences and pdb_ids must have the same length")

    masks: List[torch.Tensor] = []

    for seq, pdb_id in zip(sequences, pdb_ids):
        mask = torch.zeros(len(seq), dtype=torch.bool)

        pdb_row = df[df["PDB"] == pdb_id]
        if pdb_row.empty:
            masks.append(mask)
            continue

        pdb_row = pdb_row.iloc[0]

        for col in fragment_columns:
            if col in pdb_row and pd.notna(pdb_row[col]):
                fragment_seq = str(pdb_row[col]).strip()
                if fragment_seq:
                    start_pos = 0
                    while True:
                        pos = seq.find(fragment_seq, start_pos)
                        if pos == -1:
                            break
                        end_pos = pos + len(fragment_seq)
                        mask[pos:end_pos] = True
                        start_pos = pos + 1

        masks.append(mask)

    return masks


def masks_to_batch_tensor(masks: List[torch.Tensor]) -> torch.Tensor:
    """
    Pad and stack a list of 1D boolean masks. Adds one False pad at both ends
    to roughly align with special tokens added by some tokenizers.
    """
    if not masks:
        return torch.empty(0, 0, dtype=torch.bool)

    padded_tensors: List[torch.Tensor] = []
    for tensor in masks:
        false_tensor = torch.tensor([False], dtype=tensor.dtype, device=tensor.device)
        padded_tensor = torch.cat([false_tensor, tensor, false_tensor])
        padded_tensors.append(padded_tensor)

    lengths = [len(mask) for mask in padded_tensors]
    if len(set(lengths)) == 1:
        return torch.stack(padded_tensors)
    else:
        from torch.nn.utils.rnn import pad_sequence
        return pad_sequence(padded_tensors, batch_first=True, padding_value=False)


class EvalDataset(Dataset):
    """
    Evaluation dataset with on-the-fly tokenization and optional CDR masks.
    """

    def __init__(self, args: Namespace, data, cdr_df: Optional[pd.DataFrame] = None):
        super(EvalDataset, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.model_locate = args.model_locate
        self.max_length = args.max_length
        self.truncation = True if self.max_length else False

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_locate)
        self.dataset = self.get_dataset(data)
        self.cdr_df = cdr_df
        self.cdr_fragments: Sequence[str] = tuple(
            [frag.strip() for frag in str(getattr(args, "cdr_fragments", "L3")).split(",") if frag.strip()]
        )

    def get_dataset(self, data):
        (
            wt_ab_seqs,
            wt_ag_seqs,
            mt_ab_seqs,
            mt_ag_seqs,
            labels,
            pdb_ids,
        ) = data
        dataset = SeqDataset(
            wt_ab_seqs=wt_ab_seqs,
            mt_ab_seqs=mt_ab_seqs,
            wt_ag_seqs=wt_ag_seqs,
            mt_ag_seqs=mt_ag_seqs,
            labels=labels,
            pdb_ids=pdb_ids,
        )
        return dataset

    def collator_fn(self, batch) -> Dict[str, Any]:
        wt_ab_seqs, mt_ab_seqs, wt_ag_seqs, mt_ag_seqs, labels, pdb_ids = zip(*batch)

        wt_ab_inputs = self.tokenizer(
            wt_ab_seqs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=self.truncation,
        )
        mt_ab_inputs = self.tokenizer(
            mt_ab_seqs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=self.truncation,
        )
        wt_ag_inputs = self.tokenizer(
            wt_ag_seqs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=self.truncation,
        )
        mt_ag_inputs = self.tokenizer(
            mt_ag_seqs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=self.truncation,
        )
        labels = torch.tensor(labels, dtype=torch.float32)  # [B]

        cdr_masks = None
        if self.cdr_df is not None:
            cdr_masks_list = create_sequence_masks(
                self.cdr_df, wt_ab_seqs, pdb_ids, fragment_columns=self.cdr_fragments
            )
            cdr_masks = masks_to_batch_tensor(cdr_masks_list)

        batch_out: Dict[str, Any] = {
            "wt_ab_inputs_ids": wt_ab_inputs["input_ids"],
            "wt_ab_inputs_mask": wt_ab_inputs["attention_mask"],
            "mut_ab_inputs_ids": mt_ab_inputs["input_ids"],
            "mt_ab_inputs_mask": mt_ab_inputs["attention_mask"],
            "wt_ag_inputs_ids": wt_ag_inputs["input_ids"],
            "wt_ag_inputs_mask": wt_ag_inputs["attention_mask"],
            "mut_ag_inputs_ids": mt_ag_inputs["input_ids"],
            "mt_ag_inputs_mask": mt_ag_inputs["attention_mask"],
            "labels": labels,
            "pdb_ids": pdb_ids,
        }
        if cdr_masks is not None:
            batch_out["cdr_masks"] = cdr_masks
        return batch_out

    def get_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator_fn,
            pin_memory=True,
        )


def load_eval_dataset(filt_path: str):
    """
    Load evaluation dataset and ensure pdb_ids exist for downstream usage.
    """
    if "S1131" in filt_path:
        (
            ab_hl_chains,
            ag_ab_chains,
            mt_ab_hl_chains,
            mt_ag_ab_chains,
            labels,
        ) = get_s1131_data(filt_path)
        # Ensure pdb_ids is present; use empty placeholders when unavailable.
        n = len(labels)
        pdb_ids = [""] * n
    elif "AB" in filt_path:
        (
            ab_hl_chains,
            ag_ab_chains,
            mt_ab_hl_chains,
            mt_ag_ab_chains,
            labels,
            pdb_ids,
        ) = get_AB_pair_data(filt_path, rm_abnormal=True)
    else:
        raise ValueError(f"Invalid dataset name: {filt_path}")
    data = (
        ab_hl_chains,
        ag_ab_chains,
        mt_ab_hl_chains,
        mt_ag_ab_chains,
        labels,
        pdb_ids,
    )
    return data


def load_cdr_info(file_path: str) -> pd.DataFrame:
    """
    Load CDR info CSV and keep only supported columns.
    """
    df = pd.read_csv(file_path)

    required_columns = ["H1", "H2", "H3", "L1", "L2", "L3"]
    available_columns = [col for col in required_columns if col in df.columns]
    df_filtered = df[available_columns]

    return df_filtered


def eval(args: Namespace) -> None:
    """
    Run evaluation for a given checkpoint and dataset.
    """
    logging.info("Loading checkpoint from %s", args.ckpt_locate)
    lit_model = LitModel.load_from_checkpoint(args.ckpt_locate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lit_model.model
    model.eval()
    model.to(device)

    # dataset
    logging.info("Loading dataset from %s", args.filt_path)
    data = load_eval_dataset(args.filt_path)
    cdr_df = load_cdr_info(args.cdr_info_path) if args.cdr_info_path else None
    dataset = EvalDataset(args, data, cdr_df=cdr_df)
    data_loader = dataset.get_dataloader()

    # predict
    preds: List[float] = []
    ground_truth: List[float] = []
    with torch.no_grad():
        logging.info("========== start eval ==========")
        for step, batch in enumerate(data_loader):
            # move tensors only
            batch_tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            labels = batch_tensors["labels"]

            outputs = model(
                wt_ab_inputs_ids=batch_tensors["wt_ab_inputs_ids"],
                wt_ab_inputs_mask=batch_tensors["wt_ab_inputs_mask"],
                mut_ab_inputs_ids=batch_tensors["mut_ab_inputs_ids"],
                mt_ab_inputs_mask=batch_tensors["mt_ab_inputs_mask"],
                wt_ag_inputs_ids=batch_tensors["wt_ag_inputs_ids"],
                wt_ag_inputs_mask=batch_tensors["wt_ag_inputs_mask"],
                mut_ag_inputs_ids=batch_tensors["mut_ag_inputs_ids"],
                mt_ag_inputs_mask=batch_tensors["mt_ag_inputs_mask"],
            )
            out_np = outputs.detach().float().view(-1).cpu().numpy()
            lbl_np = labels.detach().float().view(-1).cpu().numpy()
            preds.extend(out_np.tolist())
            ground_truth.extend(lbl_np.tolist())
        logging.info("=========== end eval ===========")

    # save preds
    logging.info("Saving predictions to %s", args.preds_path)
    preds_arr = np.asarray(preds, dtype=np.float32)
    gt_arr = np.asarray(ground_truth, dtype=np.float32)
    df = pd.DataFrame({"ground_truth": gt_arr, "preds": preds_arr}, columns=["ground_truth", "preds"])
    out_dir = Path(args.preds_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_name = Path(args.ckpt_locate).name.split(".")[0]
    save_path = out_dir / f"{file_name}.csv"
    df.to_csv(save_path, index=False)
    logging.info("Saved: %s", save_path.as_posix())


def main() -> None:
    """
    CLI entry point.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    eval(args)


if __name__ == "__main__":
    main()
