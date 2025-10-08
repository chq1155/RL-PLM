"""
"""

import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from transformers import AutoTokenizer


class SeqDataset(Dataset):
    def __init__(
        self,
        wt_ab_seqs: List[str],
        mt_ab_seqs: List[str],
        wt_ag_seqs: List[str],
        mt_ag_seqs: List[str],
        labels: List[float],
        pdb_ids: List[str],
    ):
        self.wt_ab_seqs = wt_ab_seqs
        self.mt_ab_seqs = mt_ab_seqs
        self.wt_ag_seqs = wt_ag_seqs
        self.mt_ag_seqs = mt_ag_seqs
        self.labels = labels
        self.pdb_ids = pdb_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.wt_ab_seqs[idx],
            self.mt_ab_seqs[idx],
            self.wt_ag_seqs[idx],
            self.mt_ag_seqs[idx],
            self.labels[idx],
            self.pdb_ids[idx],
        )


class WrapperDataset(Dataset):
    def __init__(self, args, train_data, val_data, test_data):
        super(WrapperDataset, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.model_locate = args.model_locate
        self.max_length = args.max_length
        self.truncation = True if self.max_length else False

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_locate)
        self.train_dataset, self.val_dataset, self.test_dataset = self.get_dataset(
            train_data, val_data, test_data
        )

    def get_dataset(self, train_data, val_data, test_data):

        (
            train_wt_ab_seqs,
            train_wt_ag_seqs,
            train_mt_ab_seqs,
            train_mt_ag_seqs,
            train_labels,
        ) = train_data

        (
            val_wt_ab_seqs,
            val_wt_ag_seqs,
            val_mt_ab_seqs,
            val_mt_ag_seqs,
            val_labels,
        ) = val_data
        """
            test_data = (
                test_ab_hl_chains,
                test_ag_ab_chains,
                test_mt_ab_hl_chains,
                test_mt_ag_ab_chains,
                test_labels
            )
        """
        (
            test_wt_ab_seqs,
            test_wt_ag_seqs,
            test_mt_ab_seqs,
            test_mt_ag_seqs,
            test_labels,
        ) = test_data
        
        # note the order of data tuple
        """
            wt_ab_seqs=ab_hl_chains,
            mt_ab_seqs=mt_ab_hl_chains,
            wt_ag_seqs=ag_ab_chains,
            mt_ag_seqs=mt_ag_ab_chains,
            labels=labels,
        """
        train_dataset = SeqDataset(
            wt_ab_seqs=train_wt_ab_seqs,
            mt_ab_seqs=train_mt_ab_seqs,
            wt_ag_seqs=train_wt_ag_seqs,
            mt_ag_seqs=train_mt_ag_seqs,
            labels=train_labels,
        )

        val_dataset = SeqDataset(
            wt_ab_seqs=val_wt_ab_seqs,
            mt_ab_seqs=val_mt_ab_seqs,
            wt_ag_seqs=val_wt_ag_seqs,
            mt_ag_seqs=val_mt_ag_seqs,
            labels=val_labels,
        )

        test_dataset = SeqDataset(
            wt_ab_seqs=test_wt_ab_seqs,
            mt_ab_seqs=test_mt_ab_seqs,
            wt_ag_seqs=test_wt_ag_seqs,
            mt_ag_seqs=test_mt_ag_seqs,
            labels=test_labels,
        )
        return train_dataset, val_dataset, test_dataset

    def collator_fn(self, batch):
        """
            self.wt_ab_seqs[idx],
            self.mt_ab_seqs[idx],
            self.wt_ag_seqs[idx],
            self.mt_ag_seqs[idx],
            self.labels[idx],
        """
        wt_ab_seqs, mt_ab_seqs, wt_ag_seqs, mt_ag_seqs, labels = zip(*batch)

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

        batch = {
            "wt_ab_inputs_ids": wt_ab_inputs["input_ids"],
            "wt_ab_inputs_mask": wt_ab_inputs["attention_mask"],
            "mut_ab_inputs_ids": mt_ab_inputs["input_ids"],
            "mt_ab_inputs_mask": mt_ab_inputs["attention_mask"],
            "wt_ag_inputs_ids": wt_ag_inputs["input_ids"],
            "wt_ag_inputs_mask": wt_ag_inputs["attention_mask"],
            "mut_ag_inputs_ids": mt_ag_inputs["input_ids"],
            "mt_ag_inputs_mask": mt_ag_inputs["attention_mask"],
            "labels": labels,
        }

        return batch

    def get_train_loader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collator_fn,
        )

    def get_val_loader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collator_fn,
        )

    def get_test_loader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator_fn,
        )
