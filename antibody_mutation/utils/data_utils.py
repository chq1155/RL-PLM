import pandas as pd
import numpy as np
import os, shutil
from collections import defaultdict
from sklearn.model_selection import train_test_split


def cat_seq(seq1: str, seq2: str):
    if isinstance(seq1, float):
        return seq2
    if isinstance(seq2, float):
        return seq1
    return seq1 + seq2

def rm_abnormal_data(df):
    total_len = len(df)
    df = df[df["ddG"] != 8.0]
    print(f"del {total_len - len(df)} records")
    return df

def read_AB_csv(file_path, rm_abnormal):
    csv = pd.read_csv(
        file_path,
        usecols=[
            "PDB",
            "Mutation",
            "antibody_light_seq",
            "antibody_heavy_seq",
            "antigen_a_seq",
            "antigen_b_seq",
            "antibody_light_seq_mut",
            "antibody_heavy_seq_mut",
            "antigen_a_seq_mut",
            "antigen_b_seq_mut",
            "ddG",
        ],
    )
    if rm_abnormal:
        csv = rm_abnormal_data(csv)
    names = csv["PDB"].tolist()
    abls = csv["antibody_light_seq"].tolist()
    abhs = csv["antibody_heavy_seq"].tolist()
    agas = csv["antigen_a_seq"].tolist()
    agbs = csv["antigen_b_seq"].tolist()
    abls_m = csv["antibody_light_seq_mut"].tolist()
    abhs_m = csv["antibody_heavy_seq_mut"].tolist()
    agas_m = csv["antigen_a_seq_mut"].tolist()
    agbs_m = csv["antigen_b_seq_mut"].tolist()
    labels = csv["ddG"].tolist()
    return names, abls, abhs, agas, agbs, abls_m, abhs_m, agas_m, agbs_m, labels


def read_s1131_csv(file_path):
    csv = pd.read_csv(
        file_path, usecols=["PDB", "mutation", "a", "b", "a_mut", "b_mut", "ddG"]
    )
    names = csv["PDB"].tolist()
    ab = csv["a"].tolist()
    ag = csv["b"].tolist()
    ab_m = csv["a_mut"].tolist()
    ag_m = csv["b_mut"].tolist()
    labels = csv["ddG"].tolist()
    return names, ab, ag, ab_m, ag_m, labels


def get_AB_pair_data(file_path, rm_abnormal):
    names, abls, abhs, agas, agbs, abls_m, abhs_m, agas_m, agbs_m, labels = read_AB_csv(
        file_path, rm_abnormal
    )
    assert (
        len(abls)
        == len(abhs)
        == len(agas)
        == len(agbs)
        == len(abls_m)
        == len(abhs_m)
        == len(agas_m)
        == len(agbs_m)
        == len(labels)
    ), "data length not equal"
    ab_hl_chains = []
    ag_ab_chains = []
    mt_ab_hl_chains = []
    mt_ag_ab_chains = []
    pdb_ids = []
    for i in range(len(names)):
        ab_hl_chains.append(cat_seq(abls[i], abhs[i]))
        ag_ab_chains.append(cat_seq(agas[i], agbs[i]))
        mt_ab_hl_chains.append(cat_seq(abls_m[i], abhs_m[i]))
        mt_ag_ab_chains.append(cat_seq(agas_m[i], agbs_m[i]))
        pdb_ids.append(names[i])
    return ab_hl_chains, ag_ab_chains, mt_ab_hl_chains, mt_ag_ab_chains, labels, pdb_ids


def get_s1131_data(file_path):
    names, ab, ag, ab_m, ag_m, labels = read_s1131_csv(file_path)
    labels = np.array(labels)  # [B,]
    return ab, ag, ab_m, ag_m, labels


def get_train_val_data(data, val_ratio=0.1, random_seed=42):
    """
    split data into train and val data.
    """
    ab_hl_chains, ag_ab_chains, mt_ab_hl_chains, mt_ag_ab_chains, labels = data
    # split into train and val
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size=val_ratio,
        random_state=random_seed,
    )
    train_data = (
        [ab_hl_chains[i] for i in train_idx],
        [ag_ab_chains[i] for i in train_idx],
        [mt_ab_hl_chains[i] for i in train_idx],
        [mt_ag_ab_chains[i] for i in train_idx],
        [labels[i] for i in train_idx]
    )
    
    val_data = (
        [ab_hl_chains[i] for i in val_idx],
        [ag_ab_chains[i] for i in val_idx],
        [mt_ab_hl_chains[i] for i in val_idx],
        [mt_ag_ab_chains[i] for i in val_idx],
        [labels[i] for i in val_idx]
    )
    return train_data, val_data


def load_data(dataset_name, data_folder, val_ratio=0.1, random_seed=42, rm_abnormal=True):
    if dataset_name == "S1131":
        train_file = os.path.join(data_folder, "S1131_train.csv")
        test_file = os.path.join(data_folder, "S1131_test.csv")
        ab_hl_chains, ag_ab_chains, mt_ab_hl_chains, mt_ag_ab_chains, labels = (
            get_s1131_data(train_file)
        )
        data = (ab_hl_chains, ag_ab_chains, mt_ab_hl_chains, mt_ag_ab_chains, labels)
        train_data, val_data = get_train_val_data(data, val_ratio=val_ratio, random_seed=random_seed)
        
        (
            test_ab_hl_chains,
            test_ag_ab_chains,
            test_mt_ab_hl_chains,
            test_mt_ag_ab_chains,
            test_labels,
        ) = get_s1131_data(test_file)

    elif "AB" in dataset_name:
        train_file = os.path.join(data_folder, f"{dataset_name}_train.csv")
        test_file = os.path.join(data_folder, f"{dataset_name}_test.csv")
        ab_hl_chains, ag_ab_chains, mt_ab_hl_chains, mt_ag_ab_chains, labels = (
            get_AB_pair_data(train_file, rm_abnormal=rm_abnormal)
        )
        data = (ab_hl_chains, ag_ab_chains, mt_ab_hl_chains, mt_ag_ab_chains, labels)
        train_data, val_data = get_train_val_data(data, val_ratio=val_ratio, random_seed=random_seed)
        (
            test_ab_hl_chains,
            test_ag_ab_chains,
            test_mt_ab_hl_chains,
            test_mt_ag_ab_chains,
            test_labels,
        ) = get_AB_pair_data(test_file, rm_abnormal=rm_abnormal)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    assert (
        len(test_ab_hl_chains)
        == len(test_ag_ab_chains)
        == len(test_mt_ab_hl_chains)
        == len(test_mt_ag_ab_chains)
        == len(test_labels)
    ), "dataset len not equal"
    test_data = (
        test_ab_hl_chains,
        test_ag_ab_chains,
        test_mt_ab_hl_chains,
        test_mt_ag_ab_chains,
        test_labels
    )
    return train_data, val_data, test_data

