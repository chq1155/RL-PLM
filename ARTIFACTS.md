# RL-PLM External Artifact Manifest

Large datasets and checkpoints are distributed outside git. This manifest records the expected local files for the maintained AMP, antibody, and kinase workflows so a user can verify a downloaded artifact bundle before running experiments.

Download root: https://drive.google.com/drive/folders/1_B0OEdwxUbMbncftXQypsoLvuMgIxrxu?usp=sharing

The checksums below are computed from the artifact files that were present in repository commit `ceb3a7431911342ce3b9f9b0237fb039ee55549c` before the large files were removed from git.

## AMP Design

| Local path | Bytes | SHA256 | Description |
| --- | ---: | --- | --- |
| `amp_design/best_new_4.pth` | 166577 | `ed910eb75a1401e8991e5e8b2d52ecc04312bf04f36b20dbea46945ff1b6bbdc` | AMP reward classifier weights used by `--classifier-checkpoint`. |

## Antibody Mutation

### Sequence-Identity Data

| Local path | Bytes | SHA256 | Description |
| --- | ---: | --- | --- |
| `antibody_mutation/data/identity_data/csv/AB1101.csv` | 1561909 | `0afc90b92f78902148e814d92f173f3a203a1fd60026333e383e126be4a7f449` | Full AB1101 sequence-identity table. |
| `antibody_mutation/data/identity_data/csv/AB645.csv` | 1006521 | `ae4d6666f39ede6f7426d7c841a60dbbfa6fa3d6916fefa0a282e226756ade6d` | Full AB645 sequence-identity table. |
| `antibody_mutation/data/identity_data/csv/S1131.csv` | 763063 | `8d454a13ad1070816ed956224d81e2f8cd95bce985aa56d794700afeaec74fe8` | Full S1131 sequence-identity table. |
| `antibody_mutation/data/identity_data/csv_AB1101/AB1101_train.csv` | 1205729 | `37a89226d0d3c54de90e9f2c75d9f3fe98c7a1daf69f2957cfd275b82d20b2d8` | AB1101 training split. |
| `antibody_mutation/data/identity_data/csv_AB1101/AB1101_test.csv` | 358701 | `bda95356d6175a7928bc023de010cec79135b87fc3c3c944355248812e18b30a` | AB1101 test split. |
| `antibody_mutation/data/identity_data/csv_AB645/AB645_train.csv` | 867310 | `6f0f3295da9580f503358a96a63b8607f3c0799858be4be430a18461f001eecc` | AB645 training split. |
| `antibody_mutation/data/identity_data/csv_AB645/AB645_test.csv` | 139670 | `156899c4ff13ae0a766ce13a740d0b55db542290934bf0ae2630f159d48e142f` | AB645 test split. |
| `antibody_mutation/data/identity_data/csv_S1131/S1131_train.csv` | 628243 | `f7ddf8f00cb85182c2900fbfa330a52fcb05d44358defa66f31e33b073762cf1` | S1131 training split. |
| `antibody_mutation/data/identity_data/csv_S1131/S1131_test.csv` | 134902 | `9bb609009d4375f69a0711f2f8026a7a4a3a3367b882cf249d6b0725c0bf8660` | S1131 test split. |

### MMseqs Split Metadata

| Local path | Bytes | SHA256 | Description |
| --- | ---: | --- | --- |
| `antibody_mutation/data/identity_data/mmseqs_file/AB1101_file/AB1101_clu.tsv` | 341 | `6fdccf411294b914d3d6e671c6ccd2f2d7ffac072715ff90025538ce19356ea4` | AB1101 MMseqs cluster assignments. |
| `antibody_mutation/data/identity_data/mmseqs_file/AB1101_file/AB1101_train.csv` | 124 | `5abdbf77a8fe30964a09690a021fce5536f5f426594be2b0f655874b2669b9bc` | AB1101 MMseqs training cluster IDs. |
| `antibody_mutation/data/identity_data/mmseqs_file/AB1101_file/AB1101_test.csv` | 59 | `61d34e19594e8200234e7b79463951194785b718b81d2f3900177fd7da8099c4` | AB1101 MMseqs test cluster IDs. |
| `antibody_mutation/data/identity_data/mmseqs_file/AB645_file/AB645_clu.tsv` | 311 | `ae56b2b180523031dc58b17bb08b634f92697795fc56f94ad1ba13832d55ebb7` | AB645 MMseqs cluster assignments. |
| `antibody_mutation/data/identity_data/mmseqs_file/AB645_file/AB645_train.csv` | 136 | `79dfb7264d8556933d4522ba51e901e3f7e5cd47b8c98bdb40c4c646b11e182e` | AB645 MMseqs training cluster IDs. |
| `antibody_mutation/data/identity_data/mmseqs_file/AB645_file/AB645_test.csv` | 32 | `cf4fa4fd286bdd516036ee723a21971a983870be2bcd2824df9627d1241576af` | AB645 MMseqs test cluster IDs. |
| `antibody_mutation/data/identity_data/mmseqs_file/S1131_file/S1131_clu.tsv` | 1120 | `44ffae66d15444b25b064d0a493ca94a39dacc48380167aaecec1f732acb546c` | S1131 MMseqs cluster assignments. |
| `antibody_mutation/data/identity_data/mmseqs_file/S1131_file/S1131_train.csv` | 414 | `cc9fc88c2a638417f17f7d16a1ae768d1a67d676b0df07dce8fa14c2bad1ae68` | S1131 MMseqs training cluster IDs. |
| `antibody_mutation/data/identity_data/mmseqs_file/S1131_file/S1131_test.csv` | 154 | `0b3cee1d15700ca7dd5e370d588d6a1cf51ad4733503b48a4257d002352a2d21` | S1131 MMseqs test cluster IDs. |

### SigMul Data

| Local path | Bytes | SHA256 | Description |
| --- | ---: | --- | --- |
| `antibody_mutation/data/sigmul_data/AB1101_single.csv` | 1051171 | `5c886b6961bb61c2780216a81eaf86c57b5aefef0307264c545efb4a723e7100` | AB1101 single-mutation reward-model training data. |
| `antibody_mutation/data/sigmul_data/AB1101_multiple.csv` | 513259 | `eb5b38eaffbd081b2f6cb9dcecff7386ebb2c0f8d4fbb66281919c5483a2fc96` | AB1101 multi-mutation evaluation data. |
| `antibody_mutation/data/sigmul_data/AB1101_multiple_cdr.csv` | 220205 | `10e1d29e17f30178a2fedd0760a4b2c498b59597ccb7fb6469f284abdf1002c7` | AB1101 multi-mutation CDR subset. |
| `antibody_mutation/data/sigmul_data/AB1101_multiple_cdr_balance.csv` | 1400260 | `8783a8b7555ee079cd87650070d2112cb344efa44ac37409ea93f121f53f64ce` | AB1101 balanced CDR multi-mutation data. |
| `antibody_mutation/data/sigmul_data/AB1101_multiple_cdr_balance_train.csv` | 1161780 | `38d3a4160a0e4782955686442b7a27f2d9db1b9b2c0342b6e7fb16ea428b4773` | AB1101 balanced CDR mutation-policy training data. |
| `antibody_mutation/data/sigmul_data/AB1101_multiple_cdr_balance_test.csv` | 195863 | `9bb4761bfb19514ea77dc62bd15f9847428f1fc843c3afe0f623330c82a9f27f` | AB1101 balanced CDR evaluation data. |
| `antibody_mutation/data/sigmul_data/cdr_info.csv` | 691 | `a3dc4d83643c8a51b484276581c0a2e9fafc460463448e6e3073db4a6ef098c8` | CDR fragment annotations. |

## Kinase Mutation

| Local path | Bytes | SHA256 | Description |
| --- | ---: | --- | --- |
| `kinase_mutation/data/PhoQ.csv` | 1681962 | `13288b6efa291fd0a205b5abc5267f3b74d73d5991123df07dbc4862b628a80c` | PhoQ variant fitness table. |
| `kinase_mutation/data/PhoQ.xlsx` | 2029029 | `a64a085629412166e8dbfa5e67ec414103c0f757dedec7d04e0973c761cf48f6` | PhoQ variant fitness workbook. |
| `kinase_mutation/data/train_init_sequences.csv` | 8003 | `3508aa817de733ef9d925a0e9ca530d4afc3db14965a4f58c92fec581e3adc78` | Initial training sequences for PhoQ RL. |
| `kinase_mutation/data/test_init_sequences.csv` | 5779 | `60f78cf719ec71e9f42a45bc39ce2fd09d7b1dd8de34d58688f5e8b2842c0afa` | Initial test sequences for PhoQ evaluation. |
| `kinase_mutation/data/train_set.csv` | 1354197 | `8aa47d56b18ee549597350dcf80c13ce6ca1cb880c0b7fabaa5c423f689d44b4` | PhoQ train set export. |
| `kinase_mutation/data/test_set.csv` | 305225 | `34127a721cf4631a5330bb8cb923597b66898cd23c4b94d1a6e0d13db4a8ed59` | PhoQ test set export. |
| `kinase_mutation/esm_8m/config.json` | 759 | `1df4b28f9e45b5ae6889ef999a50bd35e4b80d47819f0bd8f93d9ed289923bca` | ESM-8M Hugging Face config. |
| `kinase_mutation/esm_8m/pytorch_model.bin` | 31403977 | `06b87b1383f44bfc0ae22d5aff6e10ec37130a79c5ecc56f1006448f8ae5f50c` | ESM-8M model weights. |
| `kinase_mutation/esm_8m/tokenizer_config.json` | 108 | `f8487f6f24410837aff3a2d5d5cfc3150d5fac57930b6a67ff20f14dd875ed98` | ESM-8M tokenizer config. |
| `kinase_mutation/esm_8m/special_tokens_map.json` | 125 | `3aedcd4211c0d43aec4e607ff60a63255f3174ead795e997350f09a5f8cd9ee1` | ESM-8M special-token map. |
| `kinase_mutation/esm_8m/vocab.txt` | 93 | `0b82cc0a7c7cf9e567b1e5892d793285b9fbae822c964ca48696f7db44598e03` | ESM-8M vocabulary. |

## Upstream And Reproduced Model Artifacts

Some workflows depend on upstream model releases or training outputs that were not recoverable as tracked files in `HEAD`. Do not leave these implicit in an archival run: record the provider, model identifier or training command, byte size, and SHA256 in the run directory before training or evaluation.

| Artifact | Expected source | How it is used |
| --- | --- | --- |
| AMP base policy | Paper setting: Amphion-SFT / ProGen2-xlarge (6.4B) converted to the local Hugging Face ProGen format with `amp_design/progen2hf/`. | Passed to AMP DPO/GRPO with `--base-model-path`; passed to AMP sampling with `--base-model-path` when `--model-path` is a PEFT adapter. |
| ProGen2 tokenizer | Tokenizer paired with the AMP base policy conversion. | Passed to AMP scripts with `--tokenizer-path`. |
| Antibody ESM2-650M encoder | Hugging Face model id `facebook/esm2_t33_650M_UR50D` or the same files mirrored under `antibody_mutation/model/esm2_650m/`. | Passed to antibody reward-model training/evaluation with `--model_locate`. |
| ProtAttBA reward checkpoint | Reproduce with `antibody_mutation/trainer_sigmul.py` using the SigMul data above, or use the checkpoint from the external RL-PLM artifact release and record its SHA256. | Passed to antibody RL/evaluation scripts with `--checkpoint_path` or `--ckpt_locate`. |

## Verification

After downloading artifacts, generate local hashes and compare them with the release values above:

```bash
find amp_design antibody_mutation/data kinase_mutation/data kinase_mutation/esm_8m \
  -type f -print0 | sort -z | xargs -0 sha256sum > artifact_sha256.txt
```
