from models import load_progen
import argparse
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import numpy as np
from torch.nn import CrossEntropyLoss

@torch.no_grad()
def eval_pet(model, dataloader, tokenizer):
    model.eval()
    losses = []
    
    for batch in tqdm(dataloader):
        tokens = tokenizer(
                batch['text'],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=1024,
            ).to(model.device)
        
        assert len(dataloader.dataset.few_shots) == 0

        targets = tokens.input_ids.masked_fill(tokens.input_ids == tokenizer.pad_token_id, -100)
            
        attention_mask = tokens.attention_mask
        
        with torch.amp.autocast(device_type=model.device.type, enabled=model.device.type == "cuda"):
            outputs = model(
                input_ids=tokens.input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss.item()

        losses.append(loss)
    return losses

def celoss_with_mask(lm_logits, labels, masks=None):
    loss_fct = CrossEntropyLoss(reduction='none')
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    if masks is not None:
        shift_masks = masks[..., 1:].contiguous()
        loss = (loss * shift_masks.view(-1)).sum() / shift_masks.sum()
    else:
        loss = loss.mean()

    return loss


@torch.no_grad()
def eval_cath(model, dataloader):
    model.eval()
    losses = []
    losses_nan = []
    losses_eos = []
    losses_eos_nan = []

    for batch in tqdm(dataloader): # embeddings, tokens, coords, strs

        assert len(batch[2]) == 1

        for N, coords in enumerate(batch[2]):
            tokens = batch[1][0][N]
            attention_mask = batch[1][1][N]

            nan_mask = torch.isnan(torch.tensor(coords)).sum(dim=(-1, -2))        
            
            # unpadding and adding bos and eos
            tokens_ = tokens[attention_mask > 0]
            attention_mask_ = attention_mask[attention_mask>0]
            tokens_ = torch.cat([torch.tensor([3]), tokens_, torch.tensor([4])]).to(model.device)
            attention_mask_ = torch.cat([torch.tensor([1]), attention_mask_, torch.tensor([1])]).to(model.device)
            
            nan_mask = torch.cat([torch.tensor([0]), nan_mask, torch.tensor([0])])
            
            eos_mask = 1 - attention_mask_.clone()
            eos_mask[..., -1] = 1

            nan_eos_mask = nan_mask.clone()
            nan_eos_mask[..., -1] = 1

            targets = tokens_.clone()

            with torch.amp.autocast(device_type=model.device.type, enabled=model.device.type == "cuda"):
                outputs = model(
                    input_ids=tokens_,
                    attention_mask=attention_mask_,
                    return_dict=True,
                    labels=targets,
                )
            
            loss = outputs.loss.item()
            loss_eos = celoss_with_mask(outputs.logits, targets, (eos_mask==0).to(model.device))
            loss_nan = celoss_with_mask(outputs.logits, targets, (nan_mask==0).to(model.device))
            loss_eos_nan = celoss_with_mask(outputs.logits, targets, (nan_eos_mask==0).to(model.device))

            losses.append(loss)
            losses_eos.append(loss_eos.item())
            losses_nan.append(loss_nan.item())
            losses_eos_nan.append(loss_eos_nan.item())

    print(f'loss: {np.mean(losses)}, eos loss: {np.mean(losses_eos)} nan losses: {np.mean(losses_nan)} eos nan loss: {np.mean(losses_eos_nan)}')
    return losses, losses_eos, losses_nan, losses_eos_nan

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a ProGen checkpoint on optional CATH/PET data.")
    parser.add_argument("--model-path", required=True, help="Local ProGen checkpoint directory.")
    parser.add_argument("--pet-data-folder", help="Optional PET fasta folder.")
    parser.add_argument("--skip-cath", action="store_true", help="Skip CATH shard evaluation.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    tokenizer, progen_model = load_progen(args.model_path)
    progen_model.to(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # losses = check_lora(progen_model)
    # print(np.mean(losses))

    if not args.skip_cath:
        from tools.shard_dataloader import get_pifold_dataset
        from tools.data_utils import pi_args, pi_test_args, pi_valid_args

        eval_cath(progen_model, get_pifold_dataset(pi_valid_args, tokenizer))
        eval_cath(progen_model, get_pifold_dataset(pi_test_args, tokenizer))
        eval_cath(progen_model, get_pifold_dataset(pi_args, tokenizer))

    if args.pet_data_folder:
        from tools.evaluation_dataloader import ProteinSeqLoader

        test_loader = DataLoader(
            ProteinSeqLoader.from_folder(args.pet_data_folder, max_len=1024),
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=2,
        )
        losses = eval_pet(progen_model, test_loader, tokenizer)
        print(np.mean(losses))
