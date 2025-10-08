import torch
from torch.utils.data.dataloader import Dataset, DataLoader
from tqdm import tqdm
from glob import glob
import pickle
import random
import numpy as np
import json


class PEFT_Dataset(Dataset):
    def __init__(self, seqs, tokenizer, embeddings=None, max_len=100, few_shot=0, shuffle_seed=None) -> None:
        super().__init__()
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(seqs)
        self.seqs = seqs
        self.embeddings = embeddings # save for Llava and blip
        self.max_len = max_len
        
        self.tokens = []
        self.few_shots = []
        self.seq_loc = []
        self.tokenizer = tokenizer

        for seq in self.seqs:
            if len(self.few_shots) < few_shot:
                self.few_shots.append('1' + seq + '2')
            
            else:
                seq = '1'+seq+'2'
                ## construct prompt ##
                prompt = ''.join(self.few_shots)
                self.seq_loc.append(len(prompt))
                self.tokens.append(prompt + seq)
                ## update prompt for next seq ##
            
                if len(self.few_shots) > 0:
                    self.few_shots.pop(0)
                    self.few_shots.append(seq)

        print(f'build dataset with {len(self.seqs)} proteins.')

    @classmethod
    def PET_folder(cls, path, tokenizer, max_len=100, split=[0.9, 0.1], shuffle_seed=None):
        seqs = [[] for i in range(len(split))]
        embeddings = []
        print(f'load dataset from {path}.')

        data_path = sorted(glob(path+'/*'))

        bins = np.cumsum(split)
        bins = bins / np.max(bins)

        print(f'split dataset with {bins}')

        for index, data in tqdm(enumerate(sorted(data_path))):
            with open(data, 'rb') as f:
                if data.endswith('fasta'):
                    data = f.readlines()
                    seq = data[1].rstrip().decode("utf-8")
                else:
                    continue
                # if len(seq) >= max_len-2:
                #     continue
            idx = np.where((bins - np.random.rand(1)) > 0)[0][0]
            seqs[idx].append(seq)
            # if index > 1000:
            #     break
        datasets = [cls(x, tokenizer, embeddings, max_len=max_len, shuffle_seed=shuffle_seed) for x in seqs]
        return datasets

    @classmethod
    def cath_web_data(cls, iterator, tokenizer, max_len=1024, shuffle_seed=None):
        seqs = []
        embeddings = []

        for batch in tqdm(iterator): # embeddings, tokens, coords, strs

            assert len(batch[-1]) == 1
            seqs.append(batch[-1][0])
        
        return cls(seqs, tokenizer, embeddings, max_len, shuffle_seed=shuffle_seed)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        tokens = self.tokens[index]
        return {
            'text': tokens
        }
    
    def collate(self, raw_batch):
        texts = []
        for batch in raw_batch:
            text = batch['text']
            # random crop
            if len(text) > self.max_len:
                crop_idx = random.randint(0, len(text) - self.max_len)
                text = text[crop_idx:]                
                text = '1' + text
            texts.append(text)
        batch = self.tokenizer(texts, padding='longest', return_tensors='pt')
        labels = batch.input_ids.masked_fill(batch.input_ids == self.tokenizer.pad_token_id, -100)
        del batch["token_type_ids"]
        batch["labels"] = labels
        
        return batch
