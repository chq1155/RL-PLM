import torch
from torch.utils.data.dataloader import Dataset, DataLoader
from tqdm import tqdm
from glob import glob
import pickle
import random
import numpy as np
import json


class ProteinSeqLoader(Dataset):
    def __init__(self, seqs, embeddings=None, max_len=100, few_shot=0) -> None:
        super().__init__()
        self.seqs = seqs
        self.embeddings = embeddings # save for Llava and blip
        self.max_len = max_len
        
        self.tokens = []
        self.few_shots = []
        self.seq_loc = []

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
    def from_folder(cls, path, max_len=100):
        seqs = []
        embeddings = []
        print(f'load dataset from {path}.')

        data_path = sorted(glob(path+'/*'))

        for index, data in tqdm(enumerate(sorted(data_path))):
            with open(data, 'rb') as f:

                if data.endswith('pkl'):
                    data = pickle.load(f)
                    seq = data['seq'] if data['seq'] is not None else print(f'seq err at {index}')
                    coord = data['coord']
                elif data.endswith('fasta'):
                    data = f.readlines()
                    seq = data[1].rstrip().decode("utf-8")
                
                if len(seq) >= max_len-2:
                    continue
            
            seqs.append(seq)
        return cls(seqs, embeddings, max_len=max_len)

    @classmethod
    def from_json(cls, path, max_len=27, num_sample=1000):
        seqs = []
        embeddings = []
        print(f'load dataset from {path}.')
        
        with open(path, 'rb') as fin:
            data = json.load(fin)
        
        for index, sample in tqdm(enumerate(data)):
            if len(sample['text']) >= max_len-2:
                continue
            seqs.append(sample['text'])
            if index >= num_sample:
                break
        
        return cls(seqs, embeddings, max_len=max_len)
            
    # def from_web(cls, args, tokenizer, ):
    #     web_data = get_pifold_dataset(args, tokenizer)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        # padding to max_len
        tokens = self.tokens[index]

        return {
            'text': tokens
        }
    
