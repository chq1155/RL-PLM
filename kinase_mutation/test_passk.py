import argparse
import torch
import gym
import numpy as np
import random
import time
import csv
import pandas as pd
import sys, os
from transformers import AutoTokenizer
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3.common import logger

collected_seqs_list = []
path_96 = 'data/test_init_sequences.csv'
AMINO_ACIDS = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
pos = {0:96, 1:97, 2:100}
re_pos = {96:0,97:1,100:2}

def find_AA(original_sequence, position, ori_aa, lm_model, tokenizer, temperature=1.0):
    masked_sequence = torch.unsqueeze(original_sequence, dim=0).clone()
    masked_sequence[0, position] = tokenizer.mask_token_id
    with torch.no_grad():
        outputs = lm_model(input_ids=masked_sequence)
        logits = outputs.logits[0, position]
        ori_token = original_sequence[position]
        
        logits = logits / temperature 
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        valid_aa_mask = torch.zeros_like(probs)
        for aa in AMINO_ACIDS:
            aa_token = tokenizer.encode(aa)[1]  
            if aa_token != ori_token: 
                valid_aa_mask[aa_token] = 1
                
        masked_probs = probs * valid_aa_mask
        
        if masked_probs.sum() > 0:
            masked_probs = masked_probs / masked_probs.sum()
            try:
                sampled_token = torch.multinomial(masked_probs, num_samples=1)
                predicted_aa = tokenizer.decode([sampled_token.item()])
                if predicted_aa in AMINO_ACIDS:  
                    return [predicted_aa]
            except RuntimeError:
                pass 
        
        sorted_indices = torch.argsort(probs, descending=True)
        for idx in sorted_indices:
            if idx == ori_token:
                continue
            predicted_aa = tokenizer.decode([idx.item()])
            if predicted_aa in AMINO_ACIDS:
                return [predicted_aa]

class PhoQEnv(gym.Env):
    def __init__(self,
                 action_space: gym.spaces,
                 observation_space: gym.spaces,
                 args: dict,
                 max_len: int = 191,
                 ):
        super(PhoQEnv, self).__init__()
        self.action_space = action_space
        self.observation_space = observation_space

        self.reward = float("-inf")
        self.reward_list = []
        self.max_step = args.max_step
        self.k = [0, 6]
        self.len_step = 0
        self.max_len = max_len
        
        datas = pd.read_csv(path_96, names=['AACombo', 'Fitness','FitnessGroup'], header=0)
        self.PhoQ = []
        self.PhoQ_fitness = []
        self.PhoQ_protein = []
        
        for i in range(len(datas)):
            protein = "SYMVWSWFIYVLSANLLLVIPLLWVAAWWSLRPIEALAKEVRELEEHNRELLNPATTRELTSLVRNLNRLLKSERERYDKYRTTLTDLTHSLKTPL__LQ__LRSLRSEKMSVSDAEPVMLEQISRISQQIGYYLHRASMRGGTLLSRELHPVAPLLDNLTSALNKVYQRKGVNISLDISPEISFVGEQ"
            protein = protein.replace("__", datas["AACombo"][i][0:2], 1)
            protein = protein.replace("__", datas["AACombo"][i][2:4], 1)
            tokens = tokenizer(protein, return_tensors="pt").to(device)
            self.PhoQ_protein.append(datas["AACombo"][i])
            self.PhoQ.append(tokens['input_ids'].squeeze(0).cpu().numpy())
            self.PhoQ_fitness.append(float(datas['Fitness'][i]))

        self.current_seq_index = 0
        self.total_seqs = len(self.PhoQ)

    def set_wildtype(self, wildtype_index):
        if wildtype_index >= self.total_seqs:
            return False
            
        self.current_seq_index = wildtype_index
        self.initial_seq = self.PhoQ[wildtype_index]
        self.initial_fitness = self.PhoQ_fitness[wildtype_index]
        self.target_fitness = self.initial_fitness 
        
        
        return True

    def reset(self):
        self.state = self.initial_seq
        self.len_step = 0
        self.reward_list = []
        self.k = [0, 6]
        return self.state

    def check_terminal(self, score, string, have):
        if score >= self.target_fitness or self.len_step >= self.max_step:
            return True
        else:
            return False

    def _get_reward(self, seq):
        seq = torch.from_numpy(seq).unsqueeze(0).to(device)
        protein = [tokenizer.decode(r) for r in seq][0][6:383][::2]
        string = protein[96] + protein[97] + protein[100] + protein[101]
        
        flag = False
        for i in string:
            if i not in AMINO_ACIDS:
                flag = True
                
        have = False
        if string in fitness.keys():
            score_truth = fitness[string]
            have = True
        elif flag:
            score_truth = -100
        else:
            score_truth = -1
            
        self.reward = score_truth
        terminal = self.check_terminal(self.reward, string, have)
        
        if terminal and not flag:
            collected_seqs_list.append(string)
            
        return self.reward, terminal, score_truth

    def _edit_sequence(self, seq, actions):
        protein = seq.copy()
        position = actions[0]+1
        protein[position] = actions[1]
        return protein

    
    def step(self, actions: torch.Tensor):
        new_seqs = self._edit_sequence(self.state, actions)

        self.len_step += 1
        term_reward, terminal, score_truth = self._get_reward(new_seqs)
        self.reward_list.append(term_reward)
        if len(self.reward_list)>=2 and self.reward_list[-1] > self.reward_list[-2]:
            self.k[1] = max(self.k[1] - 1, 18)
            self.k[0] = max(self.k[0] - 1, 0)
        if len(self.reward_list)>=3 and self.reward_list[-2] >= self.reward_list[-1] and self.reward_list[-3] >= self.reward_list[-2]:
            self.k[0] = min(self.k[0] + 1, 2)
            self.k[1] = min(self.k[1] + 1, 20)

        info = {}
        info['terminal'] = str(terminal)
        info['action'] = ",".join([str(actions[i]) for i in range(2)])
        info['old_seq'] = tokenizer.decode(self.state[0])
        info['new_seq'] = tokenizer.decode(new_seqs[0])
        info['init_seq'] = self.initial_seq if self.initial_seq is not None else "None"
        info['rewards'] = float(term_reward)
        info['score_truth'] = float(score_truth)
        info['k'] = self.k
        logger.record("state/reward", term_reward)
        logger.record("state/fitness", score_truth)

        self.state = new_seqs
        return self.state, term_reward, terminal, info

def calculate_passk(successful_sequences, total_sequences, k_values=[2, 4, 8, 16, 32, 64,128]):
    passk_results = {}
    
    for k in k_values:
        if total_sequences == 0:
            passk_results[f'pass@{k}'] = 0.0
        else:
            passk_results[f'pass@{k}'] = min(successful_sequences / total_sequences, 1.0)
    
    return passk_results

def run_sampling_for_wildtype(env, model, policy, wildtype_index, max_attempts=128):
    if not env.set_wildtype(wildtype_index):
        return None
    
    wildtype_info = {
        'index': wildtype_index,
        'protein': env.PhoQ_protein[wildtype_index],
        'initial_fitness': env.initial_fitness,
        'target_fitness': env.target_fitness,
        'attempts_needed': None,
        'best_fitness_found': None,
        'successful_fitness': None
    }
    
    for attempt in range(max_attempts):
        np.random.seed(int(time.time() * 1000) % 1000000 + attempt)
        random.seed(int(time.time() * 1000) % 1000000 + attempt)
        
        obs = env.reset()
        
        while True:
            action_pos, _ = model.predict(np.expand_dims(obs, axis=0), deterministic=False)
            action_pos = action_pos[0]
            

            position = pos[int(action_pos)]+1
            ori_token = env.state[position]
            ori_aa = ''
            for j in AMINO_ACIDS:
                if policy.AA2token[j] == ori_token:
                    ori_aa = j
                    break
                    
            obs_tensor = torch.from_numpy(np.expand_dims(env.state, axis=0)).long().to(device)

            
            AA_k = find_AA(obs_tensor[0], position, ori_aa, policy.lm_model, policy.tokenizer)
            predicted_token_id = policy.AA2token[AA_k[0]]
            position=position-1 
            action = torch.tensor([position, predicted_token_id])
            obs, reward, done, info = env.step(action)
            
            if done:
                if wildtype_info['best_fitness_found'] is None or reward > wildtype_info['best_fitness_found']:
                    wildtype_info['best_fitness_found'] = reward
                
                if reward > env.target_fitness:
                    wildtype_info['attempts_needed'] = attempt + 1
                    wildtype_info['successful_fitness'] = reward
                    return wildtype_info
                break
        
    wildtype_info['attempts_needed'] = max_attempts
    
    return wildtype_info

if __name__ == '__main__':
    from stable_baselines3.ppo import PPO
    from stable_baselines3.dpo import DPO
    from ESM_PhoQ import PolicyNet
    import warnings
    from collections import Counter

    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help="path to save results", default="./checkpoints")
    parser.add_argument('--gamma', type=float, default=0.99, help="discount_factor")
    parser.add_argument('--steps', type=int, default=500000, help="total time steps")
    parser.add_argument('--ent_coef', type=float, default=0.2, help="encourage exploration")
    parser.add_argument('--clip', type=float, default=0.2, help="")
    parser.add_argument('--max_len', type=int, default=191)
    parser.add_argument('--num_envs', type=int, default=10, help="number of environments")
    parser.add_argument('--n_steps', type=int, default=20, help="number of roll out steps")
    parser.add_argument('--max_step', type=int, default=3, help="maximum number of steps")
    parser.add_argument('--score_stop_criteria', type=float, default=70, help="stop_criteria")

    args = parser.parse_args()
    path = args.path
    t1 = time.time()
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
    action_space = gym.spaces.multi_discrete.MultiDiscrete([3, 33])
    observation_space = gym.spaces.MultiDiscrete([33] * args.max_len)

    fitness = {}
    for row in csv.reader(open("./data/PhoQ.csv")):
        if row[0] == 'AACombo':
            continue
        fitness[row[0]] = float(row[1])

    ground = {}
    for row in csv.reader(open("./data/PhoQ.csv")):
        if row[0] == 'Variants':
            continue
        ground[row[0]] = float(row[1])
        
    tokenizer = AutoTokenizer.from_pretrained("./esm_8m")

    m_env_kwargs = {"action_space": action_space, "observation_space": observation_space, "args": args}
    env = PhoQEnv(**m_env_kwargs)
    model = DPO.load(path="checkpoints/DPO.zip", env=env, device=device)

    policy = model.policy.to(device).eval()
    
    all_wildtype_results = []
    

    for wildtype_index in range(env.total_seqs):
        wildtype_result = run_sampling_for_wildtype(env, model, policy, wildtype_index, max_attempts=128)
        if wildtype_result:
            all_wildtype_results.append(wildtype_result)
    
    t2 = time.time()

    wildtype_data = []
    for wildtype_result in all_wildtype_results:
        wildtype_data.append({
            "Wildtype": wildtype_result['protein'],
            "Wildtype_Index": wildtype_result['index'],
            "Initial_Fitness": wildtype_result['initial_fitness'],
            "Target_Fitness": wildtype_result['target_fitness'],
            "Attempts_Needed": wildtype_result['attempts_needed'],
            "Best_Fitness_Found": wildtype_result['best_fitness_found'],
            "Successful_Fitness": wildtype_result['successful_fitness'],
            "Success": wildtype_result['successful_fitness'] is not None
        })
    
    wildtype_df = pd.DataFrame(wildtype_data)
    wildtype_df.to_csv(r"./output_PhoQ/DPO/wildtype_sampling_results.csv", index=False)
