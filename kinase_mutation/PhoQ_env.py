import argparse
import imp
from stable_baselines3.grpo import GRPO
from stable_baselines3.dpo import DPO
import torch
import gym
import itertools
import numpy as np
import copy
import random
import time
import csv
from contextlib import contextmanager
import pandas as pd
import sys, os
from transformers import AutoTokenizer,AutoModel,EsmForMaskedLM
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common import logger
from stable_baselines3.common.env_checker import check_env
cwd = os.path.dirname(os.path.abspath(__file__))

collected_seqs_set = set()
# path_96 or path_192 or path_288
path_96 = "data/train_init_sequences.csv"
pos = {0:96,1:97,2:100}
re_pos = {96:0,97:1,100:2}

AMINO_ACIDS = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
class PhoQEnv(gym.Env):
    def __init__(self,
                 action_space: gym.spaces,
                 observation_space: gym.spaces,
                 args: dict,
                 max_len: int = 58,
                 ):
        super(PhoQEnv, self).__init__()
        self.action_space = action_space
        self.observation_space = observation_space

        self.reward = float("-inf")
        self.reward_list = []
        self.k = [0, 6]
        self.max_step = args.max_step
        self.score_stop_criteria = args.score_stop_criteria

        self.len_step = 0
        self.max_len = max_len

        datas = pd.read_csv(path_96, names=['AACombo', 'Fitness','FitnessGroup'], header=0)
        self.PhoQ = []
        self.PhoQ_fitness = []
        self.PhoQ_protein = []
        self.num2seq = {}
        for i in range(len(datas)):
            protein = "SYMVWSWFIYVLSANLLLVIPLLWVAAWWSLRPIEALAKEVRELEEHNRELLNPATTRELTSLVRNLNRLLKSERERYDKYRTTLTDLTHSLKTPL__LQ__LRSLRSEKMSVSDAEPVMLEQISRISQQIGYYLHRASMRGGTLLSRELHPVAPLLDNLTSALNKVYQRKGVNISLDISPEISFVGEQ"
            protein = protein.replace("__", datas["AACombo"][i][0:2], 1)
            protein = protein.replace("__", datas["AACombo"][i][2:4], 1)
            tokens = tokenizer(protein, return_tensors="pt").to(device)
            self.PhoQ_protein.append(datas["AACombo"][i])
            self.PhoQ.append(tokens['input_ids'].squeeze(0).cpu().numpy())
            self.PhoQ_fitness.append(float(datas['Fitness'][i]))

        print("finish build")

    def init_seq(self):
        index = random.randint(0, len(self.PhoQ)-1)
        
        self.initial_seq = self.PhoQ[index]
        self.score_stopdata_criteria = self.PhoQ_fitness[index]

        collected_seqs_set.add(self.PhoQ_protein[index])

        return self.initial_seq

    def reset(self):

        self.state = self.init_seq()

        self.len_step = 0

        self.reward_list = []

        self.k = [0, 6]

        return self.state

    def check_terminal(self, score, stirng, have):
        if have == True:
            collected_seqs_set.add(stirng)
        if score > self.score_stop_criteria or self.len_step >= self.max_step:
            return True
        else:
            return False

    def _get_reward(self,seq):
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
        elif flag == True:
            score_truth = -100
        else:
            score_truth = -1
        if string in ground.keys():
            ground_truth = ground[string]
        else:
            ground_truth = -1
        self.reward = score_truth

        terminal = self.check_terminal(self.reward, string, have)

        return self.reward, terminal, score_truth

    def _edit_sequence(self, seq, actions):
        protein = seq
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
        # Check if the last value is increasing compared to the previous one
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

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

if __name__ == '__main__':
    import sys, os
    from stable_baselines3.ppo import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
    from ESM_PhoQ import PolicyNet,model_name
    import pickle
    import torch
    import warnings
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    learn_rate=1e-6
    
    tensorboard_log = "./tensorboard_logs/"+str(learn_rate)

    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help="path to save results", default="./checkpoints")
    parser.add_argument('--algorithm', type=str, help="RL algorithm", default="DPO")

    parser.add_argument('--gamma', type=float, default=0, help="discount_factor")
    parser.add_argument('--steps', type=int, default=10000, help="total time steps")
    parser.add_argument('--ent_coef', type=float, default=0, help="encourage exploration")

    parser.add_argument('--clip', type=float, default=0.2, help="")
    parser.add_argument('--max_len', type=int, default=191)

    parser.add_argument('--num_envs', type=int, default=10, help="number of environments")
    parser.add_argument('--n_steps', type=int, default=10, help="number of roll out steps")
    parser.add_argument('--max_step', type=int, default=3, help="maximum number of steps")
    parser.add_argument('--score_stop_criteria', type=float, default=60, help="stop_criteria")

    args = parser.parse_args()
    path = args.path
    t1 = time.time()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    action_space = gym.spaces.multi_discrete.MultiDiscrete([3,33])
    observation_space = gym.spaces.MultiDiscrete([33]*args.max_len)

    fitness = {}
    for row in csv.reader(open("data/PhoQ.csv")):
        if row[0] == 'AACombo':
            continue
        fitness[row[0]] = float(row[1])

    ground = {}
    for row in csv.reader(open("./data/PhoQ.csv")):
        if row[0] == 'Variants':
            continue
        ground[row[0]] = float(row[1])

    if model_name == "ESM_8M":
        tokenizer = AutoTokenizer.from_pretrained("./esm_8m")
    elif model_name == "ESM_35M":
        tokenizer = AutoTokenizer.from_pretrained("./esm_35m")
    elif model_name == "ESM_650M":
        tokenizer = AutoTokenizer.from_pretrained("./esm_650m")

    m_env_kwargs = {"action_space": action_space, "observation_space": observation_space, "args": args}
    
    m_env = make_vec_env(PhoQEnv, n_envs=args.num_envs, env_kwargs=m_env_kwargs)

    checkpoint_callback = CheckpointCallback(save_freq=2000, save_path=path + '/', name_prefix='rl_model')
    if args.algorithm=="PPO":
        model = PPO(PolicyNet, m_env, learning_rate=learn_rate, verbose=1, n_steps=args.n_steps, ent_coef=args.ent_coef,
                    gamma=args.gamma,  tensorboard_log=tensorboard_log, device=device, batch_size=16)
    if args.algorithm=="DPO":
        model = DPO(PolicyNet, m_env, learning_rate=learn_rate, verbose=1, n_steps=args.n_steps, ent_coef=args.ent_coef,
                    gamma=args.gamma,  tensorboard_log=tensorboard_log, device=device, batch_size=16)

    print_trainable_parameters(model.policy)

    model.learn(total_timesteps=args.steps, callback=checkpoint_callback, log_interval=1)
    t2 = time.time()

    print("finish training in %.4f" % (t2 - t1))
    print("saving model.....")
    model.save(path=path + str(model_name)+"/PPO/"+str(learn_rate))


