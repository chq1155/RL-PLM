import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.buffers import RewardRolloutBuffer


class DPO(OnPolicyAlgorithm):
    """
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param beta: Temperature parameter for DPO loss (controls preference strength)
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        beta: float = 0.1,
        ent_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(DPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=0.0,  
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            supported_action_spaces=(spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary),
        )

        self.beta = beta
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.target_kl = None  

    def _setup_model(self) -> None:
        # Call parent setup but override rollout buffer
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # Use custom RewardRolloutBuffer instead of standard RolloutBuffer
        self.rollout_buffer = RewardRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            n_envs=self.n_envs if self.n_envs is not None else 1,
        )
        
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)
        
        # Initialize beta schedule if needed
        if hasattr(self, 'beta'):
            self.beta = get_schedule_fn(self.beta)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer with DPO loss.
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current beta
        if hasattr(self, 'beta') and callable(self.beta):
            beta = self.beta(self._current_progress_remaining)
        else:
            beta = self.beta if hasattr(self, 'beta') else 0.1

        entropy_losses, all_kl_divs = [], []
        dpo_losses = []

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):

                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                
                # Use rewards directly instead of advantages
                rewards = rollout_data.advantages  # Now contains rewards from RewardRolloutBuffer

                # Normalize rewards for better training stability
                rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

                preferred_mask = rewards_normalized > 0
                non_preferred_mask = rewards_normalized <= 0
                
                if preferred_mask.sum() > 0 and non_preferred_mask.sum() > 0:
                    # Get log probabilities for preferred and non-preferred actions
                    preferred_log_prob = log_prob[preferred_mask]
                    non_preferred_log_prob = log_prob[non_preferred_mask]
                    
                    # Reference log probabilities (from old policy)
                    preferred_ref_log_prob = rollout_data.old_log_prob[preferred_mask]
                    non_preferred_ref_log_prob = rollout_data.old_log_prob[non_preferred_mask]
                    
                    # DPO loss: -log(sigmoid(beta * (log_ratio_preferred - log_ratio_non_preferred)))
                    log_ratio_preferred = preferred_log_prob - preferred_ref_log_prob
                    log_ratio_non_preferred = non_preferred_log_prob - non_preferred_ref_log_prob
                    
                    # Sample pairs for comparison
                    n_pairs = min(preferred_log_prob.shape[0], non_preferred_log_prob.shape[0])
                    if n_pairs > 0:
                        idx_pref = th.randperm(preferred_log_prob.shape[0])[:n_pairs]
                        idx_non_pref = th.randperm(non_preferred_log_prob.shape[0])[:n_pairs]
                        
                        log_ratio_pref_sampled = log_ratio_preferred[idx_pref]
                        log_ratio_non_pref_sampled = log_ratio_non_preferred[idx_non_pref]
                        
                        # DPO loss
                        dpo_loss = -th.log(th.sigmoid(beta * (log_ratio_pref_sampled - log_ratio_non_pref_sampled))).mean()
                    else:
                        dpo_loss = th.tensor(0.0, device=self.device)
                else:
                    # Fallback to standard policy gradient if no preference pairs
                    dpo_loss = -(rewards_normalized * log_prob).mean()

                # Logging
                dpo_losses.append(dpo_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = dpo_loss + self.ent_coef * entropy_loss 

                # Optimization step
                if self.policy.optimizer is not None:
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
                approx_kl_divs.append(th.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())

            all_kl_divs.append(np.mean(approx_kl_divs))

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                break

        self._n_updates += self.n_epochs

        # Logs
        logger.record("train/entropy_loss", np.mean(entropy_losses))
        logger.record("train/dpo_loss", np.mean(dpo_losses))
        logger.record("train/approx_kl", np.mean(all_kl_divs))
        logger.record("train/clip_fraction", 0.0)  # DPO doesn't use clipping
        if hasattr(self.policy, "log_std"):
            logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/clip_range", 0.0)  # DPO doesn't use clipping
        logger.record("train/beta", beta)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "DPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "DPO":

        return super(DPO, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )  # type: ignore 