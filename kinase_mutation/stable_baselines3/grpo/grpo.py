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


class GRPO(OnPolicyAlgorithm):
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
    :param beta: Temperature parameter for GRPO loss (controls preference strength)
    :param group_size: Size of groups for relative comparison
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
        group_size: int = 8,
        ent_coef: float = 0.0,
        max_grad_norm: float = 1,
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

        super(GRPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=0.0,  # GRPO doesn't use value function
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
        self.group_size = group_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.target_kl = None  # GRPO doesn't use target_kl

    def _setup_model(self) -> None:
        super(GRPO, self)._setup_model()
        # Initialize beta schedule if needed
        if hasattr(self, 'beta'):
            self.beta = get_schedule_fn(self.beta)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer with GRPO loss.
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current beta
        if hasattr(self, 'beta') and callable(self.beta):
            beta = self.beta(self._current_progress_remaining)
        else:
            beta = self.beta if hasattr(self, 'beta') else 0.1

        entropy_losses, all_kl_divs = [], []
        grpo_losses = []

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
                
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # GRPO loss calculation
                # Group actions by their advantage values and compare within groups
                n_samples = log_prob.shape[0]
                if n_samples >= self.group_size:
                    # Sort actions by advantage for grouping
                    sorted_indices = th.argsort(advantages, descending=True)
                    sorted_log_prob = log_prob[sorted_indices]
                    sorted_old_log_prob = rollout_data.old_log_prob[sorted_indices]
                    sorted_advantages = advantages[sorted_indices]
                    
                    # Create groups
                    n_groups = n_samples // self.group_size
                    if n_groups > 0:
                        grpo_loss = th.tensor(0.0, device=self.device)
                        group_losses = []
                        
                        for i in range(n_groups):
                            start_idx = i * self.group_size
                            end_idx = start_idx + self.group_size
                            
                            group_log_prob = sorted_log_prob[start_idx:end_idx]
                            group_old_log_prob = sorted_old_log_prob[start_idx:end_idx]
                            group_advantages = sorted_advantages[start_idx:end_idx]
                            
                            # Within group, compare best vs worst
                            best_idx = th.argmax(group_advantages)
                            worst_idx = th.argmin(group_advantages)
                            
                            best_log_prob = group_log_prob[best_idx]
                            worst_log_prob = group_log_prob[worst_idx]
                            best_old_log_prob = group_old_log_prob[best_idx]
                            worst_old_log_prob = group_old_log_prob[worst_idx]
                            
                            # GRPO loss: -log(sigmoid(beta * (log_ratio_best - log_ratio_worst)))
                            log_ratio_best = best_log_prob - best_old_log_prob
                            log_ratio_worst = worst_log_prob - worst_old_log_prob
                            
                            group_loss = -th.log(th.sigmoid(beta * (log_ratio_best - log_ratio_worst)))
                            group_losses.append(group_loss)
                        
                        if group_losses:
                            grpo_loss = th.stack(group_losses).mean()
                    else:
                        # Fallback to standard policy gradient if not enough samples
                        grpo_loss = -(advantages * log_prob).mean()
                else:
                    # Fallback to standard policy gradient if not enough samples
                    grpo_loss = -(advantages * log_prob).mean()

                # Logging
                grpo_losses.append(grpo_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # loss = grpo_loss + self.ent_coef * entropy_loss
                loss = grpo_loss 

                # Optimization step
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
        logger.record("train/grpo_loss", np.mean(grpo_losses))
        logger.record("train/approx_kl", np.mean(all_kl_divs))
        logger.record("train/clip_fraction", 0.0)  # GRPO doesn't use clipping
        if hasattr(self.policy, "log_std"):
            logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/clip_range", 0.0)  # GRPO doesn't use clipping
        logger.record("train/beta", beta)
        logger.record("train/group_size", self.group_size)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "GRPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "GRPO":

        return super(GRPO, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        ) 