from copy import deepcopy
import argparse

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
# import cma
import pandas as pd

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from ES import sepCMAES, sepCEM, sepMCEM
from models import RLNN
from collections import namedtuple
from random_process import GaussianNoise
from memory import Memory, Archive
from samplers import IMSampler
from util import *
import wandb 

os.environ["WANDB_API_KEY"] = "bd4f9ed4f0e350a34c7ef0ea98f697dd7cb718fe"
os.environ["WANDB_MODE"] = "offline"

wandb.init(project="CEM",name="CEM-TQC ant-v3")


Sample = namedtuple('Sample', ('params', 'score',
                               'gens', 'start_pos', 'end_pos', 'steps'))
Theta = namedtuple('Theta', ('mu', 'cov', 'samples'))
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor


def evaluate(actor, env, memory=None, n_episodes=1, random=False, noise=None, render=False):
    """
    Computes the score of an actor on a given number of runs,
    fills the memory if needed
    """

    if not random:
        def policy(state):
            state = FloatTensor(state.reshape(-1))
            action = actor(state) 
            if isinstance(action, tuple):
                action = action[0].cpu().data.numpy().flatten()
            else:
                action = action.cpu().data.numpy().flatten()

            if noise is not None:
                action += noise.sample()

            return np.clip(action, -max_action, max_action)

    else:
        def policy(state):
            return env.action_space.sample()

    scores = []
    steps = 0

    for _ in range(n_episodes):

        score = 0
        obs, info = deepcopy(env.reset())
        done = False
        truncated = False

        while not done and not truncated:

            # get next action and act
            action = policy(obs)
            n_obs, reward, done, truncated, info = env.step(action)
            done_bool = 0 if steps + \
                1 == env._max_episode_steps else float(done)
            score += reward
            steps += 1

            # adding in memory
            if memory is not None:
                memory.add((obs, n_obs, action, reward, done_bool))
            obs = n_obs

            # render if needed
            if render:
                env.render()

            # reset when done
            if done:
                env.reset()

        scores.append(score)

    return np.mean(scores), steps


class Actor(RLNN):

    def __init__(self, state_dim, action_dim, max_action, args):
        super(Actor, self).__init__(state_dim, action_dim, max_action)

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)
        self.layer_norm = args.layer_norm

        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.actor_lr)
        self.tau = args.tau
        self.discount = args.discount
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def forward(self, x):

        if not self.layer_norm:
            x = torch.tanh(self.l1(x))
            x = torch.tanh(self.l2(x))
            x = self.max_action * F.tanh(self.l3(x))

        else:
            x = torch.tanh(self.n1(self.l1(x)))
            x = torch.tanh(self.n2(self.l2(x)))
            x = self.max_action * F.tanh(self.l3(x))

        return x

    def update(self, memory, batch_size, critic, actor_t):

        # Sample replay buffer
        states, _, _, _, _ = memory.sample(batch_size)

        # Compute actor loss
        if args.use_td3:
            actor_loss = -critic(states, self(states))[0].mean()
        else:
            actor_loss = -critic(states, self(states)).mean()

        # Optimize the actor
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), actor_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)


class Critic(RLNN):
    def __init__(self, state_dim, action_dim, max_action, args):
        super(Critic, self).__init__(state_dim, action_dim, 1)

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)

        self.layer_norm = args.layer_norm
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.critic_lr)
        self.tau = args.tau
        self.discount = args.discount
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def forward(self, x, u):

        if not self.layer_norm:
            x = F.leaky_relu(self.l1(torch.cat([x, u], 1)))
            x = F.leaky_relu(self.l2(x))
            x = self.l3(x)

        else:
            x = F.leaky_relu(self.n1(self.l1(torch.cat([x, u], 1))))
            x = F.leaky_relu(self.n2(self.l2(x)))
            x = self.l3(x)

        return x

    def update(self, memory, batch_size, actor_t, critic_t):

        # Sample replay buffer
        states, n_states, actions, rewards, dones = memory.sample(batch_size)

        # Q target = reward + discount * Q(next_state, pi(next_state))
        with torch.no_grad():
            target_Q = critic_t(n_states, actor_t(n_states))
            target_Q = rewards + (1 - dones) * self.discount * target_Q

        # Get current Q estimate
        current_Q = self(states, actions)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        # Optimize the critic
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), critic_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)


class CriticTD3(RLNN):
    def __init__(self, state_dim, action_dim, max_action, args):
        super(CriticTD3, self).__init__(state_dim, action_dim, 1)

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

        if args.layer_norm:
            self.n4 = nn.LayerNorm(400)
            self.n5 = nn.LayerNorm(300)

        self.layer_norm = args.layer_norm
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.critic_lr)
        self.tau = args.tau
        self.discount = args.discount
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip

    def forward(self, x, u):

        if not self.layer_norm:
            x1 = F.leaky_relu(self.l1(torch.cat([x, u], 1)))
            x1 = F.leaky_relu(self.l2(x1))
            x1 = self.l3(x1)

        else:
            x1 = F.leaky_relu(self.n1(self.l1(torch.cat([x, u], 1))))
            x1 = F.leaky_relu(self.n2(self.l2(x1)))
            x1 = self.l3(x1)

        if not self.layer_norm:
            x2 = F.leaky_relu(self.l4(torch.cat([x, u], 1)))
            x2 = F.leaky_relu(self.l5(x2))
            x2 = self.l6(x2)

        else:
            x2 = F.leaky_relu(self.n4(self.l4(torch.cat([x, u], 1))))
            x2 = F.leaky_relu(self.n5(self.l5(x2)))
            x2 = self.l6(x2)

        return x1, x2

    def update(self, memory, batch_size, actor_t, critic_t):

        # Sample replay buffer
        states, n_states, actions, rewards, dones = memory.sample(batch_size)

        # Select action according to policy and add clipped noise
        noise = np.clip(np.random.normal(0, self.policy_noise, size=(
            batch_size, action_dim)), -self.noise_clip, self.noise_clip)
        n_actions = actor_t(n_states) + FloatTensor(noise)
        n_actions = n_actions.clamp(-max_action, max_action)

        # Q target = reward + discount * min_i(Qi(next_state, pi(next_state)))
        with torch.no_grad():
            target_Q1, target_Q2 = critic_t(n_states, n_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self(states, actions)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + \
            nn.MSELoss()(current_Q2, target_Q)

        # Optimize the critic
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), critic_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

# -----------------------------------SAC-----------------------------------
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.distributions import normal as pyd

class StableTanhTransform(TanhTransform):
    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, StableTanhTransform)

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)


class SquashedNormal(TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.base_dist = pyd.Normal(loc, scale)

        transforms = [StableTanhTransform()]
        super().__init__(self.base_dist, transforms, validate_args=False)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

class ActorSAC(RLNN):
    
        def __init__(self, state_dim, action_dim, max_action, args, hidden_state, min_log_std=-20, max_log_std=2):
            super(ActorSAC, self).__init__(state_dim, action_dim, max_action)
            
            if hidden_state is None:
                hidden_state = [256, 256]

            self.actor_net = nn.Sequential(
                nn.Linear(state_dim, hidden_state[0]),
                nn.ReLU(),
                nn.Linear(hidden_state[0], hidden_state[1]),
                nn.ReLU
            )
            
            self.mu_head = nn.Linear(hidden_state[1], action_dim)
            self.log_std_head = nn.Linear(hidden_state[1], action_dim)
            self.max_action = max_action

            self.min_log_std = min_log_std
            self.max_log_std = max_log_std
            
            self.tau = args.tau
            self.args = args
            
            self.optimizer = torch.optim.Adam(self.parameters(), lr=args.actor_lr)
            
        def forward(self, state):
            x = self.actor_net(state)
            mu = self.mu_head(x)
            log_std_head = self.log_std_head(x)
            log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std)

            std = log_std_head.exp()
            dist = SquashedNormal(mu, std)
            sample = dist.rsample()
            log_pi = dist.log_prob(sample).sum(axis=-1, keepdim=True)   

            return sample, log_pi
    
        def update(self, memory, batch_size, critic, critic_t):
    
            # Sample replay buffer
            states, _, _, _, _ = memory.sample(batch_size)
    
            # Compute actor loss
            actor_loss = -critic(states, self(states)).mean()
    
            # Optimize the actor
            self.optimizer.zero_grad()
            actor_loss.backward()
            self.optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.parameters(), critic_t.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

class CriticSAC(RLNN):
    def __init__(self, state_dim, action_dim, max_action, args):
        super(CriticSAC, self).__init__(state_dim, action_dim, 1)

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(256)
            self.n2 = nn.LayerNorm(256)

        self.layer_norm = args.layer_norm
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.critic_lr)
        self.tau = args.tau
        self.discount = args.discount
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def forward(self, x, u):

        if not self.layer_norm:
            x = F.leaky_relu(self.l1(torch.cat([x, u], 1)))
            x = F.leaky_relu(self.l2(x))
            x = self.l3(x)

        else:
            x = F.leaky_relu(self.n1(self.l1(torch.cat([x, u], 1))))
            x = F.leaky_relu(self.n2(self.l2(x)))
            x = self.l3(x)

        return x

    def update(self, memory, batch_size, actor, critic_t):

        # Sample replay buffer
        states, n_states, actions, rewards, dones = memory.sample(batch_size)

        # Q target = reward + discount * Q(next_state, pi(next_state))
        with torch.no_grad():
            target_Q = critic_t(n_states, actor(n_states))
            target_Q = rewards + (1 - dones) * self.discount * target_Q

        # Get current Q estimate
        current_Q = self(states, actions)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        # Optimize the critic
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), critic_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)



# -----------------------------------SAC-----------------------------------


# -----------------------------------TQC-----------------------------------
def quantile_huber_loss_f(
    quantiles: torch.Tensor, samples: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the quantile Huber loss for a given set of quantiles and samples.

    Args:
        quantiles (torch.Tensor): A tensor of shape (batch_size, num_nets, num_quantiles) representing the quantiles.
        samples (torch.Tensor): A tensor of shape (batch_size, num_samples) representing the samples.

    Returns:
        torch.Tensor: The quantile Huber loss.

    """
    pairwise_delta = (
        samples[:, None, None, :] - quantiles[:, :, :, None]
    )  # batch x nets x quantiles x samples
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(
        abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5
    )

    n_quantiles = quantiles.shape[2]

    tau = (
        torch.arange(n_quantiles, device=pairwise_delta.device).float() / n_quantiles
        + 1 / 2 / n_quantiles
    )
    loss = (
        torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss
    ).mean()
    return loss

class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        super().__init__()

        self.fully_connected_layers = []
        for i, next_size in enumerate(hidden_sizes):
            fully_connected_layer = nn.Linear(input_size, next_size)
            self.add_module(f"fully_connected_layer_{i}", fully_connected_layer)
            self.fully_connected_layers.append(fully_connected_layer)
            input_size = next_size

        self.output_layer = nn.Linear(input_size, output_size)

    def forward(self, state):
        for fully_connected_layer in self.fully_connected_layers:
            state = F.relu(fully_connected_layer(state))
        output = self.output_layer(state)
        return output

class CriticTQC(RLNN):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: int, 
        num_quantiles: int,
        num_critics: int,
        hidden_size: list[int] = None,
        args: dict = None,
    ):
        super(CriticTQC, self).__init__(state_dim, action_dim, 1)

        if hidden_size is None:
            hidden_size = [512, 512, 512]

        self.q_networks = []
        self.num_quantiles = num_quantiles
        self.num_critics = num_critics

        self.tau = args.tau
        self.discount = args.discount
        self.gamma = args.gamma
        self.alpha = args.alpha

        self.top_quantiles_to_drop = args.top_quantiles_to_drop

        self.quantiles_total = (
            self.num_quantiles * self.num_critics
        )

        for i in range(self.num_critics):
            critic_net = MLP(
                state_dim + action_dim, hidden_size, self.num_quantiles
            )
            self.add_module(f"critic_net_{i}", critic_net)
            self.q_networks.append(critic_net)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.critic_lr)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if isinstance(action, tuple):
            action = action[0]
        network_input = torch.cat((state, action), dim=1)
        quantiles = torch.stack(
            tuple(critic(network_input) for critic in self.q_networks), dim=1
        )
        return quantiles
    
    def update(self, memory, batch_size, actor, critic_t):
        states, n_states, actions, rewards, dones = memory.sample(batch_size)
        
        with torch.no_grad():
            next_actions, next_log_pi = actor(n_states)

            # compute and cut quantiles at the next state
            # batch x nets x quantiles
            target_q_values = critic_t(n_states, next_actions)
            sorted_target_q_values, _ = torch.sort(
                target_q_values.reshape(batch_size, -1)
            )
            top_quantile_target_q_values = sorted_target_q_values[
                :, : self.quantiles_total - self.top_quantiles_to_drop
            ]

            # compute target
            q_target = rewards + (1 - dones) * self.gamma * (
                top_quantile_target_q_values - (actor.alpha * next_log_pi).reshape(batch_size, -1)
            )

        q_values = self.forward(states, actions)
        critic_loss_total = quantile_huber_loss_f(q_values, q_target)

        self.optimizer.zero_grad()
        critic_loss_total.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), critic_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss_total.item()

class ActorTQC(RLNN):
    def __init__(self, state_dim, action_dim, max_action, args, hidden_state, min_log_std=-20, max_log_std=2):
        super(ActorTQC, self).__init__(state_dim, action_dim, max_action)
        if hidden_state is None:
                hidden_state = [256, 256]

        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, hidden_state[0]),
            nn.ReLU(),
            nn.Linear(hidden_state[0], hidden_state[1]),
            nn.ReLU()
        )
        
        self.mu_head = nn.Linear(hidden_state[1], action_dim)
        self.log_std_head = nn.Linear(hidden_state[1], action_dim)
        self.max_action = max_action

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        
        self.tau = args.tau
        self.args = args
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.actor_lr)
        
        self.init_temperature = 0.1
        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(np.log(self.init_temperature))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.actor_lr)


    def forward(self, state):
        x = self.actor_net(state)
        mu = self.mu_head(x)
        log_std_head = self.log_std_head(x)
        log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std)

        std = log_std_head.exp()
        dist = SquashedNormal(mu, std)
        sample = dist.rsample()
        log_pi = dist.log_prob(sample).sum(axis=-1, keepdim=True)   

        return sample, log_pi

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()
            
    def update(self, memory, batch_size, critic, critic_t):
        states, _, _, _, _ = memory.sample(batch_size)

        new_action, log_pi = self(states)
        
        mean_qf_pi = critic(states, new_action).mean(2).mean(1, keepdim=True)
        actor_loss = (self.alpha * log_pi - mean_qf_pi).mean()

        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()

        # update the temperature
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss.item(), alpha_loss.item()

# -----------------------------------TQC-----------------------------------



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', type=str,)
    parser.add_argument('--env', default='HalfCheetah-v2', type=str)
    parser.add_argument('--start_steps', default=10000, type=int)

    # DDPG parameters
    parser.add_argument('--actor_lr', default=0.001, type=float)
    parser.add_argument('--critic_lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--reward_scale', default=1., type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--layer_norm', dest='layer_norm', action='store_true')

    # TD3 parameters
    parser.add_argument('--use_td3', dest='use_td3', action='store_true')
    parser.add_argument('--use_sac', dest='use_sac', action='store_true')
    parser.add_argument('--use_tqc', dest='use_tqc', action='store_true')
    parser.add_argument('--policy_noise', default=0.2, type=float)
    parser.add_argument('--noise_clip', default=0.5, type=float)
    parser.add_argument('--policy_freq', default=2, type=int)

    # TQC parameteres
    parser.add_argument('--top_quantiles_to_drop', default=2, type=int)

    # Gaussian noise parameters
    parser.add_argument('--gauss_sigma', default=0.1, type=float)

    # OU process parameters
    parser.add_argument('--ou_noise', dest='ou_noise', action='store_true')
    parser.add_argument('--ou_theta', default=0.15, type=float)
    parser.add_argument('--ou_sigma', default=0.2, type=float)
    parser.add_argument('--ou_mu', default=0.0, type=float)

    # ES parameters
    parser.add_argument('--pop_size', default=10, type=int)
    parser.add_argument('--elitism', dest="elitism",  action='store_true')
    parser.add_argument('--n_grad', default=5, type=int)
    parser.add_argument('--sigma_init', default=1e-3, type=float)
    parser.add_argument('--damp', default=1e-3, type=float)
    parser.add_argument('--damp_limit', default=1e-5, type=float)
    parser.add_argument('--mult_noise', dest='mult_noise', action='store_true')

    # Sampler parameters
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon', type=float, default=0)
    parser.add_argument('--k', type=int, default=1)

    # Training parameters
    parser.add_argument('--n_episodes', default=1, type=int)
    parser.add_argument('--max_steps', default=1000000, type=int)
    parser.add_argument('--mem_size', default=1000000, type=int)
    parser.add_argument('--n_noisy', default=0, type=int)

    # Testing parameters
    parser.add_argument('--filename', default="", type=str)
    parser.add_argument('--n_test', default=1, type=int)

    # misc
    parser.add_argument('--output', default='results/', type=str)
    parser.add_argument('--period', default=5000, type=int)
    parser.add_argument('--n_eval', default=10, type=int)
    parser.add_argument('--save_all_models',
                        dest="save_all_models", action="store_true")
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--render', dest='render', action='store_true')

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    with open(args.output + "/parameters.txt", 'w') as file:
        for key, value in vars(args).items():
            file.write("{} = {}\n".format(key, value))

    # environment
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])

    # memory
    memory = Memory(args.mem_size, state_dim, action_dim)

    # critic
    if args.use_td3:
        critic = CriticTD3(state_dim, action_dim, max_action, args)
        critic_t = CriticTD3(state_dim, action_dim, max_action, args)
        critic_t.load_state_dict(critic.state_dict())
    elif args.use_sac:
        critic = CriticSAC(state_dim, action_dim, max_action, args)
        critic_t = CriticSAC(state_dim, action_dim, max_action, args)
        critic_t.load_state_dict(critic.state_dict())
    elif args.use_tqc:
        critic = CriticTQC(state_dim=state_dim, action_dim=action_dim, max_action=None, num_quantiles=25, num_critics=5, hidden_size=[512, 512, 512], args=args) # [512, 512, 512] [256, 256, 256]
        critic_t = CriticTQC(state_dim=state_dim, action_dim=action_dim, max_action=None, num_quantiles=25, num_critics=5, hidden_size=[512, 512, 512], args=args)
        critic_t.load_state_dict(critic.state_dict())
    else:
        critic = Critic(state_dim, action_dim, max_action, args)
        critic_t = Critic(state_dim, action_dim, max_action, args)
        critic_t.load_state_dict(critic.state_dict())

    # actor
    if args.use_sac :
        actor = ActorSAC(state_dim, action_dim, max_action, args)
        actor_t = ActorSAC(state_dim, action_dim, max_action, args)
        actor_t.load_state_dict(actor.state_dict())
    elif args.use_tqc:
        actor = ActorTQC(state_dim, action_dim, max_action, args, hidden_state=[256, 256])
        actor_t = ActorTQC(state_dim, action_dim, max_action, args, hidden_state=[256, 256])
    else:
        actor = Actor(state_dim, action_dim, max_action, args)
        actor_t = Actor(state_dim, action_dim, max_action, args)
        actor_t.load_state_dict(actor.state_dict())
        a_noise = GaussianNoise(action_dim, sigma=args.gauss_sigma)

    if USE_CUDA:
        critic.cuda()
        critic_t.cuda()
        actor.cuda()
        actor_t.cuda()

    # CEM
    es = sepCEM(actor.get_size(), mu_init=actor.get_params(), sigma_init=args.sigma_init, damp=args.damp, damp_limit=args.damp_limit,
                pop_size=args.pop_size, antithetic=not args.pop_size % 2, parents=args.pop_size // 2, elitism=args.elitism)
    sampler = IMSampler(es)

    # stuff to save
    df = pd.DataFrame(columns=["total_steps", "average_score",
                               "average_score_rl", "average_score_ea", "best_score"])

    # training
    step_cpt = 0
    total_steps = 0
    actor_steps = 0
    reused_steps = 0

    es_params = []
    fitness = []
    n_steps = []
    n_start = []

    old_es_params = []
    old_fitness = []
    old_n_steps = []
    old_n_start = []

    while total_steps < args.max_steps:

        fitness = np.zeros(args.pop_size)
        n_start = np.zeros(args.pop_size)
        n_steps = np.zeros(args.pop_size)
        es_params, n_r, idx_r = sampler.ask(args.pop_size, old_es_params)
        print("Reused {} samples".format(n_r))

        # udpate the rl actors and the critic
        if total_steps > args.start_steps:

            for i in range(args.n_grad):

                # set params
                actor.set_params(es_params[i])
                actor_t.set_params(es_params[i])
                actor.optimizer = torch.optim.Adam(
                    actor.parameters(), lr=args.actor_lr)

                # critic update
                for _ in tqdm(range(int((actor_steps + reused_steps) / args.n_grad))):
                    critic.update(memory, args.batch_size, actor, critic_t)

                # actor update
                for _ in tqdm(range(int(actor_steps + reused_steps))):
                    actor.update(memory, args.batch_size,
                                 critic, actor_t)

                # get the params back in the population
                es_params[i] = actor.get_params()

        actor_steps = 0
        reused_steps = 0

        # evaluate noisy actor(s)
        for i in range(args.n_noisy):
            actor.set_params(es_params[i])
            f, steps = evaluate(actor, env, memory=memory, n_episodes=args.n_episodes,
                                render=args.render, noise=a_noise)
            actor_steps += steps
            prCyan('Noisy actor {} fitness:{}'.format(i, f))

        # evaluate all actors
        for i in range(args.pop_size):

            # evaluate new actors
            if i < args.n_grad or (i >= args.n_grad and (i - args.n_grad) >= n_r):

                actor.set_params(es_params[i])
                pos = memory.get_pos()
                f, steps = evaluate(actor, env, memory=memory, n_episodes=args.n_episodes,
                                    render=args.render)
                actor_steps += steps

                # updating arrays
                fitness[i] = f
                n_steps[i] = steps
                n_start[i] = pos

                # print scores
                prLightPurple('Actor {}, fitness:{}'.format(i, f))

            # reusing actors
            else:
                idx = idx_r[i - args.n_grad]
                fitness[i] = old_fitness[idx]
                n_steps[i] = old_n_steps[idx]
                n_start[i] = old_n_start[idx]

                # duplicating samples in buffer
                memory.repeat(int(n_start[i]), int(
                    (n_start[i] + n_steps[i]) % args.mem_size))

                # adding old_steps
                reused_steps += old_n_steps[idx]

                # print reused score
                prGreen('Actor {}, fitness:{}'.format(
                    i, fitness[i]))

        # update ea
        es.tell(es_params, fitness)

        # update step counts
        total_steps += actor_steps
        step_cpt += actor_steps

        # update sampler stuff
        old_fitness = deepcopy(fitness)
        old_n_steps = deepcopy(n_steps)
        old_n_start = deepcopy(n_start)
        old_es_params = deepcopy(es_params)

        # save stuff
        if step_cpt >= args.period:

            # evaluate mean actor over several runs. Memory is not filled
            # and steps are not counted
            actor.set_params(es.mu)
            f_mu, _ = evaluate(actor, env, memory=None, n_episodes=args.n_eval,
                               render=args.render)
            prRed('Actor Mu Average Fitness:{}'.format(f_mu))

            df.to_pickle(args.output + "/log.pkl")
            res = {"total_steps": total_steps,
                   "average_score": np.mean(fitness),
                   "average_score_half": np.mean(np.partition(fitness, args.pop_size // 2 - 1)[args.pop_size // 2:]),
                   "average_score_rl": np.mean(fitness[:args.n_grad]) if args.n_grad > 0 else None,
                   "average_score_ea": np.mean(fitness[args.n_grad:]),
                   "best_score": np.max(fitness),
                   "mu_score": f_mu,
                   "n_reused": n_r}

            wandb.log(res)

            if args.save_all_models:
                os.makedirs(args.output + "/{}_steps".format(total_steps),
                            exist_ok=True)
                critic.save_model(
                    args.output + "/{}_steps".format(total_steps), "critic")
                actor.set_params(es.mu)
                actor.save_model(
                    args.output + "/{}_steps".format(total_steps), "actor_mu")
            else:
                critic.save_model(args.output, "critic")
                actor.set_params(es.mu)
                actor.save_model(args.output, "actor")
            df = pd.concat([df, pd.DataFrame([res])], ignore_index=True)
            step_cpt = 0
            print(res)

        print("Total steps", total_steps)
