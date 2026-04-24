# Spring 2026, 535518 Deep Learning
# Lab5: Value-based RL
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        ########## YOUR CODE HERE (5~10 lines) ##########
        self.is_atari = (len(input_shape) == 3)

        if self.is_atari: # For Task 2
            in_channels = input_shape[0]
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, *input_shape)
                conv_out_size = self.conv(dummy).shape[1]
            self.fc = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions),
            )
        else: # For Task 1
            in_features = input_shape[0]
            self.network = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, num_actions),
            )
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        if self.is_atari:
            x = x / 255.0
            return self.fc(self.conv(x))
        else:
            return self.network(x)


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari.
        Applies element-wise max over last 2 raw frames (DeepMind standard)
        to handle sprite flickering, then grayscale + resize to 84x84.
    """
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
        self.last_raw_obs = None

    def _gray_resize(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    def preprocess(self, obs):
        # Max-pool with previous raw frame to remove Atari sprite flicker
        if self.last_raw_obs is not None:
            maxed = np.maximum(obs, self.last_raw_obs)
        else:
            maxed = obs
        self.last_raw_obs = obs
        return self._gray_resize(maxed)

    def reset(self, obs):
        self.last_raw_obs = None
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class CartPolePreprocessor:
    """Identity preprocessor for low-dim envs like CartPole-v1."""
    def reset(self, obs):
        return obs.astype(np.float32)

    def step(self, obs):
        return obs.astype(np.float32)


class SumTree:
    """
    Binary segment tree for O(log N) priority sampling and update.
    - Internal nodes: sum of their children's priorities
    - Leaf nodes: individual transition priorities
    - Total size: 2 * capacity - 1 nodes
    - Leaf positions: tree[capacity-1 .. 2*capacity-2]
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.pos = 0  # current write position (circular, in [0, capacity-1])

    def _propagate(self, idx, delta):
        # Propagate priority delta upward to the root
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def update(self, leaf_idx, priority):
        # leaf_idx: logical leaf index in [0, capacity-1]
        tree_idx = leaf_idx + self.capacity - 1
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, delta)

    def add(self, priority):
        # Write priority to current position, return the leaf index used
        leaf_pos = self.pos
        self.update(self.pos, priority)
        self.pos = (self.pos + 1) % self.capacity
        return leaf_pos

    def get(self, value):
        # Find the leaf whose cumulative prefix sum first exceeds `value`
        # Clamp value to [0, total] to avoid floating-point overshoot
        value = min(value, self.total - 1e-6)
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        leaf_idx = idx - (self.capacity - 1)
        return leaf_idx, self.tree[idx]

    @property
    def total(self):
        return float(self.tree[0])  # root = sum of all priorities


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = [None] * capacity  # fixed-size list for O(1) indexed access
        self.tree   = SumTree(capacity)  # O(log N) priority sampling & update
        self.size   = 0                  # current number of valid transitions

    def __len__(self):
        return self.size

    def add(self, transition, error):
        ########## YOUR CODE HERE (for Task 3) ##########
        # New transitions always receive max existing priority,
        # ensuring they are sampled at least once before their TD error is known.
        max_priority = self.tree.tree[self.tree.capacity - 1:
                                      self.tree.capacity - 1 + self.size].max() \
                       if self.size > 0 else 1.0

        leaf_pos = self.tree.add(max_priority)  # write to SumTree, get leaf index
        self.buffer[leaf_pos] = transition       # store transition at same index
        self.size = min(self.size + 1, self.capacity)
        ########## END OF YOUR CODE (for Task 3) ##########
        return

    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ##########
        indices, priorities = [], []
        # Stratified sampling: divide total priority into batch_size equal segments
        # and sample one value uniformly from each segment for better coverage.
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            value = random.uniform(segment * i, segment * (i + 1))
            leaf_idx, priority = self.tree.get(value)
            # Guard: resample if we land on an unfilled slot (buffer=None) or zero priority
            while self.buffer[leaf_idx] is None or priority <= 0:
                value = random.uniform(0, self.tree.total)
                leaf_idx, priority = self.tree.get(value)
            indices.append(leaf_idx)
            priorities.append(priority)

        # IS weights: w_i = (1 / (N * P(i)))^beta, normalized so max weight = 1
        probs   = np.array(priorities, dtype=np.float32) / self.tree.total
        weights = (self.size * probs) ** (-self.beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in indices]
        ########## END OF YOUR CODE (for Task 3) ##########
        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ##########
        # Vectorized: compute new raw priorities, then update SumTree one by one.
        # raw priority = |delta| + epsilon  (epsilon prevents zero priority)
        # alpha is applied here so SumTree stores p^alpha directly.
        new_prios = np.abs(errors) + 1e-6
        for idx, prio in zip(indices, new_prios):
            self.tree.update(idx, float(prio) ** self.alpha)
        ########## END OF YOUR CODE (for Task 3) ##########
        return
        

class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.num_envs = args.num_envs if args is not None else 1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.is_atari = "ALE/" in env_name
        if self.is_atari: # For Task 2
            self.preprocessor = AtariPreprocessor()
            input_shape = (4, 84, 84)
            self.best_reward = -21
        else: # For Task 1
            self.preprocessor = CartPolePreprocessor()
            input_shape = (4,)
            self.best_reward = 0

        self.q_net = DQN(input_shape, self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(input_shape, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr, eps=1.5e-4)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Task 3 enhancement flags (all default to False for vanilla DQN compatibility)
        self.use_ddqn      = args.use_ddqn      if hasattr(args, 'use_ddqn')      else False
        self.use_per       = args.use_per        if hasattr(args, 'use_per')        else False
        self.use_multistep = args.use_multistep  if hasattr(args, 'use_multistep')  else False
        # n_step=1 when multistep disabled → gamma^1 = gamma, identical to vanilla
        self.n_step        = (args.n_step if hasattr(args, 'n_step') else 1) if self.use_multistep else 1

        # Select replay buffer based on flag
        if self.use_per:
            per_alpha = args.per_alpha if hasattr(args, 'per_alpha') else 0.6
            per_beta  = args.per_beta  if hasattr(args, 'per_beta')  else 0.4
            self.memory = PrioritizedReplayBuffer(args.memory_size, per_alpha, per_beta)
        else:
            self.memory = deque(maxlen=args.memory_size)

        # Multi-step: one n-step sliding-window buffer per env
        # Guard: multistep requires vectorized mode (num_envs > 1)
        if self.use_multistep:
            if self.num_envs == 1:
                print("[WARNING] --use-multistep requires --num-envs > 1. Multi-step DISABLED.")
                self.use_multistep = False
            else:
                self.nstep_buffers = [deque() for _ in range(self.num_envs)]

        # Milestone checkpoints for Task 3 grading (600k, 1M, 1.5M, 2M, 2.5M env steps)
        self.milestone_steps   = {600_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000}
        self.saved_milestones  = set()

        if self.num_envs > 1:
            import functools
            self.vec_env = gym.vector.AsyncVectorEnv(
                [functools.partial(gym.make, env_name,
                                   render_mode="rgb_array",
                                   max_episode_steps=self.max_episode_steps)
                 for _ in range(self.num_envs)]
            )
            if self.is_atari:
                self.vec_preprocessors = [AtariPreprocessor() for _ in range(self.num_envs)]
            else:
                self.vec_preprocessors = [CartPolePreprocessor() for _ in range(self.num_envs)]

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).unsqueeze(0).to(self.device).float()
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def select_action_batch(self, states_list):
        N = len(states_list)
        explore = [random.random() < self.epsilon for _ in range(N)]
        if all(explore):
            return [random.randint(0, self.num_actions - 1) for _ in range(N)]
        batch = torch.from_numpy(np.stack(states_list)).to(self.device).float()
        with torch.no_grad():
            greedy_actions = self.q_net(batch).argmax(dim=1).cpu().numpy().tolist()
        return [
            random.randint(0, self.num_actions - 1) if explore[i] else greedy_actions[i]
            for i in range(N)
        ]

    def run(self, episodes=1000):
        if self.num_envs > 1: # 開啟多核心運算模式
            return self.run_vectorized(episodes)

        for ep in range(episodes):
            obs, _ = self.env.reset()

            # NoopReset: random 0~30 noop actions on initial frame for state diversity (Atari only)
            if self.is_atari:
                for _ in range(random.randint(0, 30)):
                    obs, _, terminated, truncated, _ = self.env.step(0)
                    if terminated or truncated:
                        obs, _ = self.env.reset()

            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                next_state = self.preprocessor.step(next_obs)
                self.memory.append((state, action, reward, next_state, done))

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed 
                    
                    ########## END OF YOUR CODE ##########   
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            
            ########## END OF YOUR CODE ##########  
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def run_vectorized(self, episodes=1000):
        '''
        使用 run_vectorized 去支援多核心 CPU 運算，讓訓練速度加快
        '''
        N = self.num_envs
        print(f"=== {N} CORES TRAINING START===")
        obs_batch, _ = self.vec_env.reset() # 重置多個環境

        # NoopReset: apply random noop actions across the batch for initial state diversity (Atari only)
        if self.is_atari:
            num_noops = random.randint(0, 30)
            noop_actions = np.zeros(N, dtype=np.int64)
            for _ in range(num_noops):
                obs_batch, _, _, _, _ = self.vec_env.step(noop_actions)

        states = [self.vec_preprocessors[i].reset(obs_batch[i]) for i in range(N)] # 初始化多環境狀態
        ep_rewards = np.zeros(N, dtype=np.float32) # 初始化多個環境獎勵
        ep_count = 0 # 初始化 episode 計數

        while ep_count < episodes: 
            actions = np.array(self.select_action_batch(states), dtype=np.int64)
            next_obs_batch, rewards, terminateds, truncateds, infos = self.vec_env.step(actions)
            self.env_count += N

            # Save milestone checkpoints when env_count crosses target thresholds
            for ms in list(self.milestone_steps):
                if ms not in self.saved_milestones and self.env_count >= ms:
                    path = os.path.join(self.save_dir, f"model_{ms}.pt")
                    torch.save(self.q_net.state_dict(), path)
                    print(f"[Milestone] Saved {path} (env_count={self.env_count})")
                    self.saved_milestones.add(ms)

            for i in range(N):
                done = bool(terminateds[i]) or bool(truncateds[i])

                if done:
                    next_state = self.vec_preprocessors[i].reset(next_obs_batch[i])
                else:
                    next_state = self.vec_preprocessors[i].step(next_obs_batch[i])

                transition = (states[i], int(actions[i]), float(rewards[i]), next_state, done)

                if self.use_multistep:
                    self.nstep_buffers[i].append(transition)

                    # Once n transitions are accumulated, emit one n-step transition
                    if len(self.nstep_buffers[i]) >= self.n_step:
                        buf = self.nstep_buffers[i]
                        # R = r0 + γ*r1 + ... + γ^(n-1)*r_{n-1}
                        # Episode boundary is handled by the flush below — no cross-episode mixing
                        R      = sum(self.gamma ** k * buf[k][2] for k in range(self.n_step))
                        s0, a0 = buf[0][0], buf[0][1]       # state and action at t=0
                        sn, dn = buf[-1][3], buf[-1][4]     # next_state, done at t=n
                        t = (s0, a0, R, sn, dn)
                        self.memory.add(t, error=1.0) if self.use_per else self.memory.append(t)
                        self.nstep_buffers[i].popleft()     # slide window forward by 1

                    # Episode ended: flush remaining transitions (shorter than n steps)
                    if done:
                        while self.nstep_buffers[i]:
                            buf     = self.nstep_buffers[i]
                            n_rem   = len(buf)
                            R       = sum(self.gamma ** k * buf[k][2] for k in range(n_rem))
                            s0, a0  = buf[0][0], buf[0][1]
                            sn, dn  = buf[-1][3], buf[-1][4]
                            t = (s0, a0, R, sn, dn)
                            self.memory.add(t, error=1.0) if self.use_per else self.memory.append(t)
                            self.nstep_buffers[i].popleft()
                else:
                    # No multi-step: store single-step transition directly
                    self.memory.add(transition, error=1.0) if self.use_per else self.memory.append(transition)

                ep_rewards[i] += rewards[i]

                if done:
                    total_reward = float(ep_rewards[i])
                    ep_rewards[i] = 0.0

                    print(f"[Eval] Ep: {ep_count} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep_count,
                        "Total Reward": total_reward,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })

                    if ep_count % 100 == 0:
                        model_path = os.path.join(self.save_dir, f"model_ep{ep_count}.pt")
                        torch.save(self.q_net.state_dict(), model_path)
                        print(f"Saved model checkpoint to {model_path}")

                    if ep_count % 20 == 0:
                        eval_reward = self.evaluate()
                        if eval_reward > self.best_reward:
                            self.best_reward = eval_reward
                            model_path = os.path.join(self.save_dir, "best_model.pt")
                            torch.save(self.q_net.state_dict(), model_path)
                            print(f"Saved new best model to {model_path} with reward {eval_reward}")
                        print(f"[TrueEval] Ep: {ep_count} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                        wandb.log({
                            "Env Step Count": self.env_count,
                            "Update Count": self.train_count,
                            "Eval Reward": eval_reward
                        })

                    ep_count += 1
                    if ep_count >= episodes:
                        break

                states[i] = next_state

            for _ in range(self.train_per_step):
                self.train()

            if self.env_count % 1000 < N:
                print(f"[Collect] SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Epsilon": self.epsilon
                })

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).unsqueeze(0).to(self.device).float()
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self.preprocessor.step(next_obs)

        return total_reward


    def train(self):

        if len(self.memory) < self.replay_start_size:
            return 
        
        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1

        # PER β annealing: linearly increase from 0.4 → 1.0 over full training budget (2.5M env steps)
        # Slow annealing keeps IS correction gentle in early/mid training, full correction only at end
        if self.use_per:
            self.memory.beta = min(1.0, 0.4 + (1.0 - 0.4) * (self.env_count / 600_000))
       
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch — PER returns (batch, indices, IS weights); uniform returns batch only
        if self.use_per:
            batch, per_indices, per_weights = self.memory.sample(self.batch_size)
            per_weights = torch.tensor(per_weights, dtype=torch.float32).to(self.device)
        else:
            batch = random.sample(self.memory, self.batch_size)
            per_indices, per_weights = None, None
        states, actions, rewards, next_states, dones = zip(*batch)
        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        # Send as uint8 to GPU first, then convert to float32 on GPU (4x less CPU memory alloc)
        states = torch.from_numpy(np.array(states)).to(self.device).float()
        next_states = torch.from_numpy(np.array(next_states)).to(self.device).float()
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates
        with torch.no_grad():
            if self.use_ddqn:
                # Double DQN: online net selects action, target net evaluates Q value.
                # Decoupling reduces overestimation bias present in vanilla DQN.
                next_actions  = self.q_net(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q_values = self.target_net(next_states).max(1)[0]

            # Multi-step: discount over n steps since rewards already accumulates n-step return.
            # When use_multistep=False, n_step=1 so gamma^1 = gamma (vanilla behavior).
            gamma_n         = self.gamma ** self.n_step
            target_q_values = rewards + gamma_n * next_q_values * (1 - dones)

        td_errors = q_values - target_q_values  # shape: (batch_size,)

        if self.use_per:
            # PER weighted loss: IS weights correct the bias from non-uniform sampling
            loss = (per_weights * td_errors ** 2).mean()
            # Update SumTree priorities with latest TD errors
            self.memory.update_priorities(per_indices, td_errors.detach().abs().cpu().numpy())
        else:
            loss = nn.functional.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        ########## END OF YOUR CODE ##########

        # Soft target update (Polyak averaging): smoother than hard copy every N steps.
        # tau=0.005 → effective lag ~200 train steps, removes Q-function jumps that cause eval dips.
        tau = 0.005
        with torch.no_grad():
            for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                tp.data.mul_(1 - tau).add_(p.data, alpha=tau)

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f} Grad norm: {grad_norm:.4f}")


if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.9999925)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    # Task 3 enhancement flags — toggle each technique independently for ablation study
    parser.add_argument("--use-ddqn",      action="store_true", help="Enable Double DQN")
    parser.add_argument("--use-per",       action="store_true", help="Enable Prioritized Experience Replay")
    parser.add_argument("--use-multistep", action="store_true", help="Enable n-step return (requires --num-envs > 1)")
    parser.add_argument("--n-step",        type=int,   default=3,   help="Steps for multi-step return")
    parser.add_argument("--per-alpha",     type=float, default=0.6, help="PER prioritization exponent")
    parser.add_argument("--per-beta",      type=float, default=0.4, help="PER IS weight correction exponent")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if "ALE/" in args.env:
        project_name = "DLP-Lab5-DQN-" + args.env.split("/")[-1].split("-")[0]
    else:
        project_name = "DLP-Lab5-DQN-" + args.env.split("-")[0]

    wandb.init(project=project_name, name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(env_name=args.env, args=args)
    agent.run(episodes=args.episodes)
    wandb.finish()