import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from tqdm import tqdm
from gymnasium.vector import AsyncVectorEnv
import copy
import time


# Add a simple profiler class to track time during training
class SimpleProfiler:
    def __init__(self):
        self.timings = {}
        self.counts = {}
        self.active_timers = {}

    def start(self, name):
        """Start timing a section of code"""
        self.active_timers[name] = time.time()

    def stop(self, name):
        """Stop timing a section of code"""
        if name in self.active_timers:
            elapsed = time.time() - self.active_timers[name]
            if name not in self.timings:
                self.timings[name] = 0
                self.counts[name] = 0
            self.timings[name] += elapsed
            self.counts[name] += 1
            del self.active_timers[name]

    def get_stats(self):
        """Get timing statistics"""
        stats = {}
        total_time = sum(self.timings.values())

        for name, timing in self.timings.items():
            avg_time = timing / max(1, self.counts[name])
            percentage = (timing / total_time) * 100 if total_time > 0 else 0
            stats[name] = {
                "total": timing,
                "calls": self.counts[name],
                "avg": avg_time,
                "percentage": percentage,
            }
        return stats

    def reset(self):
        """Reset all timings"""
        self.timings = {}
        self.counts = {}
        self.active_timers = {}

    def print_stats(self):
        """Print timing statistics in a readable format"""
        stats = self.get_stats()
        if not stats:
            print("No profiling data collected")
            return

        print("\n===== TRAINING PERFORMANCE PROFILE =====")
        print(
            f"{'Section':<25} {'Total (s)':<10} {'Calls':<8} {'Avg (ms)':<10} {'%':<8}"
        )
        print("-" * 65)

        # Sort by total time spent
        for name, data in sorted(
            stats.items(), key=lambda x: x[1]["total"], reverse=True
        ):
            print(
                f"{name:<25} {data['total']:<10.2f} {data['calls']:<8} {data['avg']*1000:<10.2f} {data['percentage']:<8.2f}"
            )
        print("========================================\n")


class RandomAgent:
    """
    A random agent that selects actions uniformly at random.
    """

    def __init__(self, action_dim):
        self.action_dim = action_dim

    def select_action(self, state):
        action = np.random.randint(self.action_dim)
        log_prob = None  # Not used for random agent
        return action, log_prob


class PolicyNetwork(nn.Module):
    """
    A policy network that outputs a probability distribution over actions.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Use larger layers for better GPU utilization
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),  # Extra layer
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


class ValueNetwork(nn.Module):
    """
    A value network that outputs a single value for a given state.
    """

    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        # Use larger layers for better GPU utilization
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),  # Extra layer
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class PPOAgent:
    """
    A PPO agent that uses a policy network and a value network to select actions and compute returns.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=512,  # Increased default hidden dim
        gamma=0.99,
        clip_epsilon=0.2,
        learning_rate=1e-4,
        device=None,
        config=None,
    ):
        # If config is provided, use its values
        if config:
            # Allow for configurable hidden_dim with larger default
            hidden_dim = getattr(config, "hidden_dim", hidden_dim)
            gamma = config.gamma
            clip_epsilon = config.clip_epsilon
            learning_rate = config.learning_rate

            # Use larger hidden dimension for large batch sizes
            if hasattr(config, "batch_size") and config.batch_size >= 1024:
                # Scale up hidden dim for large batch sizes to better utilize GPU
                if hidden_dim < 768:  # Only scale up if not already large
                    hidden_dim = max(hidden_dim, 768)  # At least 768 for large batches
                    print(
                        f"Increasing network size to hidden_dim={hidden_dim} for better GPU utilization"
                    )

            # Adjust gradient clipping threshold based on batch size
            if hasattr(config, "clip_grad_norm"):
                # Use explicitly configured value if available
                self.grad_clip_val = config.clip_grad_norm
                print(
                    f"Using gradient clipping threshold of {self.grad_clip_val} from config"
                )
            elif hasattr(config, "batch_size"):
                if config.batch_size >= 2048:
                    self.grad_clip_val = 5.0  # Much higher for very large batches
                    print(
                        f"Using higher gradient clipping threshold of {self.grad_clip_val} for large batch size"
                    )
                elif config.batch_size >= 1024:
                    self.grad_clip_val = 2.0  # Higher for large batches
                    print(
                        f"Using higher gradient clipping threshold of {self.grad_clip_val} for large batch size"
                    )
                else:
                    self.grad_clip_val = 0.5  # Standard PPO clipping threshold
            else:
                self.grad_clip_val = 0.5  # Default value

        else:
            self.grad_clip_val = 0.5  # Default value for gradient clipping

        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Using device: {self.device}")
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim=hidden_dim).to(
            self.device
        )
        self.value = ValueNetwork(state_dim, hidden_dim=hidden_dim).to(self.device)

        # Create optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=learning_rate)

        # Add learning rate scheduler for large batch sizes
        self.use_lr_scheduler = False
        if hasattr(config, "batch_size") and config.batch_size >= 1024:
            # For large batch sizes, decrease learning rate over time
            self.policy_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.policy_optimizer,
                mode="max",
                factor=0.5,
                patience=3,
                threshold=0.01,
                threshold_mode="rel",
                cooldown=2,
                min_lr=1e-6,
            )
            self.value_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.value_optimizer,
                mode="min",
                factor=0.5,
                patience=3,
                threshold=0.01,
                threshold_mode="rel",
                cooldown=2,
                min_lr=1e-6,
            )
            self.use_lr_scheduler = True
            print(f"Using learning rate scheduler for large batch size")

        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Save config if provided
        self.config = config

        # Add these for tracking across all epochs
        self.total_training_steps = 0
        self.total_training_reward = 0
        self.all_epoch_rewards = []
        self.all_episode_rewards = []

        # Initialize profiler
        self.profiler = SimpleProfiler()
        self.enable_profiling = (
            config.enable_profiling if hasattr(config, "enable_profiling") else False
        )

        # Enable mixed precision for faster GPU processing if available
        self.use_mixed_precision = torch.cuda.is_available() and hasattr(torch, "amp")
        if hasattr(config, "use_mixed_precision"):
            self.use_mixed_precision = config.use_mixed_precision

        if self.use_mixed_precision:
            self.scaler = torch.amp.GradScaler()
            print("Using mixed precision training with CUDA")

    def select_action(self, state):
        # Set to evaluation mode for inference
        self.policy.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            probs = self.policy(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        return action.item(), dist.log_prob(action)

    def select_actions_vectorized(self, states):
        """Select actions for multiple states in parallel"""
        # Set to evaluation mode for inference
        self.policy.eval()
        with torch.no_grad():
            # states is already a batch
            states_tensor = torch.FloatTensor(states).to(self.device)
            probs = self.policy(states_tensor)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
        return actions.cpu().numpy(), log_probs.cpu().numpy()

    def collect_trajectories_cuda(
        self,
        env_fn,
        n_envs=8,
        steps_per_env=2048,
        disable_progress=False,
    ):
        """
        CUDA-optimized trajectory collection that processes batched environment data more efficiently

        Args:
            env_fn: Function that creates a single environment instance
            n_envs: Number of parallel environments
            steps_per_env: Number of steps to collect per environment
            disable_progress: Whether to disable the progress bar

        Returns:
            Tuple of collected trajectory data (converted to NumPy arrays)
        """
        # Track time if profiling is enabled
        if self.enable_profiling:
            self.profiler.start("collect_trajectories")

        # Create vector of environments
        if self.enable_profiling:
            self.profiler.start("env_creation")

        env_fns = [env_fn for _ in range(n_envs)]
        envs = AsyncVectorEnv(env_fns)

        if self.enable_profiling:
            self.profiler.stop("env_creation")

        # Initialize storage directly as CUDA tensors for more efficient processing
        total_steps = n_envs * steps_per_env

        # Track tensor allocation time
        if self.enable_profiling:
            self.profiler.start("tensor_allocation")

        # Use device directly without torch.cuda.device context manager (which is deprecated)
        # Allocate tensors directly on the device
        states = torch.zeros(
            (total_steps, self.state_dim), dtype=torch.float32, device=self.device
        )
        actions = torch.zeros(total_steps, dtype=torch.int64, device=self.device)
        rewards = torch.zeros(total_steps, dtype=torch.float32, device=self.device)
        dones = torch.zeros(total_steps, dtype=torch.bool, device=self.device)
        next_states = torch.zeros(
            (total_steps, self.state_dim), dtype=torch.float32, device=self.device
        )
        log_probs = torch.zeros(total_steps, dtype=torch.float32, device=self.device)

        if self.enable_profiling:
            self.profiler.stop("tensor_allocation")

        # CPU versions for environment interaction
        cpu_episode_rewards = np.zeros(n_envs)
        cpu_episode_lengths = np.zeros(n_envs, dtype=int)
        episode_lengths = []

        # Reset all environments
        if self.enable_profiling:
            self.profiler.start("env_reset")

        obs, _ = envs.reset()

        if self.enable_profiling:
            self.profiler.stop("env_reset")

        # Pre-allocate pinned memory buffer for faster transfers
        pin_memory = torch.cuda.is_available()
        obs_buffer = torch.zeros((n_envs, self.state_dim), dtype=torch.float32)
        if pin_memory:
            obs_buffer = obs_buffer.pin_memory()

        # Only show progress bar if not disabled
        if disable_progress:
            pbar = None
            iterator = range(steps_per_env)
        else:
            pbar = tqdm(total=steps_per_env, desc="Collecting experience")
            iterator = pbar

        # Use eval mode during data collection for better performance
        self.policy.eval()

        # Prepare for batch processing
        batch_size = min(n_envs, 64)  # Process in chunks to avoid CUDA OOM

        # Track step time
        total_env_step_time = 0
        total_inference_time = 0
        total_storage_time = 0
        steps_tracked = 0

        for t in iterator:
            # Get actions for all environments
            # Convert observations to CUDA tensor efficiently using pinned memory
            obs_buffer.copy_(torch.FloatTensor(obs))

            # Process in batches if number of environments is large
            actions_list = []
            log_probs_list = []

            # Track inference time
            if self.enable_profiling:
                self.profiler.start("policy_inference")

            for i in range(0, n_envs, batch_size):
                end_idx = min(i + batch_size, n_envs)
                # Get CUDA tensor efficiently
                obs_tensor = obs_buffer[i:end_idx].to(self.device, non_blocking=True)

                # Get actions (keeping everything on GPU)
                with torch.no_grad():
                    with torch.amp.autocast(
                        device_type="cuda", enabled=self.use_mixed_precision
                    ):
                        probs = self.policy(obs_tensor)
                        dist = torch.distributions.Categorical(probs)
                        action_tensor = dist.sample()
                        log_prob_tensor = dist.log_prob(action_tensor)

                actions_list.append(action_tensor)
                log_probs_list.append(log_prob_tensor)

            # Combine batch results
            action_tensor = torch.cat(actions_list)
            log_prob_tensor = torch.cat(log_probs_list)

            if self.enable_profiling:
                self.profiler.stop("policy_inference")

            # Convert back to CPU for environment stepping
            action_values = action_tensor.cpu().numpy()

            # Step environments
            if self.enable_profiling:
                self.profiler.start("env_step")

            env_step_start = time.time()
            next_obs, reward, term, trunc, infos = envs.step(action_values)
            env_step_time = time.time() - env_step_start
            total_env_step_time += env_step_time
            steps_tracked += 1

            if self.enable_profiling:
                self.profiler.stop("env_step")

            done = np.logical_or(term, trunc)

            # Store data (directly in CUDA tensors)
            start_idx = t * n_envs
            end_idx = start_idx + n_envs

            # Track storage time
            if self.enable_profiling:
                self.profiler.start("tensor_storage")

            storage_start = time.time()
            # Efficient storage to GPU tensors
            states[start_idx:end_idx].copy_(
                obs_buffer.to(self.device, non_blocking=True)
            )
            actions[start_idx:end_idx].copy_(action_tensor)
            rewards[start_idx:end_idx].copy_(
                torch.FloatTensor(reward).to(self.device, non_blocking=True)
            )
            dones[start_idx:end_idx].copy_(
                torch.BoolTensor(done).to(self.device, non_blocking=True)
            )
            # Pre-copy next_obs to buffer for next iteration
            obs_buffer.copy_(torch.FloatTensor(next_obs))
            next_states[start_idx:end_idx].copy_(
                obs_buffer.to(self.device, non_blocking=True)
            )
            log_probs[start_idx:end_idx].copy_(log_prob_tensor)
            storage_time = time.time() - storage_start
            total_storage_time += storage_time

            if self.enable_profiling:
                self.profiler.stop("tensor_storage")

            # Update episode tracking (keep on CPU for efficiency)
            cpu_episode_rewards += reward
            cpu_episode_lengths += 1

            # Handle completed episodes
            for i in range(n_envs):
                if done[i]:
                    self.all_episode_rewards.append(float(cpu_episode_rewards[i]))
                    episode_lengths.append(int(cpu_episode_lengths[i]))
                    cpu_episode_rewards[i] = 0
                    cpu_episode_lengths[i] = 0

            # Update state
            obs = next_obs

            # Update progress bar if it exists
            if pbar:
                pbar.update(1)

            # Store info from last step
            if t == steps_per_env - 1:
                # Store the last info safely
                if isinstance(infos, dict):
                    # Handle dictionary type infos (common for vectorized envs)
                    self.last_info = {}
                    for key, values in infos.items():
                        if isinstance(values, np.ndarray) and values.size > 0:
                            # Take the first item if it's an array
                            self.last_info[key] = (
                                float(values[0])
                                if np.issubdtype(values.dtype, np.number)
                                else values[0]
                            )
                        else:
                            self.last_info[key] = (
                                float(values)
                                if isinstance(values, np.number)
                                else values
                            )
                elif isinstance(infos, list) and len(infos) > 0:
                    # Handle list type infos
                    self.last_info = infos[0]
                else:
                    # Fallback
                    self.last_info = infos

        # Close progress bar if it exists
        if pbar:
            pbar.close()

        # Calculate and store average times
        if steps_tracked > 0 and self.enable_profiling:
            avg_env_step_time = total_env_step_time / steps_tracked
            avg_storage_time = total_storage_time / steps_tracked
            # Store in profiler as separate entries
            self.profiler.timings["avg_env_step"] = avg_env_step_time
            self.profiler.counts["avg_env_step"] = 1
            self.profiler.timings["avg_tensor_storage"] = avg_storage_time
            self.profiler.counts["avg_tensor_storage"] = 1

        envs.close()

        # Set last_episode_rewards for backward compatibility
        self.last_episode_rewards = (
            [float(r) for r in self.all_episode_rewards[-n_envs:]]
            if self.all_episode_rewards
            else []
        )

        # Convert CUDA tensors to NumPy arrays before returning
        if self.enable_profiling:
            self.profiler.start("tensor_to_numpy")

        result = (
            states.cpu().numpy(),
            actions.cpu().numpy(),
            rewards.cpu().numpy(),
            dones.cpu().numpy(),
            next_states.cpu().numpy(),
            log_probs.cpu().numpy(),
            episode_lengths,
        )

        if self.enable_profiling:
            self.profiler.stop("tensor_to_numpy")
            self.profiler.stop("collect_trajectories")

        return result

    def collect_trajectories_parallel(
        self,
        env_fn,
        n_envs=8,  # Default value, can be overridden by config
        steps_per_env=2048,
        disable_progress=False,  # Add parameter to disable progress bar
    ):
        """
        Collect trajectories from multiple environments in parallel.

        Args:
            env_fn: Function that creates a single environment instance
            n_envs: Number of parallel environments
            steps_per_env: Number of steps to collect per environment
            disable_progress: Whether to disable the progress bar

        Returns:
            Tuple of collected trajectory data
        """
        # Use the single environment version if n_envs is 1 or GPU isn't available
        if n_envs == 1 or not torch.cuda.is_available():
            # Create a single environment
            env = env_fn()

            # Collect data using the single-environment method
            return self.collect_trajectories(
                env,
                num_steps=steps_per_env,
                display=False,
            )

        # Call optimized CUDA version if using GPU and multiple environments
        if torch.cuda.is_available() and n_envs > 1:
            return self.collect_trajectories_cuda(
                env_fn, n_envs, steps_per_env, disable_progress
            )

        # Original CPU implementation for multiple environments
        env_fns = [env_fn for _ in range(n_envs)]
        envs = AsyncVectorEnv(env_fns)

        # Initialize storage
        total_steps = n_envs * steps_per_env
        states = np.zeros((total_steps, self.state_dim), dtype=np.float32)
        actions = np.zeros(total_steps, dtype=np.int64)
        rewards = np.zeros(total_steps, dtype=np.float32)
        dones = np.zeros(total_steps, dtype=bool)
        next_states = np.zeros((total_steps, self.state_dim), dtype=np.float32)
        log_probs = np.zeros(total_steps, dtype=np.float32)
        episode_lengths = []

        # Reset all environments
        obs, _ = envs.reset()

        # Collect data
        step_idx = 0
        episode_reward = np.zeros(n_envs)
        episode_length = np.zeros(n_envs, dtype=int)

        # Only show progress bar if not disabled
        if disable_progress:
            pbar = None
            iterator = range(steps_per_env)
        else:
            pbar = tqdm(total=steps_per_env, desc="Collecting experience")
            iterator = pbar

        for t in iterator:
            # Get actions for all environments
            action_values, log_prob_values = self.select_actions_vectorized(obs)

            # Step environments
            next_obs, reward, term, trunc, infos = envs.step(action_values)
            done = np.logical_or(term, trunc)

            # Store data
            start_idx = t * n_envs
            end_idx = start_idx + n_envs

            states[start_idx:end_idx] = obs
            actions[start_idx:end_idx] = action_values
            rewards[start_idx:end_idx] = reward
            dones[start_idx:end_idx] = done
            next_states[start_idx:end_idx] = next_obs
            log_probs[start_idx:end_idx] = log_prob_values

            # Update episode tracking
            episode_reward += reward
            episode_length += 1

            # Handle completed episodes
            for i in range(n_envs):
                if done[i]:
                    self.all_episode_rewards.append(
                        float(episode_reward[i])
                    )  # Convert to Python float
                    episode_lengths.append(
                        int(episode_length[i])
                    )  # Convert to Python int
                    episode_reward[i] = 0
                    episode_length[i] = 0

            # Update state
            obs = next_obs
            step_idx += n_envs

            # Update progress bar if it exists
            if pbar:
                pbar.update(1)

            # Store info from last step
            if t == steps_per_env - 1:
                # Store the last info safely
                if isinstance(infos, dict):
                    # Handle dictionary type infos (common for vectorized envs)
                    self.last_info = {}
                    for key, values in infos.items():
                        if isinstance(values, np.ndarray) and values.size > 0:
                            # Take the first item if it's an array
                            self.last_info[key] = (
                                float(values[0])
                                if np.issubdtype(values.dtype, np.number)
                                else values[0]
                            )
                        else:
                            self.last_info[key] = (
                                float(values)
                                if isinstance(values, np.number)
                                else values
                            )
                elif isinstance(infos, list) and len(infos) > 0:
                    # Handle list type infos
                    self.last_info = infos[0]
                else:
                    # Fallback
                    self.last_info = infos

        # Close progress bar if it exists
        if pbar:
            pbar.close()

        envs.close()

        # Set last_episode_rewards for backward compatibility
        self.last_episode_rewards = (
            [float(r) for r in self.all_episode_rewards[-n_envs:]]
            if self.all_episode_rewards
            else []
        )

        return states, actions, rewards, dones, next_states, log_probs, episode_lengths

    def compute_returns(self, rewards, dones, next_value):
        # Convert inputs to tensors if they aren't already
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.FloatTensor(rewards).to(self.device)
        if not isinstance(dones, torch.Tensor):
            dones = torch.BoolTensor(dones).to(self.device)
        if not isinstance(next_value, torch.Tensor):
            next_value = torch.tensor(next_value, dtype=torch.float32).to(self.device)

        # Handle empty arrays
        if len(rewards) == 0:
            return torch.tensor([], dtype=torch.float32).to(self.device)

        # Pre-allocate returns tensor
        returns = torch.zeros_like(rewards).to(self.device)

        # Initialize with next_value
        R = next_value

        # Calculate returns in reverse order
        for i in range(len(rewards) - 1, -1, -1):
            # Reset return if episode ended
            R = rewards[i] + self.gamma * R * (1 - dones[i].float())
            returns[i] = R

        return returns

    def update(
        self,
        states,
        actions,
        rewards,
        dones,
        next_states,
        old_log_probs,
        batch_size=128,
        epochs=10,
    ):
        # Track time if profiling is enabled
        if self.enable_profiling:
            self.profiler.start("update")

        # Use config values if available
        if self.config:
            batch_size = self.config.batch_size
            epochs = self.config.ppo_epochs
            entropy_coef = self.config.entropy_coef
            value_coef = self.config.value_coef
        else:
            entropy_coef = 0.01
            value_coef = 0.5

        # Initialize tracking variables for wandb
        self.last_value_loss = 0
        self.last_policy_loss = 0
        self.last_entropy = 0
        self.last_clip_fraction = 0
        self.last_learning_rate = self.policy_optimizer.param_groups[0]["lr"]
        value_loss_sum = 0
        policy_loss_sum = 0
        entropy_sum = 0
        clip_count = 0
        total_count = 0
        update_count = 0

        # Set to training mode for updates
        self.policy.train()
        self.value.train()

        # Update total training metrics
        if not isinstance(rewards, torch.Tensor):
            rewards_tensor = torch.FloatTensor(rewards)
        else:
            rewards_tensor = rewards.cpu()  # Make sure it's on CPU for numpy conversion

        # Update cumulative training statistics
        batch_reward = rewards_tensor.sum().item()
        self.total_training_steps += len(rewards)
        self.total_training_reward += batch_reward

        # Store epoch reward
        self.all_epoch_rewards.append(batch_reward)

        # Process data - convert to tensors and move to device only once
        if self.enable_profiling:
            self.profiler.start("data_to_device")

        if not isinstance(states, torch.Tensor):
            # Use non_blocking for concurrent CPU-GPU transfer
            states = torch.FloatTensor(np.array(states)).to(
                self.device, non_blocking=True
            )
            actions = torch.LongTensor(np.array(actions)).to(
                self.device, non_blocking=True
            )
            old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(
                self.device, non_blocking=True
            )
            rewards = torch.FloatTensor(np.array(rewards)).to(
                self.device, non_blocking=True
            )
            dones = torch.BoolTensor(np.array(dones)).to(self.device, non_blocking=True)
            next_states = torch.FloatTensor(np.array(next_states)).to(
                self.device, non_blocking=True
            )

        if self.enable_profiling:
            self.profiler.stop("data_to_device")

        # Compute returns and advantages with tensors
        if self.enable_profiling:
            self.profiler.start("compute_advantages")

        with torch.no_grad():
            # Use mixed precision for performance if available
            with torch.amp.autocast(
                device_type="cuda", enabled=self.use_mixed_precision
            ):
                next_values = self.value(next_states).squeeze(-1)

        # Using compute_returns with tensors
        returns = self.compute_returns(
            rewards, dones, next_values[-1] if len(next_values) > 0 else 0
        )

        # Calculate advantages
        with torch.no_grad():
            with torch.amp.autocast(
                device_type="cuda", enabled=self.use_mixed_precision
            ):
                values = self.value(states).squeeze(-1)
                advantages = returns - values

        # Normalize advantages (reduces variance)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if self.enable_profiling:
            self.profiler.stop("compute_advantages")

        # Determine optimal batch size based on available memory
        # Larger batch sizes improve GPU utilization
        sample_count = len(states)

        # Check if batch size was manually set
        batch_size_manually_set = (
            hasattr(self.config, "batch_size_manually_set")
            and self.config.batch_size_manually_set
        )

        # For very large samples, adjust batch size and epochs for better GPU utilization
        if sample_count > 10000 and not batch_size_manually_set:
            # For large samples, use larger batches (only if not manually set)
            batch_size = max(batch_size, min(2048, sample_count // 8))

        # When using very large batch sizes (manually set), adjust number of PPO epochs
        if batch_size >= 1024 and epochs > 5:
            orig_epochs = epochs
            epochs = min(epochs, 5)  # Reduce number of epochs for very large batches
            if self.enable_profiling:
                print(
                    f"Large batch size detected ({batch_size}), reducing PPO epochs from {orig_epochs} to {epochs}"
                )

        # Calculate number of batches
        num_batches = (sample_count + batch_size - 1) // batch_size

        # PPO update for multiple epochs
        if self.enable_profiling:
            self.profiler.start("ppo_epochs")

        for epoch_idx in range(epochs):
            # Generate random mini-batches
            indices = torch.randperm(sample_count, device=self.device)

            # Process in mini-batches
            for batch_idx in range(num_batches):
                if self.enable_profiling:
                    self.profiler.start("ppo_batch")

                # Get batch indices
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, sample_count)
                idx = indices[start_idx:end_idx]

                # Get mini-batch
                mini_states = states[idx]
                mini_actions = actions[idx]
                mini_advantages = advantages[idx]
                mini_returns = returns[idx]
                mini_old_log_probs = old_log_probs[idx]

                # Use mixed precision training if available
                if self.use_mixed_precision:
                    if self.enable_profiling:
                        self.profiler.start("forward_pass")

                    with torch.amp.autocast(device_type="cuda"):
                        # Get new log probs and values with mixed precision
                        current_probs = self.policy(mini_states)
                        dist = torch.distributions.Categorical(current_probs)
                        current_log_probs = dist.log_prob(mini_actions)
                        entropy = dist.entropy().mean()
                        current_values = self.value(mini_states).squeeze(-1)

                        # Compute ratio for PPO
                        ratio = torch.exp(current_log_probs - mini_old_log_probs)

                        # Compute PPO losses
                        surrogate1 = ratio * mini_advantages
                        surrogate2 = (
                            torch.clamp(
                                ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                            )
                            * mini_advantages
                        )

                        # Policy loss (negative because we're doing gradient ascent)
                        policy_loss = -torch.min(surrogate1, surrogate2).mean()

                        # Value loss
                        value_loss = ((current_values - mini_returns) ** 2).mean()

                        # Entropy bonus (encourages exploration)
                        entropy_loss = -entropy_coef * entropy

                        # Track for wandb
                        policy_loss_sum += policy_loss.item()
                        value_loss_sum += value_loss.item()
                        entropy_sum += entropy.item()
                        update_count += 1

                        # Total loss
                        total_loss = (
                            policy_loss + value_coef * value_loss + entropy_loss
                        )

                    if self.enable_profiling:
                        self.profiler.stop("forward_pass")
                        self.profiler.start("backward_pass")

                    # Update with mixed precision
                    self.policy_optimizer.zero_grad(
                        set_to_none=True
                    )  # More efficient than just zero_grad()
                    self.value_optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(total_loss).backward()

                    # Clip gradients for stability
                    self.scaler.unscale_(self.policy_optimizer)
                    self.scaler.unscale_(self.value_optimizer)

                    # Track gradient clipping statistics
                    policy_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.grad_clip_val
                    )
                    value_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.value.parameters(), self.grad_clip_val
                    )

                    # Count clipped gradients - convert to float if tensor
                    if (
                        (
                            isinstance(policy_grad_norm, torch.Tensor)
                            and policy_grad_norm.item() > self.grad_clip_val
                        )
                        or (
                            not isinstance(policy_grad_norm, torch.Tensor)
                            and policy_grad_norm > self.grad_clip_val
                        )
                        or (
                            isinstance(value_grad_norm, torch.Tensor)
                            and value_grad_norm.item() > self.grad_clip_val
                        )
                        or (
                            not isinstance(value_grad_norm, torch.Tensor)
                            and value_grad_norm > self.grad_clip_val
                        )
                    ):
                        clip_count += 1
                    total_count += 1

                    # Update with scaled gradients
                    self.scaler.step(self.policy_optimizer)
                    self.scaler.step(self.value_optimizer)
                    self.scaler.update()

                    if self.enable_profiling:
                        self.profiler.stop("backward_pass")

                else:
                    # Standard training without mixed precision
                    if self.enable_profiling:
                        self.profiler.start("forward_pass")

                    # Get new log probs and values
                    current_probs = self.policy(mini_states)
                    dist = torch.distributions.Categorical(current_probs)
                    current_log_probs = dist.log_prob(mini_actions)
                    entropy = dist.entropy().mean()

                    current_values = self.value(mini_states).squeeze(-1)

                    # Compute ratio for PPO
                    ratio = torch.exp(current_log_probs - mini_old_log_probs)

                    # Compute PPO losses
                    surrogate1 = ratio * mini_advantages
                    surrogate2 = (
                        torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                        * mini_advantages
                    )

                    # Policy loss (negative because we're doing gradient ascent)
                    policy_loss = -torch.min(surrogate1, surrogate2).mean()

                    # Value loss
                    value_loss = ((current_values - mini_returns) ** 2).mean()

                    # Entropy bonus (encourages exploration)
                    entropy_loss = -entropy_coef * entropy

                    # Track for wandb
                    policy_loss_sum += policy_loss.item()
                    value_loss_sum += value_loss.item()
                    entropy_sum += entropy.item()
                    update_count += 1

                    # Total loss
                    total_loss = policy_loss + value_coef * value_loss + entropy_loss

                    if self.enable_profiling:
                        self.profiler.stop("forward_pass")
                        self.profiler.start("backward_pass")

                    # Update networks
                    self.policy_optimizer.zero_grad(set_to_none=True)
                    self.value_optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()

                    # Clip gradients for stability
                    policy_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.grad_clip_val
                    )
                    value_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.value.parameters(), self.grad_clip_val
                    )

                    # Count clipped gradients - convert to float if tensor
                    if (
                        (
                            isinstance(policy_grad_norm, torch.Tensor)
                            and policy_grad_norm.item() > self.grad_clip_val
                        )
                        or (
                            not isinstance(policy_grad_norm, torch.Tensor)
                            and policy_grad_norm > self.grad_clip_val
                        )
                        or (
                            isinstance(value_grad_norm, torch.Tensor)
                            and value_grad_norm.item() > self.grad_clip_val
                        )
                        or (
                            not isinstance(value_grad_norm, torch.Tensor)
                            and value_grad_norm > self.grad_clip_val
                        )
                    ):
                        clip_count += 1
                    total_count += 1

                    self.policy_optimizer.step()
                    self.value_optimizer.step()

                    if self.enable_profiling:
                        self.profiler.stop("backward_pass")

                if self.enable_profiling:
                    self.profiler.stop("ppo_batch")

        if self.enable_profiling:
            self.profiler.stop("ppo_epochs")
            self.profiler.stop("update")

        # At the end of the update method, adjust learning rate if using scheduler
        if self.use_lr_scheduler:
            # Schedule based on rewards (for policy) and value loss (for value network)
            # Convert rewards to numpy if it's a tensor
            if isinstance(rewards, torch.Tensor):
                rewards_np = rewards.cpu().detach().numpy()
                avg_reward = float(rewards_np.mean()) if len(rewards_np) > 0 else 0.0
            else:
                rewards_np = (
                    np.array(rewards)
                    if not isinstance(rewards, np.ndarray)
                    else rewards
                )
                avg_reward = float(rewards_np.mean()) if len(rewards_np) > 0 else 0.0

            avg_value_loss = float(value_loss_sum / max(1, update_count))

            # Update schedulers
            self.policy_scheduler.step(avg_reward)
            self.value_scheduler.step(avg_value_loss)

            # Update tracked learning rate
            self.last_learning_rate = self.policy_optimizer.param_groups[0]["lr"]

        # Store average losses
        if update_count > 0:
            self.last_policy_loss = policy_loss_sum / update_count
            self.last_value_loss = value_loss_sum / update_count
            self.last_entropy = entropy_sum / update_count

        # Calculate clip fraction
        if total_count > 0:
            self.last_clip_fraction = clip_count / total_count

    def collect_trajectories(
        self, env, num_steps=2048, display=False, window_size=(1024, 768)
    ):
        states, actions, rewards, dones, log_probs, next_states = [], [], [], [], [], []
        episode_rewards = []  # Track rewards per episode
        episode_lengths = []

        state, _ = env.reset()
        done = False
        ep_length = 0
        current_episode_reward = 0  # Track current episode's reward

        self.last_info = None  # Initialize last_info
        self.last_episode_rewards = []  # Initialize with empty list

        for t in range(num_steps):
            # Convert state to tensor and get action
            if isinstance(state, np.ndarray):
                state_tensor = torch.FloatTensor(state).to(self.device)
            else:
                state_tensor = state  # Assume already a tensor

            # Get action from policy
            self.policy.eval()  # Set to evaluation mode for action selection
            with torch.no_grad():
                probs = self.policy(state_tensor.unsqueeze(0))
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            # Convert to scalar values
            action_value = action.item()
            log_prob_value = log_prob.item()

            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action_value)
            done = terminated or truncated

            # Store last info for portfolio value tracking
            self.last_info = info

            # Append to trajectories
            states.append(state)
            actions.append(action_value)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob_value)
            next_states.append(next_state)

            # Update episodic information
            current_episode_reward += reward
            ep_length += 1

            # Handle episode completion
            if done:
                episode_rewards.append(current_episode_reward)
                self.last_episode_rewards.append(current_episode_reward)
                episode_lengths.append(ep_length)

                # Reset episode tracking variables
                state, _ = env.reset()
                current_episode_reward = 0
                ep_length = 0
            else:
                state = next_state

            # Display if requested
            if display:
                frame = env.render()
                if frame is not None:
                    frame = cv2.resize(frame, window_size)
                    cv2.imshow("Environment", frame)
                    cv2.waitKey(1)

        # Store all episode rewards for tracking
        if episode_rewards:  # Only update if we have any completed episodes
            self.all_episode_rewards.extend(episode_rewards)

        return states, actions, rewards, dones, next_states, log_probs, episode_lengths

    def save_model(self, path="saved_model"):
        """Save policy and value networks to files"""
        # Save model data
        policy_state = self.policy.state_dict()
        value_state = self.value.state_dict()

        # Create a dictionary with only the necessary data
        save_dict = {
            "policy_state_dict": policy_state,
            "value_state_dict": value_state,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }

        # Save with appropriate options
        torch.save(save_dict, f"{path}.pt")
        tqdm.write(f"Model saved to {path}.pt")

    @classmethod
    def load_model(cls, path="saved_model"):
        """Load policy and value networks from files"""
        try:
            # Set weights_only=False to handle newer PyTorch 2.6+ behavior
            checkpoint = torch.load(f"{path}.pt", weights_only=False)
            state_dim = checkpoint["state_dim"]
            action_dim = checkpoint["action_dim"]

            # Extract hidden_dim from the checkpoint if possible
            hidden_dim = 512  # Default value
            if "policy_state_dict" in checkpoint:
                # Get the first layer weight to determine hidden dimension
                first_layer = checkpoint["policy_state_dict"]["net.0.weight"]
                if hasattr(first_layer, "shape") and len(first_layer.shape) > 0:
                    hidden_dim = first_layer.shape[0]
                    tqdm.write(f"Detected hidden_dim={hidden_dim} from model")

            # Create agent with correct hidden_dim
            agent = cls(state_dim, action_dim, hidden_dim=hidden_dim)

            # Load state dictionaries
            agent.policy.load_state_dict(checkpoint["policy_state_dict"])
            agent.value.load_state_dict(checkpoint["value_state_dict"])

            tqdm.write(f"Model loaded from {path}.pt")
            return agent
        except FileNotFoundError:
            tqdm.write(f"No saved model found at {path}.pt")
            return None
        except Exception as e:
            tqdm.write(f"Error loading model: {str(e)}")
            tqdm.write("Creating a new model instead.")
            return None

    def print_training_summary(self, eval_rewards=None, overall_stats=None):
        """Print a simple summary of the training performance"""
        print("\nTraining Summary:")

        # Show overall statistics if provided
        if overall_stats is not None:
            print(f"Total Episodes: {overall_stats['total_episodes']}")
            print(f"Overall Average Episode Reward: {overall_stats['avg_reward']:.2f}")
            print(
                f"Perfect Episodes: {overall_stats['perfect_episodes']}/{overall_stats['total_episodes']} ({overall_stats['perfect_percentage']:.1f}%)"
            )
            if "incomplete_episodes" in overall_stats:
                print(
                    f"Incomplete Episodes (cut off at epoch boundaries): {overall_stats['incomplete_episodes']}"
                )

        # Only show evaluation rewards if provided
        if eval_rewards is not None and len(eval_rewards) > 0:
            print(f"Number of Training Epochs: {len(eval_rewards)}")
            print(f"Final Average Reward: {eval_rewards[-1]:.2f}")
            print(f"Peak Average Reward: {max(eval_rewards):.2f}")

            # Calculate improvement
            if len(eval_rewards) > 1:
                first_reward = eval_rewards[0]
                last_reward = eval_rewards[-1]
                improvement = ((last_reward - first_reward) / first_reward) * 100
                print(f"Improvement: {improvement:.2f}%")
        else:
            print("No evaluation metrics available.")

    # Add other methods as needed
