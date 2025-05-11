import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from tqdm import tqdm

# Hyperparameters
HIDDEN_DIM = 256  # Number of neurons in hidden layers
DROPOUT_RATE = 0.1  # Probability of dropping a neuron during training (regularization)
GAMMA = 0.99  # Discount factor for future rewards (closer to 1 = more long-term focus)
CLIP_EPSILON = 0.1  # PPO clipping parameter to limit policy update size
LEARNING_RATE = 1e-3  # Step size for optimizer updates
BATCH_SIZE = 128  # Number of samples processed in each training mini-batch
PPO_EPOCHS = 10  # Number of times to reuse each collected trajectory for updates
ENTROPY_COEF = 0.01  # Coefficient for entropy bonus (higher = more exploration)
VALUE_COEF = 0.5  # Coefficient for value loss in the total loss function


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

    def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


class ValueNetwork(nn.Module):
    """
    A value network that outputs a single value for a given state.
    """

    def __init__(self, state_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
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
        gamma=GAMMA,
        clip_epsilon=CLIP_EPSILON,
        lr=LEARNING_RATE,
        device=None,
    ):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.value = ValueNetwork(state_dim).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Add these for tracking across all epochs
        self.total_training_steps = 0
        self.total_training_reward = 0
        self.all_epoch_rewards = []
        self.all_episode_rewards = []

    def select_action(self, state):
        # Set to evaluation mode for inference
        self.policy.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            probs = self.policy(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        return action.item(), dist.log_prob(action)

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
        batch_size=BATCH_SIZE,
        epochs=PPO_EPOCHS,
    ):
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

        # Convert to tensors and move to device only once
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device)

        # Compute returns and advantages with tensors
        with torch.no_grad():
            next_values = self.value(
                torch.FloatTensor(np.array(next_states)).to(self.device)
            ).squeeze(-1)

        # Our compute_returns now returns a tensor, so no need to convert back and forth
        returns = self.compute_returns(
            rewards, dones, next_values[-1] if len(next_values) > 0 else 0
        )

        # Calculate advantages
        advantages = returns - self.value(states).squeeze(-1).detach()

        # Normalize advantages (reduces variance)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update for multiple epochs
        for _ in range(epochs):
            # Generate random mini-batches
            indices = torch.randperm(len(states))
            for start_idx in range(0, len(states), batch_size):
                # Get mini-batch
                idx = indices[start_idx : start_idx + batch_size]

                # Get new log probs and values
                current_probs = self.policy(states[idx])
                dist = torch.distributions.Categorical(current_probs)
                current_log_probs = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()

                current_values = self.value(states[idx]).squeeze(-1)

                # Compute ratio for PPO
                ratio = torch.exp(current_log_probs - old_log_probs[idx])

                # Compute PPO losses
                surrogate1 = ratio * advantages[idx]
                surrogate2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * advantages[idx]
                )

                # Policy loss (negative because we're doing gradient ascent)
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                # Value loss
                value_loss = ((current_values - returns[idx]) ** 2).mean()

                # Entropy bonus (encourages exploration)
                entropy_loss = -ENTROPY_COEF * entropy

                # Total loss
                total_loss = policy_loss + VALUE_COEF * value_loss + entropy_loss

                # Update networks
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                self.policy_optimizer.step()
                self.value_optimizer.step()

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

            agent = cls(state_dim, action_dim)
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
