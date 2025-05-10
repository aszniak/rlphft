import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from tqdm import tqdm


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

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
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

    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class PPOAgent:
    """
    A PPO agent that uses a policy network and a value network to select actions and compute returns.
    """

    def __init__(self, state_dim, action_dim, gamma=0.99, clip_epsilon=0.2, lr=3e-4):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_returns(self, rewards, dones, next_value):
        returns = []
        R = next_value
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        return returns

    def update(
        self,
        states,
        actions,
        rewards,
        dones,
        next_states,
        old_log_probs,
        batch_size=64,
        epochs=10,
    ):
        # Convert to tensors - first convert lists to numpy arrays for efficiency
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        old_log_probs = torch.FloatTensor(np.array(old_log_probs))

        # Compute returns and advantages
        with torch.no_grad():
            next_values = self.value(torch.FloatTensor(np.array(next_states))).squeeze(
                -1
            )

        returns = self.compute_returns(
            rewards, dones, next_values[-1] if len(next_values) > 0 else 0
        )
        returns = torch.FloatTensor(np.array(returns))

        # Compute values for current states
        values = self.value(states).squeeze(-1)

        # Calculate advantages
        advantages = returns - values.detach()

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
                entropy_loss = -0.01 * entropy

                # Total loss
                total_loss = policy_loss + 0.5 * value_loss + entropy_loss

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
        episode_lengths = []

        state, _ = env.reset()
        done = False
        ep_length = 0

        # Use tqdm for steps collection - but disable if already in an outer tqdm
        pbar = tqdm(range(num_steps), desc="Collecting steps", leave=False)
        for step in pbar:
            action, log_prob = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Render if display is enabled
            if display:
                frame = env.render()
                frame = cv2.resize(frame, window_size)
                cv2.imshow("CartPole Training", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Store trajectory
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob.item())
            next_states.append(next_state)

            # Update state and episode length
            state = next_state
            ep_length += 1

            # If episode ended, reset environment and store episode length
            if done:
                episode_lengths.append(ep_length)
                ep_length = 0
                state, _ = env.reset()
                done = False
                pbar.set_postfix({"episodes_completed": len(episode_lengths)})

        # Add the final episode length if it didn't end
        if ep_length > 0:
            episode_lengths.append(ep_length)

        return states, actions, rewards, dones, next_states, log_probs, episode_lengths

    # Add other methods as needed
