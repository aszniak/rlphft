import gymnasium as gym
import numpy as np
from agent import PPOAgent, RandomAgent
import matplotlib.pyplot as plt
import cv2

# Create a single figure with three subplots for our three visualizations
plt.ion()
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
fig.tight_layout(pad=5.0)
lines = [None, None, None]  # Will hold line objects for each subplot


def setup_plots():
    """Initialize the three plots with proper titles and labels"""
    global lines, fig, axs

    # Clear figure if it exists
    plt.figure(fig.number)
    for ax in axs:
        ax.clear()

    # Random agent performance
    axs[0].set_title("Random Agent Performance")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")
    (lines[0],) = axs[0].plot([], [], "r-")

    # Training progress
    axs[1].set_title("PPO Training Progress")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Average Reward")
    (lines[1],) = axs[1].plot([], [], "b-")

    # Trained agent performance
    axs[2].set_title("Trained Agent Performance")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Reward")
    (lines[2],) = axs[2].plot([], [], "g-")

    # Set initial axis limits
    for ax in axs:
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 500)  # CartPole max is 500

    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)


def evaluate_agent(
    agent, env, episodes=5, display=True, window_size=(1024, 768), plot_idx=None
):
    """Evaluate an agent and optionally update the corresponding plot"""
    total_rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, _ = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if display:
                frame = env.render()
                frame = cv2.resize(frame, window_size)
                cv2.imshow("CartPole", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            ep_reward += reward

        total_rewards.append(ep_reward)

        # Update the corresponding plot if specified
        if plot_idx is not None:
            lines[plot_idx].set_data(range(1, len(total_rewards) + 1), total_rewards)
            axs[plot_idx].relim()
            axs[plot_idx].autoscale_view()
            plt.draw()
            plt.pause(0.01)

    if display:
        cv2.destroyAllWindows()

    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {episodes} episodes: {avg_reward}")

    return total_rewards


def train_ppo(
    env,
    agent,
    num_epochs=100,
    steps_per_epoch=2048,
    display_env=False,
    window_size=(1024, 768),
):
    """Train the PPO agent and update the middle plot with training progress"""
    rewards_history = []
    eval_epochs = []  # Store epoch numbers for the x-axis (1-based)

    for epoch in range(num_epochs):
        # Collect trajectories
        states, actions, rewards, dones, next_states, log_probs, ep_lengths = (
            agent.collect_trajectories(
                env, steps_per_epoch, display=display_env, window_size=window_size
            )
        )

        # Update agent
        agent.update(states, actions, rewards, dones, next_states, log_probs)

        # Evaluate agent EVERY epoch - always do evaluation
        eval_rewards = evaluate_agent(agent, env, episodes=3, display=False)
        avg_reward = np.mean(eval_rewards)
        rewards_history.append(avg_reward)

        # Store 1-based epoch number for display
        current_epoch = epoch + 1
        eval_epochs.append(current_epoch)

        # COMPLETELY redraw the middle plot
        axs[1].clear()
        axs[1].set_title("PPO Training Progress")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Average Reward")

        # Plot using explicit markers and connecting lines
        axs[1].plot(eval_epochs, rewards_history, "bo-", markersize=6)

        # Better labels with correct epoch numbers (1-based)
        # Show fewer ticks if we have many epochs
        if len(eval_epochs) > 10:
            tick_indices = list(
                range(0, len(eval_epochs), max(1, len(eval_epochs) // 10))
            )
            axs[1].set_xticks([eval_epochs[i] for i in tick_indices])
            axs[1].set_xticklabels([str(eval_epochs[i]) for i in tick_indices])
        else:
            axs[1].set_xticks(eval_epochs)
            axs[1].set_xticklabels([str(e) for e in eval_epochs])

        # Fix y-axis limits more sensibly
        max_reward = max(rewards_history) if rewards_history else 50
        axs[1].set_ylim(0, min(500, max(200, max_reward * 1.2)))

        # Ensure x-axis shows enough space
        axs[1].set_xlim(0, max(eval_epochs) + 5 if eval_epochs else 20)

        # Grid for better readability
        axs[1].grid(True, linestyle="--", alpha=0.7)

        # Force redraw
        fig.canvas.draw()
        plt.pause(0.1)

        print(f"Epoch {current_epoch}/{num_epochs}, Average Reward: {avg_reward:.2f}")

    return rewards_history, eval_epochs  # Return both for final plot


def main():
    # Setup plots first
    setup_plots()

    # Create environment and agents
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    random_agent = RandomAgent(action_dim)
    ppo_agent = PPOAgent(state_dim, action_dim)

    # 1. Evaluate random agent
    print("Random Agent Performance:")
    random_rewards = evaluate_agent(random_agent, env, episodes=20, plot_idx=0)

    # 2. Train PPO agent
    print("\nTraining PPO Agent...")
    rewards_history, eval_epochs = train_ppo(
        env,
        ppo_agent,
        num_epochs=50,
        steps_per_epoch=2048,
        display_env=False,
        window_size=(800, 600),
    )

    # 3. Evaluate trained PPO agent
    print("\nTrained PPO Agent Performance:")
    ppo_rewards = evaluate_agent(ppo_agent, env, episodes=20, display=True, plot_idx=2)

    # Make sure the trained agent plot is visible WITH MARGIN
    axs[2].set_xlim(0, 21)
    # Add 10% margin above 500 to ensure we can see the line clearly
    axs[2].set_ylim(0, 550)  # Fixed upper limit with margin
    axs[2].grid(True, linestyle="--", alpha=0.7)
    fig.canvas.draw()

    # Keep the plots open until user closes window
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
