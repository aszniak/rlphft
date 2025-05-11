import gymnasium as gym
import numpy as np
from agent import PPOAgent, RandomAgent
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import argparse

# Hyperparameters and constants
DEFAULT_EPOCHS = 50  # Default number of training epochs
STEPS_PER_EPOCH = 2000  # Number of environment steps per training epoch (4 x max episode length of 500)
EVAL_EPISODES = 10  # Default number of episodes for evaluation
TRAIN_EVAL_EPISODES = 3  # Number of episodes for evaluation during training
PLOT_FIGSIZE = (10, 15)  # Figure size for the plots (width, height)
TRAIN_WINDOW_SIZE = (800, 600)  # Window size for training env display
EVAL_WINDOW_SIZE = (800, 600)  # Window size for evaluation env display

# Create a single figure with three subplots for our three visualizations
plt.ion()
fig, axs = plt.subplots(3, 1, figsize=PLOT_FIGSIZE)
fig.tight_layout(pad=5.0)


def setup_plots():
    """Initialize the three plots with proper titles and labels"""
    global fig, axs

    # Clear figure if it exists
    plt.figure(fig.number)
    for ax in axs:
        ax.clear()

    # Random agent performance
    axs[0].set_title("Random Agent Performance")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")
    axs[0].set_ylim(0, 75)  # Random agent typically has low scores
    axs[0].set_xlim(0, EVAL_EPISODES + 1)  # Set based on actual episodes

    # Training progress
    axs[1].set_title("PPO Training Progress")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Average Reward")
    axs[1].set_ylim(0, 550)  # Full range for training progress
    axs[1].set_xlim(0, DEFAULT_EPOCHS / 2)  # Show half the epochs initially

    # Trained agent performance
    axs[2].set_title("Trained Agent Performance")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Reward")
    axs[2].set_ylim(450, 510)  # Trained agent typically has high scores near max
    axs[2].set_xlim(0, EVAL_EPISODES + 1)  # Set based on actual episodes

    # Add grid to all plots
    for ax in axs:
        ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)


def evaluate_agent(
    agent,
    env,
    episodes=EVAL_EPISODES,
    display=True,
    window_size=EVAL_WINDOW_SIZE,
    plot_idx=None,
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
                if cv2.waitKey(6) & 0xFF == ord("q"):
                    break

            ep_reward += reward

        total_rewards.append(ep_reward)

        # Update the corresponding plot if specified
        if plot_idx is not None:
            # COMPLETELY redraw the plot for consistent styling across all plots
            axs[plot_idx].clear()

            # Set title based on plot index
            if plot_idx == 0:
                axs[plot_idx].set_title("Random Agent Performance")
            elif plot_idx == 2:
                axs[plot_idx].set_title("Trained Agent Performance")

            axs[plot_idx].set_xlabel("Episode")
            axs[plot_idx].set_ylabel("Reward")

            # Plot with markers and connecting lines (use color based on plot index)
            color = "r-" if plot_idx == 0 else "g-"
            axs[plot_idx].plot(
                range(1, len(total_rewards) + 1), total_rewards, color, markersize=6
            )

            # Auto-scale axes with appropriate limits for each plot type
            max_reward = max(total_rewards) if total_rewards else 50
            min_reward = min(total_rewards) if total_rewards else 0

            if plot_idx == 0:  # Random agent (typically low scores)
                # For random agent, scale based on actual performance with some headroom
                upper_limit = min(200, max(75, max_reward * 1.5))
                axs[plot_idx].set_ylim(0, upper_limit)
            elif plot_idx == 2:  # Trained agent (typically high scores)
                # For trained agent with perfect scores (near 500)
                if max_reward > 480:
                    # If all episodes reached max reward, zoom in to see small variations
                    if min_reward > 480:
                        axs[plot_idx].set_ylim(480, 510)
                    else:
                        # If some episodes didn't reach max, show from lowest to max+margin
                        axs[plot_idx].set_ylim(max(0, min_reward - 20), 510)
                else:
                    # Otherwise use standard scaling with appropriate headroom
                    upper_limit = min(500, max(200, max_reward * 1.2))
                    axs[plot_idx].set_ylim(0, upper_limit)

            # Ensure x-axis shows enough space for all episodes plus a small margin
            axs[plot_idx].set_xlim(0, episodes + 1)

            # Add grid for better readability
            axs[plot_idx].grid(True, linestyle="--", alpha=0.7)

            # Force redraw
            fig.canvas.draw()
            plt.pause(0.01)

    if display:
        cv2.destroyAllWindows()

    avg_reward = np.mean(total_rewards)
    tqdm.write(f"Average reward over {episodes} episodes: {avg_reward:.2f}")

    return total_rewards


def train_ppo(
    env,
    agent,
    num_epochs=DEFAULT_EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    display_env=False,
    window_size=TRAIN_WINDOW_SIZE,
):
    """Train the PPO agent and update the middle plot with training progress"""
    rewards_history = []
    eval_epochs = []  # Store epoch numbers for the x-axis (1-based)

    # Track statistics on episodes
    total_episodes = 0  # Count of complete episodes
    all_episode_rewards = []  # List of all episode rewards
    total_steps = 0  # Total steps taken across all epochs
    incomplete_episodes = 0  # Count of episodes cut off at epoch boundaries

    # How many episodes reached max reward (500)
    perfect_episodes = 0
    max_steps_in_episode = 500  # Maximum steps in a CartPole episode

    for epoch in tqdm(range(num_epochs), desc="Training PPO"):
        # Collect trajectories
        states, actions, rewards, dones, next_states, log_probs, ep_lengths = (
            agent.collect_trajectories(
                env, steps_per_epoch, display=display_env, window_size=window_size
            )
        )

        # Update agent
        agent.update(states, actions, rewards, dones, next_states, log_probs)

        # Store all episode rewards
        all_episode_rewards.extend(agent.last_episode_rewards)

        # Update counters and statistics
        epoch_episode_rewards = sum(agent.last_episode_rewards)
        epoch_episodes = len(agent.last_episode_rewards)
        total_episodes += epoch_episodes
        total_steps += steps_per_epoch

        # Track if the last episode was cut off (didn't terminate naturally)
        last_episode_incomplete = False
        if (
            hasattr(agent, "last_episode_lengths")
            and len(agent.last_episode_lengths) > 0
        ):
            # If the last episode has exactly the remainder steps and didn't terminate
            if ep_lengths[-1] > 0:  # Positive length means it was incomplete
                incomplete_episodes += 1
                last_episode_incomplete = True

        # Count perfect episodes (reward = 500)
        perfect_in_epoch = 0
        for reward in agent.last_episode_rewards:
            if (
                reward >= max_steps_in_episode
            ):  # CartPole gives 1 reward per step, max is 500
                perfect_episodes += 1
                perfect_in_epoch += 1

        # Calculate weighted average
        epoch_avg_reward = (
            epoch_episode_rewards / epoch_episodes if epoch_episodes > 0 else 0
        )

        # Report the epoch's episode rewards with concise formatting
        incomplete_str = " (last episode incomplete)" if last_episode_incomplete else ""
        perfect_str = (
            f", perfect episodes: {perfect_in_epoch}/{epoch_episodes}"
            if epoch_episodes > 0
            else ""
        )
        tqdm.write(
            f"Epoch {epoch+1}: {epoch_episodes} episodes{incomplete_str}, avg reward: {epoch_avg_reward:.1f}{perfect_str}"
        )

        # Evaluate agent EVERY epoch - always do evaluation
        eval_rewards = evaluate_agent(
            agent, env, episodes=TRAIN_EVAL_EPISODES, display=False
        )
        avg_reward = np.mean(eval_rewards)
        rewards_history.append(avg_reward)

        # Store 1-based epoch number for display
        current_epoch = epoch + 1
        eval_epochs.append(current_epoch)

        # Update the progress bar description
        tqdm.write(
            f"Epoch {current_epoch}/{num_epochs}, Average Reward: {avg_reward:.2f}"
        )

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

        # Fix y-axis limits more sensibly with margin
        max_reward = max(rewards_history) if rewards_history else 50
        if max_reward > 450:  # If we're approaching the max of 500
            axs[1].set_ylim(0, 550)  # Add fixed margin
        else:
            axs[1].set_ylim(0, min(500, max(200, max_reward * 1.2)))

        # Ensure x-axis shows enough space
        axs[1].set_xlim(0, max(eval_epochs) + 5 if eval_epochs else 20)

        # Grid for better readability
        axs[1].grid(True, linestyle="--", alpha=0.7)

        # Force redraw
        fig.canvas.draw()
        plt.pause(0.1)

    # After training is complete - print concise training summary
    overall_avg_reward = (
        sum(all_episode_rewards) / total_episodes if total_episodes > 0 else 0
    )
    steps_per_episode = total_steps / total_episodes if total_episodes > 0 else 0
    perfect_episode_percentage = (
        (perfect_episodes / total_episodes) * 100 if total_episodes > 0 else 0
    )

    # Print training statistics (concise version)
    tqdm.write(
        f"Training complete: {total_episodes} episodes, {perfect_episodes} perfect ({perfect_episode_percentage:.1f}%), {incomplete_episodes} incomplete"
    )

    # Print info to match the training summary in the agent
    agent.print_training_summary(
        rewards_history,
        overall_stats={
            "total_episodes": total_episodes,
            "avg_reward": overall_avg_reward,
            "perfect_episodes": perfect_episodes,
            "perfect_percentage": perfect_episode_percentage,
            "incomplete_episodes": incomplete_episodes,
        },
    )

    return rewards_history, eval_epochs, overall_avg_reward


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CartPole-v1 with PPO")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a new model even if a saved one exists",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="saved_model",
        help="Path to save/load the model",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs"
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=EVAL_EPISODES,
        help="Number of evaluation episodes",
    )
    args = parser.parse_args()

    # Setup plots first
    setup_plots()

    # Create environment and agents
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    random_agent = RandomAgent(action_dim)

    # Try to load a pre-trained agent, or create a new one if not found or forced to train
    loaded_agent = None
    if not args.train:
        try:
            loaded_agent = PPOAgent.load_model(args.model_path)
        except Exception as e:
            tqdm.write(f"Failed to load model: {str(e)}")
            loaded_agent = None

    if loaded_agent is not None:
        ppo_agent = loaded_agent
        tqdm.write("Using pre-trained agent. Skipping training phase.")
        training_needed = False
    else:
        if args.train:
            tqdm.write("Training new model as requested by --train flag.")
        else:
            tqdm.write("No valid saved model found. Training a new model.")
        ppo_agent = PPOAgent(state_dim, action_dim)
        training_needed = True

    # 1. Evaluate random agent
    tqdm.write("Random Agent Performance:")
    random_rewards = evaluate_agent(
        random_agent, env, episodes=args.eval_episodes, plot_idx=0
    )

    # 2. Train PPO agent if needed
    if training_needed:
        tqdm.write("\nTraining PPO Agent...")
        rewards_history, eval_epochs, overall_avg_reward = train_ppo(
            env,
            ppo_agent,
            num_epochs=args.epochs,
            steps_per_epoch=STEPS_PER_EPOCH,
            display_env=False,
            window_size=TRAIN_WINDOW_SIZE,
        )

        # Print simple average reward summary
        tqdm.write(f"Average episode reward: {overall_avg_reward:.2f}")

        # Save the trained model
        try:
            ppo_agent.save_model(args.model_path)
        except Exception as e:
            tqdm.write(f"Error saving model: {str(e)}")

    # 3. Evaluate trained PPO agent
    tqdm.write("\nTrained PPO Agent Performance:")
    ppo_rewards = evaluate_agent(
        ppo_agent,
        env,
        episodes=args.eval_episodes,
        display=True,
        plot_idx=2,
        window_size=EVAL_WINDOW_SIZE,
    )

    # Keep the plots open until user closes window
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
