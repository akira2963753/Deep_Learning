import torch
import numpy as np
import random
import gymnasium as gym
import ale_py
import imageio
import os
import re
import argparse
from dqn_task3 import DQN, AtariPreprocessor

gym.register_envs(ale_py)


def parse_env_steps(path):
    """Extract the training step count from filenames like LAB5_XXX_task3_600000.pt."""
    m = re.search(r"(\d+)(?=\.pt$)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    preprocessor = AtariPreprocessor()
    input_shape = (4, 84, 84)
    num_actions = env.action_space.n

    model = DQN(input_shape, num_actions).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    env_steps = parse_env_steps(args.model_path)
    rewards = []

    for ep in range(args.episodes):
        ep_seed = args.seed + ep
        obs, _ = env.reset(seed=ep_seed)
        state = preprocessor.reset(obs)
        done = False
        total_reward = 0
        frames = []

        while not done:
            if args.record:
                frames.append(env.render())
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = preprocessor.step(next_obs)

        rewards.append(total_reward)

        if args.record:
            out_path = os.path.join(args.output_dir, f"eval_ep{ep}_seed{ep_seed}.mp4")
            with imageio.get_writer(out_path, fps=30, macro_block_size=1) as video:
                for f in frames:
                    video.append_data(f)

        print(f"Environment steps: {env_steps}, seed: {ep_seed}, eval reward: {int(total_reward)}")

    print(f"\nAverage reward: {np.mean(rewards):.2f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Use underscore arg name to match spec example: --model_path
    parser.add_argument("--model_path", "--model-path", dest="model_path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./eval_videos_task3")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record", action="store_true", help="Save mp4 videos of each evaluation episode")
    args = parser.parse_args()
    evaluate(args)
