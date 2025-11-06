import os
import sys
import yaml
import argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment.sumo_env import MultiAgentSumoEnv
from src.agents.ppo_agent import PPOAgent

def evaluate(config_path: str, model_path: str, num_episodes: int = 5, use_gui: bool = True):
    """Evaluate trained model"""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['sumo']['use_gui'] = use_gui
    
    device = config['experiment']['device']
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    print("Creating environment...")
    env = MultiAgentSumoEnv(config)
    observations, _ = env.reset()
    
    sample_obs = observations[list(observations.keys())[0]]
    obs_dim = len(sample_obs)
    action_dim = env.action_space.n
    
    agent = PPOAgent(obs_dim, action_dim, config, device)
    agent.load(model_path)
    print(f"Model loaded from: {model_path}")
    
    all_rewards = []
    all_waiting_times = []
    all_queues = []
    
    print(f"\nEvaluating for {num_episodes} episodes...")
    print("=" * 50)
    
    for episode in range(num_episodes):
        observations, _ = env.reset()
        episode_reward = {ts_id: 0 for ts_id in env.ts_ids}
        episode_waiting_time = []
        episode_queue = []
        done = False
        step = 0
        
        pbar = tqdm(total=config['training']['max_steps_per_episode'],
                   desc=f"Episode {episode+1}/{num_episodes}",
                   leave=True)
        
        while not done:
            actions = {}
            for ts_id in env.ts_ids:
                action, _, _ = agent.select_action(observations[ts_id], deterministic=True)
                actions[ts_id] = action
            
            next_observations, rewards, dones, truncateds, infos = env.step(actions)
            
            total_waiting = sum(infos[ts_id]['waiting_time'] for ts_id in env.ts_ids)
            total_queue = sum(infos[ts_id]['queue'] for ts_id in env.ts_ids)
            episode_waiting_time.append(total_waiting)
            episode_queue.append(total_queue)
            
            for ts_id in env.ts_ids:
                episode_reward[ts_id] += rewards[ts_id]
            
            observations = next_observations
            done = dones['__all__']
            step += 1
            
            pbar.update(1)
            pbar.set_postfix({
                'reward': f"{np.mean(list(episode_reward.values())):.2f}",
                'waiting': f"{total_waiting:.0f}",
                'queue': f"{total_queue:.0f}"
            })
        
        pbar.close()
        
        avg_reward = np.mean(list(episode_reward.values()))
        avg_waiting = np.mean(episode_waiting_time)
        avg_queue = np.mean(episode_queue)
        
        all_rewards.append(avg_reward)
        all_waiting_times.append(avg_waiting)
        all_queues.append(avg_queue)
        
        print(f"\nEpisode {episode+1} Results:")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average Waiting Time: {avg_waiting:.2f}")
        print(f"  Average Queue Length: {avg_queue:.2f}")
    
    print("\n" + "=" * 50)
    print("Evaluation Summary:")
    print(f"  Average Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"  Average Waiting Time: {np.mean(all_waiting_times):.2f} ± {np.std(all_waiting_times):.2f}")
    print(f"  Average Queue Length: {np.mean(all_queues):.2f} ± {np.std(all_queues):.2f}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained PPO agent')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of evaluation episodes')
    parser.add_argument('--gui', action='store_true',
                       help='Use SUMO GUI')
    args = parser.parse_args()
    
    evaluate(args.config, args.model, args.episodes, args.gui)
