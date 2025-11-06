import os
import sys
import yaml
import argparse
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment.sumo_env import MultiAgentSumoEnv
from src.agents.ppo_agent import PPOAgent

def train(config_path: str):
    """Main training loop"""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = config['experiment']['device']
    if device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        device = 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    log_dir = config['experiment']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{log_dir}/models", exist_ok=True)
    os.makedirs(f"{log_dir}/tensorboard", exist_ok=True)
    
    writer = SummaryWriter(f"{log_dir}/tensorboard/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    print("Creating environment...")
    env = MultiAgentSumoEnv(config)
    observations, _ = env.reset(seed=config['experiment']['seed'])
    
    sample_obs = observations[list(observations.keys())[0]]
    obs_dim = len(sample_obs)
    action_dim = env.action_space.n
    
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Number of agents: {len(env.ts_ids)}")
    
    agent = PPOAgent(obs_dim, action_dim, config, device)
    
    total_episodes = config['training']['total_episodes']
    save_freq = config['training']['save_freq']
    log_freq = config['training']['log_freq']
    
    best_reward = -np.inf
    episode_rewards = []
    
    print(f"\nStarting training for {total_episodes} episodes...")
    print(f"Device: {device}")
    print("=" * 50)
    
    for episode in range(total_episodes):
        observations, _ = env.reset()
        episode_reward = {ts_id: 0 for ts_id in env.ts_ids}
        episode_length = 0
        done = False
        
        pbar = tqdm(total=config['training']['max_steps_per_episode'], 
                   desc=f"Episode {episode+1}/{total_episodes}",
                   leave=False)
        
        while not done:
            actions = {}
            for ts_id in env.ts_ids:
                action, log_prob, value = agent.select_action(observations[ts_id])
                actions[ts_id] = action
                
                if episode_length > 0:  
                    agent.storage[ts_id][-1]['log_prob'] = log_prob
                    agent.storage[ts_id][-1]['value'] = value
            
            next_observations, rewards, dones, truncateds, infos = env.step(actions)
            
            for ts_id in env.ts_ids:
                agent.store_transition(
                    ts_id,
                    observations[ts_id],
                    actions[ts_id],
                    rewards[ts_id],
                    0.0,  
                    0.0,  
                    dones[ts_id]
                )
                episode_reward[ts_id] += rewards[ts_id]
            
            observations = next_observations
            done = dones['__all__']
            episode_length += 1
            
            pbar.update(1)
            pbar.set_postfix({
                'avg_reward': f"{np.mean(list(episode_reward.values())):.2f}",
                'step': episode_length
            })
        
        pbar.close()
        
        update_info = agent.update()
        
        avg_episode_reward = np.mean(list(episode_reward.values()))
        episode_rewards.append(avg_episode_reward)
        
        if (episode + 1) % log_freq == 0:
            print(f"\nEpisode {episode+1}/{total_episodes}")
            print(f"  Average Reward: {avg_episode_reward:.2f}")
            print(f"  Episode Length: {episode_length}")
            if update_info:
                print(f"  Policy Loss: {update_info['policy_loss']:.4f}")
                print(f"  Value Loss: {update_info['value_loss']:.4f}")
                print(f"  Entropy: {update_info['entropy']:.4f}")
            
            writer.add_scalar('Train/AverageReward', avg_episode_reward, episode)
            writer.add_scalar('Train/EpisodeLength', episode_length, episode)
            if update_info:
                writer.add_scalar('Train/PolicyLoss', update_info['policy_loss'], episode)
                writer.add_scalar('Train/ValueLoss', update_info['value_loss'], episode)
                writer.add_scalar('Train/Entropy', update_info['entropy'], episode)
        
        if avg_episode_reward > best_reward:
            best_reward = avg_episode_reward
            agent.save(f"{log_dir}/models/best_model.pth")
            print(f"  ✓ New best model saved! Reward: {best_reward:.2f}")
        
        if (episode + 1) % save_freq == 0:
            agent.save(f"{log_dir}/models/checkpoint_ep{episode+1}.pth")
            print(f"  ✓ Checkpoint saved")
    
    agent.save(f"{log_dir}/models/final_model.pth")
    print("\n" + "=" * 50)
    print(f"Training completed!")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Models saved in: {log_dir}/models/")
    
    env.close()
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO agent for traffic signal control')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    train(args.config)

