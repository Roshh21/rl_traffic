import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

class PPOAgent:
    """
    Proximal Policy Optimization agent for traffic signal control
    """
    
    def __init__(self, obs_dim: int, action_dim: int, config: dict, device: str):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device(device)
        
        self.lr = config['training']['learning_rate']
        self.gamma = config['training']['gamma']
        self.gae_lambda = config['training']['gae_lambda']
        self.clip_epsilon = config['training']['clip_epsilon']
        self.entropy_coef = config['training']['entropy_coef']
        self.value_loss_coef = config['training']['value_loss_coef']
        self.max_grad_norm = config['training']['max_grad_norm']
        self.ppo_epochs = config['training']['ppo_epochs']
        self.batch_size = config['training']['batch_size']
        
        self.policy = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        self.reset_storage()
        
    def reset_storage(self):
        """Reset storage for trajectory data"""
        self.storage = defaultdict(list)
        
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Select action using current policy
        
        Returns:
            action, log_prob, value
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, value = self.policy(obs_tensor)
            
        if deterministic:
            action = torch.argmax(action_probs, dim=1).item()
            log_prob = torch.log(action_probs[0, action]).item()
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor([action], device=self.device)).item()
        
        return action, log_prob, value.item()
    
    def store_transition(self, agent_id: str, obs: np.ndarray, action: int, 
                        reward: float, log_prob: float, value: float, done: bool):
        """Store transition for later training"""
        self.storage[agent_id].append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'log_prob': log_prob,
            'value': value,
            'done': done
        })
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                    dones: List[bool], next_value: float) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation
        
        Returns:
            advantages, returns
        """
        advantages = []
        returns = []
        gae = 0
        
        values = values + [next_value]
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def update(self):
        """Update policy using PPO"""
        if not self.storage:
            return {}
        
        all_obs = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []
        all_returns = []
        
        for agent_id, transitions in self.storage.items():
            if len(transitions) == 0:
                continue
            
            obs = [t['obs'] for t in transitions]
            actions = [t['action'] for t in transitions]
            rewards = [t['reward'] for t in transitions]
            old_log_probs = [t['log_prob'] for t in transitions]
            values = [t['value'] for t in transitions]
            dones = [t['done'] for t in transitions]
            
            if dones[-1]:
                next_value = 0.0
            else:
                next_obs = obs[-1]
                obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    _, next_value = self.policy(obs_tensor)
                    next_value = next_value.item()
            
            advantages, returns = self.compute_gae(rewards, values, dones, next_value)
            
            all_obs.extend(obs)
            all_actions.extend(actions)
            all_old_log_probs.extend(old_log_probs)
            all_advantages.extend(advantages)
            all_returns.extend(returns)
        
        if len(all_obs) == 0:
            return {}
        
        obs_tensor = torch.FloatTensor(np.array(all_obs)).to(self.device)
        actions_tensor = torch.LongTensor(all_actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(all_old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(all_advantages).to(self.device)
        returns_tensor = torch.FloatTensor(all_returns).to(self.device)
        
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        dataset_size = len(all_obs)
        indices = np.arange(dataset_size)
        
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                batch_obs = obs_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                action_probs, values = self.policy(batch_obs)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())
        
        self.reset_storage()
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropy_losses)
        }
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, obs):
        shared_features = self.shared(obs)
        action_probs = self.policy_head(shared_features)
        value = self.value_head(shared_features)
        return action_probs, value