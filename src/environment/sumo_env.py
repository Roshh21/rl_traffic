import os
import sys
import numpy as np
import traci
from typing import Dict, Tuple, List, Optional
import gymnasium as gym
from gymnasium import spaces

class MultiAgentSumoEnv(gym.Env):
    """
    Multi-agent SUMO environment for traffic signal control
    Each traffic light is an independent agent
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.sumo_cfg = config['sumo']['cfg_file']
        self.use_gui = config['sumo']['use_gui']
        self.delta_time = config['sumo']['delta_time']
        self.yellow_time = config['sumo']['yellow_time']
        self.min_green = config['sumo']['min_green']
        self.max_green = config['sumo']['max_green']
        self.max_steps = config['training']['max_steps_per_episode']
        self.reward_type = config['environment']['reward_type']
        
        self.sumo_running = False
        self.ts_ids = [] 
        self.agents = {}
        self.current_step = 0
        
        self.observation_space = None
        self.action_space = None
        
        self.max_lanes = 0
        self.max_phases = 0
        
    def start_sumo(self):
        """Start SUMO simulation"""
        sumo_binary = 'sumo-gui' if self.use_gui else 'sumo'
        sumo_cmd = [
            sumo_binary,
            '-c', self.sumo_cfg,
            '--waiting-time-memory', '10000',
            '--time-to-teleport', '-1',
            '--no-step-log', 'true',
            '--no-warnings', 'true',
            '--duration-log.disable', 'true',
        ]
        
        traci.start(sumo_cmd)
        self.sumo_running = True
        
    def _get_traffic_lights(self):
        """Get all traffic light IDs from SUMO"""
        self.ts_ids = traci.trafficlight.getIDList()
        print(f"Found {len(self.ts_ids)} traffic lights")
        
        temp_agents = []
        for ts_id in self.ts_ids:
            temp_agent = TrafficSignalAgent(
                ts_id, 
                self.delta_time, 
                self.yellow_time,
                self.min_green,
                self.max_green
            )
            temp_agents.append(temp_agent)
            self.max_lanes = max(self.max_lanes, len(temp_agent.incoming_lanes))
            self.max_phases = max(self.max_phases, len(temp_agent.green_phases))
        
        print(f"Max lanes across all signals: {self.max_lanes}")
        print(f"Max green phases across all signals: {self.max_phases}")
        
        for temp_agent in temp_agents:
            temp_agent.max_lanes = self.max_lanes
            temp_agent.max_phases = self.max_phases
            self.agents[temp_agent.id] = temp_agent
            
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        """Reset environment"""
        if seed is not None:
            np.random.seed(seed)
            
        if self.sumo_running:
            traci.close()
            self.sumo_running = False
            
        self.start_sumo()
        self._get_traffic_lights()
        self.current_step = 0
        
        if self.observation_space is None:
            sample_obs = self._get_observations()
            obs_dim = len(sample_obs[self.ts_ids[0]])
            print(f"Observation dimension (fixed): {obs_dim}")
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(2)
        
        observations = self._get_observations()
        infos = {ts_id: {} for ts_id in self.ts_ids}
        
        return observations, infos
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Execute actions for all agents
        
        Args:
            actions: Dict mapping agent_id to action (0: keep, 1: switch)
            
        Returns:
            observations, rewards, dones, truncateds, infos
        """
        previous_metrics = self._get_metrics()
        
        for ts_id, action in actions.items():
            self.agents[ts_id].set_action(action)
        
        for _ in range(self.delta_time):
            traci.simulationStep()
        
        self.current_step += 1
        
        observations = self._get_observations()
        
        current_metrics = self._get_metrics()
        rewards = self._calculate_rewards(previous_metrics, current_metrics)
        
        done = (self.current_step >= self.max_steps) or (traci.simulation.getMinExpectedNumber() <= 0)
        dones = {ts_id: done for ts_id in self.ts_ids}
        dones['__all__'] = done
        
        truncateds = {ts_id: False for ts_id in self.ts_ids}
        truncateds['__all__'] = False
        
        infos = {
            ts_id: {
                'waiting_time': current_metrics[ts_id]['waiting_time'],
                'queue': current_metrics[ts_id]['queue'],
            }
            for ts_id in self.ts_ids
        }
        
        return observations, rewards, dones, truncateds, infos
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents"""
        observations = {}
        
        for ts_id in self.ts_ids:
            obs = self.agents[ts_id].get_observation()
            observations[ts_id] = np.array(obs, dtype=np.float32)
            
        return observations
    
    def _get_metrics(self) -> Dict:
        """Get current metrics for all traffic lights"""
        metrics = {}
        
        for ts_id in self.ts_ids:
            metrics[ts_id] = {
                'waiting_time': self.agents[ts_id].get_waiting_time(),
                'queue': self.agents[ts_id].get_queue_length(),
                'pressure': self.agents[ts_id].get_pressure(),
            }
            
        return metrics
    
    def _calculate_rewards(self, previous: Dict, current: Dict) -> Dict[str, float]:
        """Calculate rewards for all agents"""
        rewards = {}
        
        for ts_id in self.ts_ids:
            if self.reward_type == 'diff-waiting-time':
                reward = previous[ts_id]['waiting_time'] - current[ts_id]['waiting_time']
            elif self.reward_type == 'queue':
                reward = -current[ts_id]['queue']
            elif self.reward_type == 'pressure':
                reward = -current[ts_id]['pressure']
            else:
                reward = -current[ts_id]['waiting_time']
            
            rewards[ts_id] = reward
            
        return rewards
    
    def close(self):
        """Close SUMO"""
        if self.sumo_running:
            traci.close()
            self.sumo_running = False


class TrafficSignalAgent:
    """Individual traffic signal agent"""
    
    def __init__(self, ts_id: str, delta_time: int, yellow_time: int, 
                 min_green: int, max_green: int):
        self.id = ts_id
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = 0
        
        self.max_lanes = 0
        self.max_phases = 0
        
        self.lanes = traci.trafficlight.getControlledLanes(self.id)
        self.lanes = list(set(self.lanes))  
        
        self.incoming_lanes = []
        for lane in self.lanes:
            edge = traci.lane.getEdgeID(lane)
            if not edge.startswith(':'): 
                self.incoming_lanes.append(lane)
        
        logic = traci.trafficlight.getAllProgramLogics(self.id)[0]
        self.phases = logic.phases
        self.green_phases = []
        for i, phase in enumerate(self.phases):
            if 'y' not in phase.state.lower() and ('G' in phase.state or 'g' in phase.state):
                self.green_phases.append(i)
        
        if len(self.green_phases) == 0:
            self.green_phases = list(range(len(self.phases)))
        
    def set_action(self, action: int):
        """
        Set action for traffic light
        0: Keep current phase
        1: Switch to next phase
        """
        current_phase = traci.trafficlight.getPhase(self.id)
        
        if action == 1 and self.time_since_last_phase_change >= self.min_green:
            self._set_yellow_phase()
        else:
            self.time_since_last_phase_change += self.delta_time
            
    def _set_yellow_phase(self):
        """Set yellow phase before changing to next green phase"""
        current_phase = traci.trafficlight.getPhase(self.id)
        
        try:
            current_idx = self.green_phases.index(current_phase)
            next_idx = (current_idx + 1) % len(self.green_phases)
            next_green = self.green_phases[next_idx]
        except ValueError:
            next_green = self.green_phases[0]
        
        if current_phase + 1 < len(self.phases):
            traci.trafficlight.setPhase(self.id, current_phase + 1)
        
        self.next_action_time = self.yellow_time
        self.green_phase = next_green
        self.is_yellow = True
        self.time_since_last_phase_change = 0
        
    def get_observation(self) -> List[float]:
        """
        Get observation for this traffic light - FIXED SIZE
        
        Observation includes (padded to max values):
        - Queue length per lane (normalized) - max_lanes values
        - Average speed per lane (normalized) - max_lanes values
        - Waiting time per lane (normalized) - max_lanes values
        - Current phase (one-hot encoded) - max_phases values
        - Time since last phase change (normalized) - 1 value
        
        Total: max_lanes * 3 + max_phases + 1
        """
        obs = []
        
        max_queue = 20.0
        max_speed = 15.0
        max_waiting = 100.0
        
        for i in range(self.max_lanes):
            if i < len(self.incoming_lanes):
                lane = self.incoming_lanes[i]
                
                queue = traci.lane.getLastStepHaltingNumber(lane)
                obs.append(min(queue / max_queue, 1.0))
                
                speed = traci.lane.getLastStepMeanSpeed(lane)
                obs.append(speed / max_speed)
                
                waiting = traci.lane.getWaitingTime(lane)
                obs.append(min(waiting / max_waiting, 1.0))
            else:
                obs.extend([0.0, 0.0, 0.0])
        
        current_phase = traci.trafficlight.getPhase(self.id)
        for i in range(self.max_phases):
            if i < len(self.green_phases):
                obs.append(1.0 if current_phase == self.green_phases[i] else 0.0)
            else:
                obs.append(0.0)
        
        obs.append(min(self.time_since_last_phase_change / self.max_green, 1.0))
        
        return obs
    
    def get_queue_length(self) -> int:
        """Get total queue length"""
        return sum(traci.lane.getLastStepHaltingNumber(lane) 
                   for lane in self.incoming_lanes)
    
    def get_waiting_time(self) -> float:
        """Get total waiting time"""
        return sum(traci.lane.getWaitingTime(lane) 
                   for lane in self.incoming_lanes)
    
    def get_pressure(self) -> float:
        """
        Get pressure (difference between incoming and outgoing vehicles)
        """
        incoming = sum(traci.lane.getLastStepVehicleNumber(lane) 
                      for lane in self.incoming_lanes)
        return float(incoming)