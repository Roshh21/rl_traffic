#  Multi-Agent RL Traffic Signal Control

This project applies **Multi-Agent Reinforcement Learning (MARL)** using **SUMO** to optimize traffic signal control.  
Each intersection acts as an independent RL agent that learns to minimize waiting times and improve overall traffic flow.  
The model uses the **PPO (Proximal Policy Optimization)** algorithm to train agents through interaction with the SUMO simulation.

---

##  Key Idea
- Each traffic light = an RL agent  
- The environment = SUMO traffic simulator  
- Reward = reduced waiting time and queue length  
- The system learns optimal timing policies through continuous interaction and feedback  

For **best results**, train for **at least 1000 episodes** to ensure stable learning and convergence.

---

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt 
   ```

2. **Run SUMO**
   ```bash
   sumo-gui -c maps/map.sumocfg
   ```

3. **Training** 

   Start training the RL agents:
   ```bash
   python scripts/train.py --config config.yaml
   ```
   Training will generate checkpoints, logs, and model files under the results/ directory.

4. **Evaluation**
    
    After training, evaluate the best model:
    ```bash
    python scripts/evaluate.py --config config.yaml --model results/models/best_model.pth --gui
    ```

    This will load the trained model and visualize the learned behavior in the SUMO GUI.
