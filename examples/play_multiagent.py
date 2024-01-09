import os
from datetime import datetime
from gym_battlesnake.gymbattlesnake import BattlesnakeEnv
from gym_battlesnake.custompolicy import CustomPolicy
from stable_baselines import PPO2

## CONFIGURATION
PPO2_LATEST_MODEL = "models/ppo2_battlesnake_latest"
NUM_AGENTS = 4

placeholder_env = BattlesnakeEnv(n_threads=4, n_envs=16)
models = [PPO2.load("models/ppo2_battlesnake20240109-165321.pkl") for _ in range(NUM_AGENTS)]
# Close environment to free allocated resources
placeholder_env.close()

# Load the trained model
model = models[0]
env = BattlesnakeEnv(n_threads=1, n_envs=1, opponents=[ m for m in models if m is not model])
obs = env.reset()
for _ in range(10000):
    action,_ = model.predict(obs)
    obs,_,_,_ = env.step(action)
    env.render()