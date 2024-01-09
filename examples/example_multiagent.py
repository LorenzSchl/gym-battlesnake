import os
from datetime import datetime
from os import terminal_size

from gym_battlesnake.gymbattlesnake import BattlesnakeEnv
from gym_battlesnake.custompolicy import CustomPolicy
from stable_baselines import PPO2

## CONFIGURATION
PPO2_LATEST_MODEL = "./models/ppo2_battlesnake_latest.pkl"
TENSORBOARD_LOGDIR = "./tensorboard/"
EPOCHS = 15
NUM_TIMESTEPS = 100_000
NUM_AGENTS = 4

# placeholder_env necessary for model to recognize,
# the observation and action space, and the vectorized environment
placeholder_env = BattlesnakeEnv(n_threads=4, n_envs=16)
models = [PPO2(CustomPolicy, placeholder_env, verbose=0, learning_rate=1e-3, tensorboard_log=TENSORBOARD_LOGDIR) for _ in range(NUM_AGENTS)]
# Close environment to free allocated resources
placeholder_env.close()

# Save the trained model to "/models/ppo2_battlesnake_<DateTime>"
modelpath = f"./models/ppo2_battlesnake_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    for model in models:
        env = BattlesnakeEnv(n_threads=4, n_envs=16, opponents=[ m for m in models if m is not model])
        model.set_env(env)
        model.learn(total_timesteps=NUM_TIMESTEPS)
        env.close()
    for i, model in enumerate(models):
        model.save(f"{modelpath}_{epoch}_{i}")

model = models[0]

# Update Symlink "/model/ppo2_battlesnake_latest" -> modelpath
if os.path.exists(PPO2_LATEST_MODEL):
    os.remove(PPO2_LATEST_MODEL)
os.symlink(f"{modelpath}_{EPOCHS-1}_0.pkl", PPO2_LATEST_MODEL)

# Load the trained model
model = PPO2.load(PPO2_LATEST_MODEL)
env = BattlesnakeEnv(n_threads=1, n_envs=1, opponents=[ m for m in models if m is not model])
obs = env.reset()
for _ in range(10000):
    action,_ = model.predict(obs)
    obs,_,_,_ = env.step(action)
    env.render()