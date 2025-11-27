# rollout_eval_no_norm.py
import upkie.envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import os

upkie.envs.register()

ENV_ID = "Upkie-Spine-Pendulum"
ENV_KWARGS = dict(frequency=200.0)
SEED = 0
MODEL_DIR = "./models/checkpoints"      # the dir to save checkpoint models


def main():
    env = make_vec_env(ENV_ID, n_envs=4, env_kwargs=ENV_KWARGS, seed=SEED)

    # define model from scratch or resume training with pre-trained model
    load_pre_trained = True if os.listdir(MODEL_DIR) else False
    if load_pre_trained:
        print("loading checkpoint model")
        model = PPO.load("./models/checkpoints/ppo_upkie_600000_steps", env=env)
    else:
        print("defining new model")
        model = PPO(
        "MlpPolicy",   # fully connected neural network
        env,
        learning_rate=3e-4,
        n_steps=2048,      # number of steps to collect before each update
        batch_size=64,
        gamma=0.99,        # discount factor
        verbose=1,
        device="cuda",
        )

    # Save every 50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,          # save every 50k steps
        save_path="./models/checkpoints",  # folder where checkpoints are stored
        name_prefix="ppo_upkie"    # prefix for the file names
    )



    TIMESTEPS = 700_000  # arbitrary selection of timesteps to learn
    print("preparing to train")
    model.learn(
        total_timesteps=TIMESTEPS,
        callback=checkpoint_callback)
    print("model trained")

    # Save the trained model
    model.save("./models/ppo_upkie_final.zip")
    print("model saved")

if __name__ == "__main__":
    main()
