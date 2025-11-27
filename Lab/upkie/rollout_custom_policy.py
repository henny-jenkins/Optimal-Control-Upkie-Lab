# rollout_eval_no_norm.py
import upkie.envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
import numpy as np

upkie.envs.register()

MODEL_PATH = "./models/pendulum_best/best_model.zip"  # or "./models/ppo_upkie_final.zip"
ENV_ID = "Upkie-Spine-Pendulum"
ENV_KWARGS = dict(frequency=200.0)
SEED = 0
sim_length = 1500

def main():
    env = make_vec_env(ENV_ID, n_envs=1, env_kwargs=ENV_KWARGS, seed=SEED)
    model = PPO.load("./models/ppo_upkie_final", env=env)

    obs = env.reset()
    ep_return = 0.0
    ep_len = 0
    action_history = np.zeros(sim_length, dtype=float)
    state_history = np.zeros((4, sim_length), dtype=float)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        ep_return += float(reward[0])

        # log the state and action variables for plotting
        state_history[0, ep_len] = obs[0][0]
        state_history[1, ep_len] = obs[0][1]
        state_history[2, ep_len] = obs[0][2]
        state_history[3, ep_len] = obs[0][3]
        action_history[ep_len] = action[0][0]
        ep_len += 1
        if bool(dones[0]):
            return state_history, action_history
    print(f"[EVAL] return={ep_return:.3f}, length={ep_len} steps")

if __name__ == "__main__":
    state_history, action_history = main()
    T = np.arange(len(state_history[0]))

    # Create subplots (4 states + 1 action)
    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    fig.suptitle("State Variables and Action Over PPO Rollout")

    # theta
    axs[0].plot(T, state_history[0], label=r"$\theta$")
    axs[0].set_ylabel(r"$\theta$ [rad]")
    axs[0].grid(True)

    # theta dot
    axs[1].plot(T, state_history[1], label=r"$\dot{\theta}$")
    axs[1].set_ylabel(r"$\dot{\theta}$ [rad/s]")
    axs[1].grid(True)

    # p
    axs[2].plot(T, state_history[2], label=r"$p$")
    axs[2].set_ylabel(r"$p$ [m]")
    axs[2].grid(True)

    # p dot
    axs[3].plot(T, state_history[3], label=r"$\dot{p}$")
    axs[3].set_ylabel(r"$\dot{p}$ [m/s]")
    axs[3].grid(True)

    # Action
    axs[4].plot(T, action_history, color="purple", label="action")
    axs[4].set_ylabel("Action [m/s]")
    axs[4].set_xlabel("Timestep")
    axs[4].grid(True)

    plt.tight_layout()
    plt.savefig("upkie_rollout.png", dpi=300)
    plt.show()