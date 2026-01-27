import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from rl_env import CoherentGoniometerEnv


def main():
    # Create environment
    env = CoherentGoniometerEnv(
        beam_energy_E0=11600.0,
        coherent_edge_Ei=8600.0,
        orientation_index=0,
        pitch_step_deg=1e-3,
        yaw_step_deg=1e-3,
        dose_per_step=0.1,
        max_steps=500
    )

    # Sanity check
    check_env(env, warn=True)

    # PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./ppo_gonio_tensorboard/",
    )

    # Train
    model.learn(total_timesteps=200_000, tb_log_name="PPO_normObs_expReward_noTrueAngle")

    # Save
    model.save("ppo_goniometer")

    env.close()


if __name__ == "__main__":
    main()
