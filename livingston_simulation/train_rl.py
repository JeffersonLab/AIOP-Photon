import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
0
from rl_env import CoherentGoniometerEnv

def make_env():

    env = CoherentGoniometerEnv(
        beam_energy_E0=11600.0,
        coherent_edge_Ei=8600.0,
        orientation_index=1,
        pitch_step_deg=1e-3,
        yaw_step_deg=1e-3,
        dose_per_step=0.1,
        max_steps=500,
    )
    return Monitor(env)

def main():
    check_env(make_env(), warn=True)

    train_env = DummyVecEnv([make_env])
    
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    # Eval environment
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )
    eval_env.training = False
    eval_env.norm_reward = False

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best/",
        log_path="./models/eval_logs/",
        eval_freq=10_000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )
    
    # PPO model
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./ppo_gonio_tensorboard/"
        #target_kl=0.01,
    )

    # Train
    model.learn(total_timesteps=200_000, callback=eval_callback)

    # Save
    model.save("./models/ppo_goniometer_final")
    train_env.save("./models/vecnormalize.pkl")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
