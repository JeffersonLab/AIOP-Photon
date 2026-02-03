import argparse
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback

from rl_env import CoherentGoniometerEnv


class PhysicsLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_errors = []
        self.episode_peak_positions = []

    def _on_step(self) -> bool:
        # Check if any environment in the batch is done
        dones = self.locals['dones']
        infos = self.locals['infos']
        
        for idx, done in enumerate(dones):
            if done:
                # Extract the final error from the info dict
                # Note: 'info' comes from your environment's step() return
                final_error = infos[idx].get("error", 0.0)
                self.episode_errors.append(abs(final_error))

                final_peak = infos[idx].get("peak", 0.0)
                self.episode_peak_positions.append(final_peak)
                
        return True

    def _on_rollout_end(self) -> None:
        # Log the average final error for this rollout period
        if len(self.episode_errors) > 0:
            avg_error = sum(self.episode_errors) / len(self.episode_errors)
            self.logger.record("physics/final_abs_error_MeV", avg_error)
            self.episode_errors = [] # Reset for next rollout
        
        # Log the final peak positions
        if len(self.episode_peak_positions) > 0:
            avg_peak = sum(self.episode_peak_positions) / len(self.episode_peak_positions)
            self.logger.record("physics/avg_final_peak_MeV", avg_peak)
            self.episode_peak_positions = []

def main(args):
    log_name = args.name

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
    # Evaluation environment
    eval_env = CoherentGoniometerEnv(
        beam_energy_E0=11600.0,
        coherent_edge_Ei=8600.0,
        base_peak_position=8580.0,
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

    # eval callback
    eval_callback = EvalCallback(eval_env, best_model_save_path='./models/best_eval/',
                                 log_path='./models/eval_logs/', eval_freq=10_000, n_eval_episodes=20,
                                 deterministic=True, render=False)

    callbacks = [eval_callback, PhysicsLoggingCallback()]
    # Train
    model.learn(total_timesteps=200_000, tb_log_name=log_name, callback=callbacks)

    # Save
    model.save("./models/ppo_goniometer_model")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for Coherent Goniometer Environment")
    parser.add_argument('-n', '--name', type=str, default='PPO_goni', help='log name for training session')

    args = parser.parse_args()
    
    main(args)
