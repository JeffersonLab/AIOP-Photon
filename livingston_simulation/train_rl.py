import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
        self.spisode_stabilities = []

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
                
                stability = infos[idx].get("episode_stability", 0.0)
                self.spisode_stabilities.append(stability)

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

        if len(self.spisode_stabilities) > 0:
            avg_stability = sum(self.spisode_stabilities) / len(self.spisode_stabilities)
            self.logger.record("physics/avg_episode_stability", avg_stability)
            self.spisode_stabilities = []


class TrajectoryPlotCallback(BaseCallback):
    """
    Custom callback that plays one episode, records the peak trajectory, 
    and logs a plot to TensorBoard.
    """
    def __init__(self, eval_env, run_name, check_freq=5000, save_dir = "./plots", verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.check_freq = check_freq

        self.run_name = run_name
        self.save_dir = save_dir

        self.run_dir = os.path.join(self.save_dir, self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            print("DEBUG!!!!!!!!!!!!!!!!")


            # 1. Reset the plotting environment
            obs, _ = self.eval_env.reset()

            target_val = self.eval_env.coherent_edge_Ei

            done = False
            
            # 2. Storage for history
            peaks = []
            steps = []
            current_step = 0
            
            # 3. Run the Episode
            while not done:
                # Use deterministic=True for evaluation (no random exploration noise)
                action, _ = self.model.predict(obs, deterministic=True)
                
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                
                # Store data
                peaks.append(info['peak'])
                steps.append(current_step)
                current_step += 1
            
            # 4. Create Matplotlib Figure
            fig = plt.figure(figsize=(10, 6))
            plt.plot(steps, peaks, label="Actual Peak", color="blue")
            plt.axhline(y=target_val, label="Target", linestyle="--", color="red")
            # plt.plot(steps, targets, label="Target", linestyle="--", color="red")
            
            plt.xlabel("Step")
            plt.ylabel("Energy (MeV)")
            plt.title(f"Episode Trajectory (Training Step {self.n_calls})")
            plt.ylim(target_val - 50, target_val + 50)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            filename = f"step_{self.num_timesteps:07d}.png"
            save_path = os.path.join(self.run_dir, filename)
            plt.savefig(save_path)
            print(f"Saved trajectory plot to {save_path}")

            # 5. Log to TensorBoard
            # SB3 supports logging matplotlib figures directly
            self.logger.record("trajectory/peak_tracking", fig, exclude=("stdout", "log", "json", "csv"))
            
            # Close the figure to prevent memory leaks
            plt.close(fig)
            
        return True




def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def main(args):
    log_name = args.name

    # Create environment
    env = CoherentGoniometerEnv(
        beam_energy_E0=11600.0,
        target_edge_low=8400.0,     # MeV, for randomization
        target_edge_high=8800.0,    # MeV, for randomization
        orientation_index=0,
        pitch_step_deg=2e-4,
        yaw_step_deg=2e-4,
        dose_per_step=0.0,
        max_steps=500
    )
    # Evaluation environment
    eval_env = CoherentGoniometerEnv(
        beam_energy_E0=11600.0,
        target_edge_low=8400.0,
        target_edge_high=8800.0, 
        orientation_index=0,
        pitch_step_deg=2e-4,
        yaw_step_deg=2e-4,
        dose_per_step=0.0,
        max_steps=500
    )

    plot_env = CoherentGoniometerEnv(
        beam_energy_E0=11600.0,
        target_edge_low=8400.0,
        target_edge_high=8800.0, 
        orientation_index=0,
        pitch_step_deg=2e-4,
        yaw_step_deg=2e-4,
        dose_per_step=0.0,
        max_steps=500
    )

    # Sanity check
    check_env(env, warn=True)


    # PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=linear_schedule(1e-4),
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./ppo_gonio_tensorboard/",
    )

    # plot callback
    plot_callback = TrajectoryPlotCallback(
        eval_env=plot_env, 
        run_name=log_name,
        save_dir="./experiments_plots",
        check_freq=20000)
    
    # eval callback
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=f'./models/best_eval/{log_name}/',
        log_path='./models/eval_logs/', 
        eval_freq=10_000, 
        n_eval_episodes=20,
        deterministic=True, 
        render=False)

    callbacks = [eval_callback, PhysicsLoggingCallback(), plot_callback]
    # callbacks = [eval_callback, PhysicsLoggingCallback()]
    # Train
    model.learn(total_timesteps=300_000, tb_log_name=log_name, callback=callbacks)

    # Save
    model.save(f"./models/{log_name}")

    env.close()
    eval_env.close()
    plot_env.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for Coherent Goniometer Environment")
    parser.add_argument('-n', '--name', type=str, default='PPO_goni', help='log name for training session')

    args = parser.parse_args()
    
    main(args)
