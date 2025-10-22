import os
import sys
import math

import retro
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from custYuYuHakushoWrapper import CustYuYuHakushoWrapper

Total_Timesteps = 3000000
NUM_ENV = 16
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

'''
该函数定义了一个线性调度器，功能如下：

将初始值和最终值转换为浮点数，并确保初始值大于0。
返回一个内部函数scheduler，根据进度progress（0到1之间）线性插值计算当前值，计算公式为：current_value = final_value + progress * (initial_value - final_value)。
'''
# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

'''
该Python函数make_env用于创建一个复古游戏环境，并进行监控和种子设置：

1接受游戏名、状态和种子参数。
2通过retro.make创建环境，限制动作并设置图像观察类型。
3封装环境至Monitor以记录性能。
4设置随机种子以确保实验可重复。
5返回准备好的环境。
'''
def make_env(game, state, seed=0):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED, 
            obs_type=retro.Observations.IMAGE    
        )

        env = CustYuYuHakushoWrapper(env)
        env = Monitor(env, 'logs/')
        env.seed(seed)
        return env
    return _init

# DoubleDragonIITheRevenge-Nes
# BalloonFight-Nes
# StreetFighterIISpecialChampionEdition-Genesis

def main():
    game = "YuYuHakusho-Genesis"
    state = "Level1.state"
    env = SubprocVecEnv([make_env(game, state=state, seed=i) for i in range(NUM_ENV)])

    # Set linear schedule for learning rate
    # Start
    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)

    # fine-tune
    # lr_schedule = linear_schedule(5.0e-5, 2.5e-6)

    # Set linear scheduler for clip range
    # Start
    clip_range_schedule = linear_schedule(0.15, 0.025)

    # fine-tune
    # clip_range_schedule = linear_schedule(0.075, 0.025)

    model = PPO(
        "CnnPolicy", 
        env,
        device="cuda", 
        verbose=1,
        n_steps=512,
        batch_size=512,
        n_epochs=4,
        gamma=0.94,
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        tensorboard_log="logs"
    )

    # Set the save directory
    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)

    # Load the model from file
    # model_path = "trained_models/ppo_yuyuhakusho_final_base.zip"
    
    # Load model and modify the learning rate and entropy coefficient
    # custom_objects = {
    #     "learning_rate": lr_schedule,
    #     "clip_range": clip_range_schedule,
    #     "n_steps": 512
    # }
    # model = PPO.load(model_path, env=env, device="cuda", custom_objects=custom_objects)

    # Set up callbacks
    # Note that 1 timesetp = 6 frame
    checkpoint_interval = int(Total_Timesteps / NUM_ENV / 4) # checkpoint_interval * num_envs = total_steps_per_checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix="ppo_yuyuhakusho")

    # Writing the training logs from stdout to a file
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file
    
        model.learn(
            total_timesteps=int(Total_Timesteps), # total_timesteps = stage_interval * num_envs * num_stages (1120 rounds)
            callback=[checkpoint_callback], # stage_increase_callback]
            progress_bar= True          # 是否显示进度条
        )
        env.close()

    # Restore stdout
    sys.stdout = original_stdout

    # Save the final model
    model.save(os.path.join(save_dir, "ppo_yuyuhakusho_final.zip"))

if __name__ == "__main__":
    main()