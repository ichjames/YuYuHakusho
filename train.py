import os
import sys
import math

import retro
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from custYuYuHakushoWrapper import CustYuYuHakushoWrapper


'''
日志参数说明：

1. rollout/ —— 环境交互阶段（Rollout Phase）
这是指智能体在环境中运行、收集经验（状态、动作、奖励等）的过程。
ep_len_mean	每个 episode（回合）平均持续了 566 步。这反映任务的平均完成时间或生存时间。数值稳定说明探索较一致。
ep_rew_mean	每个 episode 的平均累积奖励为 -971。负值说明：要么任务本身惩罚较多（如控制任务中偏离目标受罚），要么模型尚未学会有效策略。需要结合具体任务判断是否正常。

2. time/ —— 时间与训练进度
fps	  Frames Per Second，每秒处理的环境步数（timesteps），反映训练效率。越高越好（说明采样+训练速度快）。354 属于中等偏上水平。
iterations	当前已完成 17 轮 PPO 更新迭代。每轮通常包括：采样一批数据 → 多次优化更新策略网络。
time_elapsed	已过去 393 秒 ≈ 6.5 分钟。从训练开始到现在的总耗时（秒）。
total_timesteps	 到目前为止，智能体总共与环境交互了 139,264 步。这是衡量训练量的核心指标。


3. train/ —— 训练阶段（Policy Update）
这部分来自神经网络的训练过程，通常基于 rollout 收集的数据进行多轮梯度更新。
approx_kl	Approximate KL divergence（近似KL散度），衡量新旧策略之间的差异。PPO 用它来判断是否“走得太远”。<br>✅ 一般 < 0.01 是好的；> 0.01~0.02 可能提示策略变化太大，可考虑触发 early stopping。<br>👉 当前值 0.0085 是健康的。
clip_fraction	表示在 PPO 的 clip 机制中，有 26.4% 的梯度更新被裁剪了。<br>理想值在 0.1~0.5 之间。太低说明 clip_range 太大，太高说明太小。<br>👉 0.264 是合理范围，说明 clip 设置得当。
clip_range	PPO 的 clipping 范围（ε），即策略更新的“信任区域”半径。<br>常见初始值为 0.1~0.2。如果是固定值则不变；若自适应调整，这里会变化。
entropy_loss	策略熵的负值（实际是 entropy 的损失项）。<br>熵越高，策略越随机（探索性强）；太低会导致过早收敛。<br>👉 -7.86 的绝对值较大，说明策略仍有较强探索性（可能是好现象，尤其在早期）。注意：有些框架记录为 -entropy，所以是负的。
explained_variance	评估值函数（Value Network）对回报（return）的拟合程度。<br>公式：1 - Var(y - ŷ)/Var(y)<br>👉 0.271 偏低，说明 value network 还不能很好预测未来回报（有待提升）。<br>🎯 理想接近 1.0；低于 0 或接近 0 表示拟合差。
learning_rate	当前学习率。PPO 常用 Adam 优化器，初始 lr 通常为 3e-4，这里略低，可能是学习率衰减后的值。
loss	总损失（可能是 value_loss + policy_loss + entropy_loss 的加权和），用于反向传播。<br>这个值本身意义不大，关键是看它是否稳定下降。125 偏高，但要结合任务复杂度看趋势。
n_updates	当前已完成 64 次参数更新（可能是每次 rollout 后做多次 minibatch 更新）。
policy_gradient_loss	PPO 的策略梯度损失（通常是 clipped surrogate loss）。<br>值较小是正常的，关键是看其下降趋势。当前 0.009 是合理范围。
value_loss	值函数（Critic）的损失，通常是均方误差（MSE）：(estimated_value - actual_return)²。<br>182 偏高，结合 explained_variance=0.271，说明 critic 拟合效果较差，可能需要：<br>- 更大网络容量<br>- 更多训练 epochs<br>- 更好的 GAE 参数（如 γ, λ）
'''

Total_Timesteps = 3000000
NUM_ENV = 16
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

'''d
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
    state = "Level2.state"
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
    checkpoint_interval = 100000 / NUM_ENV # checkpoint_interval * num_envs = total_steps_per_checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix="ppo_yuyuhakusho")

    # Writing the training logs from stdout to a file
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file
    
        model.learn(
            total_timesteps=int(Total_Timesteps), # total_timesteps = stage_interval * num_envs * num_stages (1120 rounds)
            callback=[checkpoint_callback], # stage_increase_callback]
        )
        env.close()

    # Restore stdout
    sys.stdout = original_stdout

    # Save the final model
    model.save(os.path.join(save_dir, "ppo_yuyuhakusho_final.zip"))

if __name__ == "__main__":
    main()