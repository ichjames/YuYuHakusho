# Custom environment wrapper
from datetime import datetime
import gym
import time
import math
import numpy as np
import collections

def normalize(min_num, max_num, num):
    return (num - min_num) / (max_num - min_num) * 100

COST_TOTALSCORELIANXUCOUNT = 10

class CustYuYuHakushoWrapper(gym.Wrapper):

    def __init__(self, env, reset_round=True, rendering=False,limit_step=True):
        super(CustYuYuHakushoWrapper, self).__init__(env)
        self.env = env

        # Use a deque to store the last 9 frames
        self.num_frames = 9
        self.frame_stack = collections.deque(maxlen=self.num_frames)
        self.num_step_frames = 6
        self.reward_coeff = 3.0
        # 注意,这里的shape为 112,120,3 
        # 怎么得出此值的？ 因为这个游戏的画面输出是 224*320，我们的目的是隔行取值（用于减小运算的图像数据量）。所以输出的图像为 112*160
        # 这里的shape的值，直接影响到reset时，返回np.stack()的维度。需要特别注意
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(112, 160, 3), dtype=np.uint8)

        self.reset_round = reset_round
        self.rendering = rendering
        
        self.full_hp = 223
        self.full_sp = 223
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp
        self.prev_player_sp = self.full_sp
        self.prev_oppont_sp = self.full_sp

        self.totalScoreLianXuCount = 0  #连续得分
        self.COST_SWITCH_LIANXU = False

    def _stack_observation(self):
        return np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)
    
    def reset(self):
        observation = self.env.reset()
        
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp
        self.prev_player_sp = self.full_sp
        self.prev_oppont_sp = self.full_sp

        # 这里处理的是隔行取值（每一帧图像，取隔行隔列的像素值）用于减小图像数据的运算量
        self.frame_stack.clear()
        for _ in range(self.num_frames):
            self.frame_stack.append(observation[::2, ::2, :])

        # print("线程{}：执行reset，已重置".format(threading.get_ident()))

        # 这里是将3帧（隔行取值后的图像）图像堆叠成为一张完整的RGB图像，用于后续的计算。
        return np.stack([self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1)

    def step(self, action):
          
        # 执行原始环境的步骤
        observation, reward, done, info = self.env.step(action)
        self.frame_stack.append(observation[::2, ::2, :])

        custom_done = False

        # Render the game if rendering flag is set to True.
        if self.rendering:
            self.env.render()
            time.sleep(0.005)
        
        for _ in range(self.num_step_frames - 1):
            # Keep the button pressed for (num_step_frames - 1) frames.
            obs, _reward, _done, info = self.env.step(action)
            self.frame_stack.append(obs[::2, ::2, :])
            if self.rendering:
                self.env.render()
                time.sleep(0.005)
      

        # 敌我双方的HP和SP
        curr_agent_hp = info['agent_hp']
        curr_agent_sp = info['agent_sp']
        curr_enemy_hp = info['enemy_hp']
        curr_enemy_sp = info['enemy_sp']

        # Game is over and player loses.
        if curr_agent_hp < 0:
            # Use the remaining health points of opponent as penalty. 
            # If the opponent also has negative health points, it's a even game and the reward is +1.
            custom_reward = -math.pow(self.full_hp, (curr_enemy_hp + 1) / (self.full_hp + 1))    
            custom_done = True

            print(f"Trainning: Game is over and player loses. custom_reward={custom_reward}")

        # Game is over and player wins.
        elif curr_enemy_hp < 0:

            # Use the remaining health points of player as reward.
            # Multiply by reward_coeff to make the reward larger than the penalty to avoid cowardice of agent.
            # custom_reward = curr_player_health * self.reward_coeff 

            # custom_reward = math.pow(self.full_hp, (5940 - self.total_timesteps) / 5940) * self.reward_coeff # Use the remaining time steps as reward.
            custom_reward = math.pow(self.full_hp, (curr_agent_hp + 1) / (self.full_hp + 1)) * self.reward_coeff
            custom_done = True

            print(f"Trainning: Game is over and player wins. custom_reward={custom_reward}")

        # While the fighting is still going on
        else:
            makeDamageReward = self.reward_coeff * (self.prev_oppont_health - curr_enemy_hp) - (self.prev_player_health - curr_agent_hp)

            if makeDamageReward != 0:
                if self.COST_SWITCH_LIANXU == False:
                    self.COST_SWITCH_LIANXU = True
                # 如果N步之内连续得分，则奖励翻倍
                if self.totalScoreLianXuCount > 0 and self.totalScoreLianXuCount < COST_TOTALSCORELIANXUCOUNT:
                    if makeDamageReward < 0:
                        makeDamageReward *= 1.14
                    else:
                        makeDamageReward *= 1.18
                    self.totalScoreLianXuCount = 0
                    self.COST_SWITCH_LIANXU = False;
                    print("{}次内连续得分，或失分，reward翻倍，{}分".format(COST_TOTALSCORELIANXUCOUNT, makeDamageReward))

            # # 如果是消耗sp攻击对方，则奖励翻倍
            if curr_agent_sp < self.prev_player_sp:
                makeDamageReward = self.reward_coeff * (self.prev_player_sp - curr_agent_sp) * 0.3
                print(f"通过技能攻击，奖励翻倍，{makeDamageReward}分")


            self.prev_player_health = curr_agent_hp
            self.prev_oppont_health = curr_enemy_hp
            self.prev_player_sp = curr_agent_sp
            self.prev_oppont_sp = curr_enemy_sp

            # 把SP值也作为奖励因素，如果我方SP值大于敌方，则奖励，否则惩罚
            sp_adjust = self.reward_coeff * (curr_agent_sp - self.prev_player_sp) * 0.15
            # sp_adjust = 0
            
            # 什么事情都不做 就连续扣分
            doNothingReward = 0 - self.reward_coeff * 0.5
            # doNothingReward = 0

            # 本step最终奖励分值
            custom_reward = makeDamageReward + sp_adjust + doNothingReward

            if custom_reward != 0:
                print(f'Trainning: custom_reward={custom_reward} =  | sp_adjust={sp_adjust} | doNothingReward={doNothingReward}')
            custom_done = False
        
        # When reset_round flag is set to False (never reset), the session should always keep going.
        if not self.reset_round:
            custom_done = False

        if self.COST_SWITCH_LIANXU == True:
            self.totalScoreLianXuCount += 1

        if self.totalScoreLianXuCount >= COST_TOTALSCORELIANXUCOUNT:
            self.totalScoreLianXuCount = 0
            self.COST_SWITCH_LIANXU = False;
        
        # Max reward is 6 * full_hp = 1054 (damage * 3 + winning_reward * 3) norm_coefficient = 0.001
        return self._stack_observation(), custom_reward, custom_done, info # reward normalization