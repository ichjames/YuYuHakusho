import gym
import numpy as np
import retro

lstCombos = [
            [],                         # 0: 无操作
            ['RIGHT'],                  # 1: 向右走
            ['LEFT'],                   # 2: 向左走
            ['DOWN'],                   # 3: 蹲下
            ['UP'],                     # 4: 跳
            ['B'],                      # 5: 轻攻击
            ['C'],                      # 6: 防御
            ['A'],                      # 7: 重攻击
            ['DOWN','RIGHT', 'A'],      # 8: 下+右+重攻击
            # ['A','B'],                  # 9: 重攻击+轻攻击
            # ['DOWN','DOWN', 'A'],       # 10: 下+右+重攻击
            # ['DOWN','UP', 'A'],       # 10: 下+右+重攻击
            # ['DOWN','RIGHT','DOWN','RIGHT', 'A'],       # 10: 下+右+重攻击
            # ['RIGHT','DOWN','RIGHT','DOWN', 'A'],       # 10: 下+右+重攻击
        ]

class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()

class YuYuHakushoDiscretizer(Discretizer):
    """
    Use YuYuHakusho-specific discrete actions
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """
    def __init__(self, env):
        super().__init__(env=env, combos=lstCombos)