import retro
import gym
import numpy as np
import time

# 1. 设置环境（确保你已安装 retro 并导入了对应游戏）
# 游戏示例：Street Fighter II
ENV_NAME = "YuYuHakusho-Genesis"
state = "Level_Trainning"

# 2. 定义动作映射
# 获取游戏的动作空间（通常是二进制向量，每个位代表一个按钮）
# 我们将创建一个字典，将字符串指令映射为按钮的二进制向量
BUTTONS = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']

def get_action_from_buttons(button_list):
    """
    根据按钮名称列表生成动作向量
    例如：['DOWN', 'RIGHT'] -> [0,0,0,0,0,1,0,1,0,0,0,0]
    """
    action = [0] * len(BUTTONS)
    for btn in button_list:
        if btn in BUTTONS:
            action[BUTTONS.index(btn)] = 1
    return action

# 预定义常用动作
ACTIONS = {
    "NOTHING": [0] * len(BUTTONS),
    "LEFT": get_action_from_buttons(['LEFT']),
    "RIGHT": get_action_from_buttons(['RIGHT']),
    "UP": get_action_from_buttons(['UP']),
    "DOWN": get_action_from_buttons(['DOWN']),
    "A": get_action_from_buttons(['A']),
    "B": get_action_from_buttons(['B']),
    "C": get_action_from_buttons(['C']),
    # 组合键
    "DOWN+RIGHT": get_action_from_buttons(['DOWN', 'RIGHT']),
    "DOWN+LEFT": get_action_from_buttons(['DOWN', 'LEFT']),
    "UP+RIGHT": get_action_from_buttons(['UP', 'RIGHT']),
    "UP+LEFT": get_action_from_buttons(['UP', 'LEFT']),
}

# 3. 定义搓招序列（例如波动拳：↓ ↘ → + A）
SHORYUKEN_SEQUENCE = ["DOWN", "DOWN+RIGHT", "RIGHT","DOWN", "DOWN+RIGHT", "RIGHT", "A"]
HADOUKEN_SEQUENCE = ["DOWN", "DOWN+RIGHT", "RIGHT", "A"]  # 波动拳
TATSUMAKI_SEQUENCE = ["DOWN", "DOWN+LEFT", "LEFT", "A"]   # 龙卷旋风脚

# 4. 搓招执行函数
def perform_special_move(env, sequence, hold_frames=5):
    """
    执行一个搓招序列
    :param env: retro 环境
    :param sequence: 指令列表，如 ["DOWN", "DOWN+RIGHT", "RIGHT", "A"]
    :param hold_frames: 每个动作保持的帧数
    """
    for move in sequence:
        if move not in ACTIONS:
            print(f"未知动作: {move}")
            continue
        action = ACTIONS[move]
        for _ in range(hold_frames):
            obs, reward, done, info = env.step(action)
            # 可选：渲染环境观察搓招效果
            env.render()
            time.sleep(0.01)  # 控制速度便于观察
        # 可选：在每个动作之间加一点延迟（更像真实搓招）
        time.sleep(0.05)

# 5. 主循环示例
def main():
    env = retro.make(game=ENV_NAME, state=state)
    obs = env.reset()
    
    print("环境已启动，准备执行搓招...")
    time.sleep(2)

    # 示例：连续释放 5 次波动拳
    for i in range(50):
        print(f"执行第 {i+1} 次波动拳...")
        perform_special_move(env, SHORYUKEN_SEQUENCE, hold_frames=6)  # 每步约 6 帧
        # time.sleep(1)  # 搓招后暂停一下

    print("搓招演示结束。")
    env.close()

if __name__ == "__main__":
    main()