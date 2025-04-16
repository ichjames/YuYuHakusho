import os
import time 
import retro
from stable_baselines3 import PPO
from custYuYuHakushoWrapper import CustYuYuHakushoWrapper

RESET_ROUND = True  # Whether to reset the round when fight is over. 
RENDERING = True    # Whether to render the game screen.

MODEL_NAME = r"ppo_yuyuhakusho_final" # Specify the model file to load. Model "ppo_ryu_2500000_steps_updated" is capable of beating the final stage (Bison) of the game.

# Model notes:
# ppo_balloon_final: Just beginning to overfit state, generalizable but not quite capable.
# ppo_balloon_93750_steps: Approaching the final overfitted state, cannot dominate first round but partially generalizable. High chance of beating the final stage.
# ppo_balloon_62500_steps: Near the final overfitted state, almost dominate first round but barely generalizable.
# ppo_balloon_31250_steps: Overfitted, dominates first round but not generalizable. 

RANDOM_ACTION = False
NUM_EPISODES = 10 # Make sure NUM_EPISODES >= 3 if you set RESET_ROUND to False to see the whole final stage game.
MODEL_DIR = r"trained_models/"

def make_env(game, state):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE
        )
        env = CustYuYuHakushoWrapper(env, reset_round=RESET_ROUND, rendering=RENDERING,limit_step=False)
        return env
    return _init

game = "YuYuHakusho-Genesis"
state = "Level1"
env = make_env(game, state=state)()
# model = PPO("CnnPolicy", env)

if not RANDOM_ACTION:
    model = PPO.load(os.path.join(MODEL_DIR, MODEL_NAME), env=env)

obs = env.reset()
done = False

num_episodes = NUM_EPISODES
episode_reward_sum = 0
num_victory = 0

print("\nFighting Begins!\n")

for _ in range(num_episodes):
    done = False
    
    if RESET_ROUND:
        obs = env.reset()

    total_reward = 0

    while not done:
        timestamp = time.time()

        if RANDOM_ACTION:
            obs, reward, done, info = env.step(env.action_space.sample())
        else:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)

        if reward != 0:
            total_reward += reward
            # print("Total reward: {:.3f}, Reward: {:.3f}, lives: {}, score:{}".format(total_reward,reward, info['agent_hp'], info['enemy_hp']))
        
        if info['agent_hp'] < 0:
            done = True
        
    print("Total reward: {}\n".format(total_reward))
    episode_reward_sum += total_reward

    if not RESET_ROUND:
        if info['agent_hp'] < 0:
        # Inter scene transition. Do nothing.
            # obs, reward, done, info = env.step([0] * 12)
            # env.render()
            obs = env.reset()

env.close()
print("Winning rate: {}".format(1.0 * num_victory / num_episodes))
if RANDOM_ACTION:
    print("Average reward for random action: {}".format(episode_reward_sum/num_episodes))
else:
    print("Average reward for {}: {}".format(MODEL_NAME, episode_reward_sum/num_episodes))