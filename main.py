import numpy as np

import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2352921 import Policy2352921

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
# NUM_EPISODES = np.random.randint(low = 1, high = 10)
NUM_EPISODES = 100

if __name__ == "__main__":
    # Reset the environment
    # random_seed = np.random.randint(low = 0, high = 100)
    # observation, info = env.reset(seed=random_seed)
    observation, info = env.reset(seed=42)

    # # Test GreedyPolicy
    # print("\n========= Running Greedy Policy =========\n")
    # gd_policy = GreedyPolicy()
    # ep = 0
    # while ep < NUM_EPISODES:
    #     action = gd_policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     if terminated or truncated:
    #         print(info)
    #         observation, info = env.reset(seed=ep)
    #         ep += 1

    # # Reset the environment
    # observation, info = env.reset(seed=random_seed)

    # # Test RandomPolicy
    # print("\n========= Running Random Policy =========\n")
    # rd_policy = RandomPolicy()
    # ep = 0
    # while ep < NUM_EPISODES:
    #     action = rd_policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     if terminated or truncated:
    #         print(info)
    #         observation, info = env.reset(seed=ep)
    #         ep += 1

    # Reset the environment
    # observation, info = env.reset(seed=random_seed)

    # # Test Policy2352921
    # print("\n========= Running Column Generation Policy =========\n")
    # policy = Policy2352921()
    # ep = 0
    # while ep < NUM_EPISODES:    
    #     action = policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     if terminated or truncated:
    #         print(info)
    #         observation, info = env.reset(seed=ep)
    #         ep += 1

    # Uncomment the following code to test your policy
    # Reset the environment
    observation, info = env.reset(seed=42)
    print(info)

    policy2352921 = Policy2352921()
    for _ in range(200):
        action = policy2352921.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        print(info)

        if terminated or truncated:
            observation, info = env.reset()

env.close()
