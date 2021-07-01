import numpy as np

def sample_trajectory(env, policy, max_path_length):
  obs = []
  acs = []
  rewards = []
  path_length = 0

  ob = env.reset()
  while True:
    obs.append(ob)
    ac = policy.get_action(ob)
    acs.append(ac)
    ob, reward, done, _ = env.step(ac)
    rewards.append(reward)
    path_length += 1

    if (path_length >= max_path_length) or done:
      break
  
  return {"obs": np.array(obs),
          "acs": np.array(acs),
          "rewards": np.array(rewards)}

def sample_n_trajectories(env, policy, max_path_length, n):
  obs = []
  acs = []
  rewards = []
  for _ in range(n):
    trajectory = sample_trajectory(env, policy, max_path_length)
    obs.append(trajectory["obs"])
    acs.append(trajectory["acs"])
    rewards.append(trajectory["rewards"])
  trajectories = {"obs": obs,
                  "acs": acs,
                  "rewards": rewards}
  return trajectories

def expected_rewards_sum(env, policy, max_path_length, n):
  trajectories = sample_n_trajectories(env, policy, max_path_length, n)
  expected_reward_sum = 0
  for i in range(n):
    expected_reward_sum += trajectories["rewards"][i].sum()
  expected_reward_sum /= n
  return expected_reward_sum











