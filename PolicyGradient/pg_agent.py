import numpy as np
from PolicyGradient.MLP_policy_PG import MLPPolicyPG
from PolicyGradient.MLP_critic import MLPCritic

# params = {"actor_params": {---}, "critic_params": {---},
#          "baseline_enable": <Bool>}
class PGAgent():
  def __init__(self, params):
    self.actor = MLPPolicyPG(params["actor_params"])
    self.baseline_enable = params["baseline_enable"]
    if self.baseline_enable:
      self.critic = MLPCritic(params["critic_params"])

  def train(self, trajectories):
    q_values = self.estimate_q_values(trajectories["rewards"])
    if self.baseline_enable:
      self.critic.update(trajectories["obs"], q_values)
    advantages = self.estimate_advantages(q_values, trajectories["obs"])
    self.actor.update(trajectories, advantages)

  def estimate_q_values(self, rewards):
    N = len(rewards) # Number of Trajectories
    q_values = []
    for i in range(N):
      q_values_i = []
      rewards_i = rewards[i]
      l = rewards_i.shape[0] # Trajectory length
      for j in range(l):
        q_values_i.append(rewards_i[j:].sum()) 
      q_values.append(np.array(q_values_i))
    return q_values

  def estimate_advantages(self, q_values, obs):
    N = len(obs) # Number of Trajectories
    advantages = []
    for i in range(N):
      advantages_i = []
      q_values_i = q_values[i]
      obs_i = obs[i]
      l = obs_i.shape[0] # Trajectory length
      for j in range(l):
        if self.baseline_enable:
          value_ij = float(self.critic.get_value(obs_i[j]))
          # print(f"Q-Value: {q_values_i[j]}, Value: {value_ij}")
          advantages_i.append(q_values_i[j] - float(value_ij))
        else:
          advantages_i.append(q_values_i[j])
      advantages.append(np.array(advantages_i))
    return advantages







